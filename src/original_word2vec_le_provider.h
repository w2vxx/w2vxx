#ifndef ORIGINAL_WORD2VEC_LE_PROVIDER_H_
#define ORIGINAL_WORD2VEC_LE_PROVIDER_H_

#include <string>
#include <vector>
#include <optional>
#include <memory>
#include <limits>
#include <math.h>
#include "learning_example_provider.h"
#include "original_word2vec_vocabulary.h"


const size_t MAX_SENTENCE_LENGTH = 1000;
const size_t MAX_STRING = 100;


// информация, описывающая рабочий контекст одного потока управления (thread)
struct ThreadEnvironment_w2v
{
  FILE* fi;                              // хэндлер файла, содержащего обучающее множество (открывается с позиции, рассчитанной для данного потока управления).
  std::vector<size_t> sentence;          // последнее считанное предложение
  int position_in_sentence;              // текущая позиция в предложении
  unsigned long long next_random;        // поле для вычисления случайных величин
  unsigned long long words_count;        // количество прочитанных словарных слов
  ThreadEnvironment_w2v()
  : fi(NULL)
  , position_in_sentence(-1)
  , next_random(0)
  , words_count(0)
  {
    sentence.reserve(MAX_SENTENCE_LENGTH);
  }
};


class OriginalWord2VecLearningExampleProvider : public CustomLearningExampleProvider
{
public:
  // конструктор
  OriginalWord2VecLearningExampleProvider(const std::string& trainFilename, size_t threadsCount, size_t ctxWindow, float sampleThreshold, std::shared_ptr< OriginalWord2VecVocabulary> words_vocabulary)
  : CustomLearningExampleProvider(threadsCount)
  , train_filename(trainFilename)
  , train_file_size(0)
  , window(ctxWindow)
  , sample(sampleThreshold)
  , vocabulary(words_vocabulary)
  , train_words(0)
  {
    thread_environment.resize(threads_count);
    for (size_t i = 0; i < threads_count; ++i)
      thread_environment[i].next_random = i;
    if ( vocabulary )
      train_words = vocabulary->cn_sum();
    try
    {
      train_file_size = get_file_size(train_filename);
    } catch (const std::runtime_error& e) {
      std::cout << "LearningExampleProvider can't get file size for: " << train_filename << "\n  " << e.what() << std::endl;
    }
  } // constructor-end
  // деструктор
  virtual ~OriginalWord2VecLearningExampleProvider()
  {
  }
  // подготовительные действия, выполняемые перед каждой эпохой обучения
  bool epoch_prepare(size_t threadIndex)
  {
    auto& t_environment = thread_environment[threadIndex];
    t_environment.fi = fopen(train_filename.c_str(), "rb");
    if ( t_environment.fi == NULL )
    {
      std::cout << "LearningExampleProvider: epoch prepare error: " << std::strerror(errno) << std::endl;
      return false;
    }
    int succ = fseek(t_environment.fi, train_file_size / threads_count * threadIndex, SEEK_SET);
    if (succ != 0)
    {
      std::cout << "LearningExampleProvider: epoch prepare error: " << std::strerror(errno) << std::endl;
      return false;
    }
    t_environment.sentence.clear();
    t_environment.position_in_sentence = 0;
    t_environment.words_count = 0;
    return true;
  } // method-end
  // заключительные действия, выполняемые после каждой эпохи обучения
  bool epoch_unprepare(size_t threadIndex)
  {
    auto& t_environment = thread_environment[threadIndex];
    fclose( t_environment.fi );
    return true;
  }
  // получение очередного обучающего примера
  std::optional<LearningExample> get(size_t threadIndex)
  {
    auto& t_environment = thread_environment[threadIndex];
    if (t_environment.sentence.empty())
      read_sentence(threadIndex);
    if (t_environment.sentence.empty())  // это признак конца эпохи
      return std::nullopt;
    LearningExample result;
    result.word = t_environment.sentence[t_environment.position_in_sentence];
    t_environment.next_random = t_environment.next_random * (unsigned long long)25214903917 + 11;
    auto current_window = t_environment.next_random % window;

    // можно сделать так:
    // ++current_window;  // current_window попадает в диапазон [1; window]
    // но для удобства сопоставления результатов с оригинальным word2vec сделаем по аналогии с оригиналом
    current_window = window - current_window;

    for (int i = t_environment.position_in_sentence - current_window, iEnd = t_environment.position_in_sentence + current_window; i <= iEnd; ++i)
    {
      if ( i < 0 ) continue;
      if ( i == t_environment.position_in_sentence ) continue; // пропускаем само слово, для которого ищем контекст
      if ( i >= static_cast<int>(t_environment.sentence.size()) ) break;
      result.context.emplace_back(t_environment.sentence[i]);
    }
    ++t_environment.position_in_sentence;
    if ( t_environment.position_in_sentence == static_cast<int>(t_environment.sentence.size()) )
      t_environment.sentence.clear();
    return result;
  } // method-end
  // получение количества слов, фактически считанных из обучающего множества (т.е. без учета сабсэмплинга)
  uint64_t getWordsCount(size_t threadIndex) const
  {
    return thread_environment[threadIndex].words_count;
  }
private:
  // информация, описывающая рабочие контексты потоков управления (thread)
  std::vector<ThreadEnvironment_w2v> thread_environment;
  // имя "тренировочного" файла
  std::string train_filename;
  // размер тренировочного файла
  uint64_t train_file_size;
  // максимальный размер контекстного окна
  size_t window;
  // порог для алгоритма сэмплирования (subsampling)
  float sample;
  // словарь
  std::shared_ptr< OriginalWord2VecVocabulary> vocabulary;
  // количество слов в обучающем множестве (приблизительно, т.к. могло быть подрезание по порогу частоты при построении словаря)
  uint64_t train_words;

  // чтение одного слова из файла в предположении, что разделителями служат space + tab + EOL
  void read_word(FILE *fin, std::string& word)
  {
    word.clear();
    size_t a = 0;
    while ( !feof(fin) )
    {
      int ch = fgetc(fin);
      if (ch == 13) continue;   //  \r
      if ((ch == ' ') || (ch == '\t') || (ch == '\n'))
      {
        // если есть прочитанный фрагмент слова
        if (a > 0)
        {
          if (ch == '\n')
            ungetc(ch, fin);
          break;
        }
        // если прочитанного фрагмента слова нет
        if (ch == '\n')
        {
          word = "</s>";
          return;
        }
        else
          continue;
      } // if (delimiter) ...
      if ( a < MAX_STRING )
        word.push_back( ch );
      ++a;
    }
  } // method-end

  // чтение одного предложения
  void read_sentence(size_t threadIndex)
  {
    auto& t_environment = thread_environment[threadIndex];
    t_environment.sentence.clear();
    t_environment.position_in_sentence = 0;
    std::string word;
    word.reserve(MAX_STRING);
    while (true)
    {
      read_word(t_environment.fi, word);
      if ( feof(t_environment.fi) ) break;
      auto wordIdx = vocabulary->word_to_idx(word);
      if (wordIdx == std::numeric_limits<size_t>::max()) continue;  // несловарное слово
      ++t_environment.words_count;
      if (wordIdx == 0) break;                                      // маркер конца предложения/параграфа
      // The subsampling randomly discards frequent words while keeping the ranking same
      if (sample > 0)
      {
        auto&& dict_word = vocabulary->idx_to_data(wordIdx);
        float ran = (sqrt(dict_word.cn / (sample * train_words)) + 1) * (sample * train_words) / dict_word.cn;
        t_environment.next_random = t_environment.next_random * (unsigned long long)25214903917 + 11;
        if (ran < (t_environment.next_random & 0xFFFF) / (float)65536) continue;
      }
      t_environment.sentence.push_back( wordIdx );
      if (t_environment.sentence.size() >= MAX_SENTENCE_LENGTH) break;
    }
    // не настал ли конец эпохи?
    if ( feof(t_environment.fi) || (t_environment.words_count > train_words / threads_count) )
    {
      t_environment.sentence.clear();
      t_environment.position_in_sentence = 0;
    }
  } // method-end
};


#endif /* ORIGINAL_WORD2VEC_LE_PROVIDER_H_ */
