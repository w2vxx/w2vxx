#include <memory>
#include <string>
#include <thread>
#include <chrono>
#include "simple_profiler.h"
#include "cbow_command_line_parameters.h"
#include "original_word2vec_vocabulary.h"
#include "original_word2vec_le_provider.h"
#include "trainer.h"



class CbowTrainer_Mikolov : public CustomTrainer
{
public:
  // конструктор
  CbowTrainer_Mikolov( std::shared_ptr< CustomLearningExampleProvider> learning_example_provider,
               std::shared_ptr< CustomVocabulary > words_vocabulary,
               std::shared_ptr< CustomVocabulary > contexts_vocabulary,
               size_t embedding_size = 100,
               size_t epochs = 5,
               float learning_rate = 0.05,
               const std::string& optimization = "ns",
               size_t negative_count = 5 )
  : CustomTrainer(learning_example_provider, words_vocabulary, contexts_vocabulary, contexts_vocabulary, words_vocabulary, embedding_size, epochs, learning_rate, optimization, negative_count)
  {
    if (optimization_algo == loaNegativeSampling)
      InitUnigramTable_w();
  }
  // деструктор
  virtual ~CbowTrainer_Mikolov()
  {
  }
  // функция, реализующая модель обучения cbow
  void learning_model(LearningExample& le, float *neu1, float *neu1e)
  {
    if (le.context.size() == 0) return;
    // зануляем текущие значения выходов нейронов проекционного слоя и текущие значения ошибок
    std::fill(neu1, neu1+layer1_size, 0.0);
    std::fill(neu1e, neu1e+layer1_size, 0.0);
    // вычисляем выход проекционного слоя ( in --> hidden )
    size_t cw = 0;
    for (auto&& ctx_idx : le.context)
    {
      size_t ctxOffset = ctx_idx * layer1_size;
      for (size_t c = 0; c < layer1_size; ++c)
        neu1[c] += syn0[c + ctxOffset];
      ++cw;
    }
    for (size_t c = 0; c < layer1_size; ++c)
      neu1[c] /= cw;
    //
    if (optimization_algo == loaHierarchicalSoftmax)  // hierarchical softmax
    {
      auto&& current_word_data = w_vocabulary->idx_to_data(le.word);
      for (size_t d = 0; d < current_word_data.huffman_code_float.size(); ++d)
      {
        float f = 0;
        long long l2 = current_word_data.huffman_path[d] * layer1_size;
        // Propagate hidden -> output
        for (size_t c = 0; c < layer1_size; ++c)
          f += neu1[c] * syn1[c + l2];
        if (f <= -MAX_EXP) continue;
        else if (f >= MAX_EXP) continue;
        else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
        // 'g' is the gradient multiplied by the learning rate
        float g = (1.0 - current_word_data.huffman_code_float[d] - f) * alpha;
        // Propagate errors output -> hidden
        for (size_t c = 0; c < layer1_size; ++c)
          neu1e[c] += g * syn1[c + l2];
        // Learn weights hidden -> output
        for (size_t c = 0; c < layer1_size; ++c)
          syn1[c + l2] += g * neu1[c];
      }
    }
    else if (optimization_algo == loaNegativeSampling) // negative sampling
    {
      for (size_t d = 0; d < negative + 1; ++d)
      {
        size_t target;
        int label; // знаковое целое
        if (d == 0)
        {
          target = le.word;
          label = 1;
        }
        else
        {
          next_random_ns = next_random_ns * (unsigned long long)25214903917 + 11;
          target = table[(next_random_ns >> 16) % table_size];
          if (target == 0) target = next_random_ns % (w_vocabulary->size() - 1) + 1;
          if (target == le.word) continue;
          label = 0;
        }
        long long l2 = target * layer1_size;
        float f = 0;
        for (size_t c = 0; c < layer1_size; ++c)
          f += neu1[c] * syn1neg[c + l2];
        float g = 0;
        if (f > MAX_EXP) g = (label - 1) * alpha;
        else if (f < -MAX_EXP) g = (label - 0) * alpha;
        else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
        for (size_t c = 0; c < layer1_size; ++c)
          neu1e[c] += g * syn1neg[c + l2];
        for (size_t c = 0; c < layer1_size; ++c)
          syn1neg[c + l2] += g * neu1[c];
      }
    }
    // hidden -> in
    for (auto&& ctx_idx : le.context)
    {
      size_t ctxOffset = ctx_idx * layer1_size;
      for (size_t c = 0; c < layer1_size; ++c)
        syn0[c + ctxOffset] += neu1e[c];
    }
  } // method-end
};





int main(int argc, char **argv)
{
  SimpleProfiler global_profiler;

  // выполняем разбор параметров командной строки
  CbowCommandLineParameters cmdLineParams;
  cmdLineParams.parse(argc, argv);
  cmdLineParams.dbg_cout();

  // загрузка словаря
  std::shared_ptr< OriginalWord2VecVocabulary> v = std::make_shared< OriginalWord2VecVocabulary>();
  if ( !v->load( cmdLineParams.getAsString("-words-vocab") ) )
    return -1;

  // создание поставщика обучающих примеров
  // к моменту создания "поставщика обучающих примеров" словарь должен быть загружен (в частности, используется cn_sum())
  std::shared_ptr< CustomLearningExampleProvider> lep = std::make_shared< OriginalWord2VecLearningExampleProvider > ( cmdLineParams.getAsString("-train"),
                                                                                                                      cmdLineParams.getAsInt("-threads"),
                                                                                                                      cmdLineParams.getAsInt("-window"),
                                                                                                                      cmdLineParams.getAsFloat("-sample"),
                                                                                                                      v );

  // создаем объект, организующий обучение
  CbowTrainer_Mikolov trainer( lep, v , v,
                       cmdLineParams.getAsInt("-size"),
                       cmdLineParams.getAsInt("-iter"),
                       cmdLineParams.getAsFloat("-alpha"),
                       cmdLineParams.getAsString("-optimization"),
                       cmdLineParams.getAsFloat("-negative"));

  // инициализация нейросети
  trainer.init_net();

  // запускаем потоки, осуществляющие обучение
  size_t threads_count = cmdLineParams.getAsInt("-threads");
  std::vector<std::thread> threads_vec;
  threads_vec.reserve(threads_count);
  for (size_t i = 0; i < threads_count; ++i)
    threads_vec.emplace_back(&CustomTrainer::train_entry_point, &trainer, i);
  for (size_t i = 0; i < threads_count; ++i)
    threads_vec[i].join();

  // сохраняем вычисленные вектора в файл
  trainer.saveEmbeddings( cmdLineParams.getAsString("-output") );
  if (cmdLineParams.isDefined("-backup"))
    trainer.backup( cmdLineParams.getAsString("-backup") );

  return 0;
}
