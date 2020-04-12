#ifndef TRAINER_H_
#define TRAINER_H_

#include <memory>
#include <string>
#include <chrono>
#include <iostream>

#ifdef _MSC_VER
  #define posix_memalign(p, a, s) (((*(p)) = _aligned_malloc((s), (a))), *(p) ? 0 : errno)
  #define free_aligned(p) _aligned_free((p))
#else
  #define free_aligned(p) free((p))
#endif


#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6

enum LearningOptimizationAlgo
{
  loaUndefined,
  loaHierarchicalSoftmax,
  loaNegativeSampling
};


// хранит общие параметры и данные для всех потоков
// реализует общую логику обучения (которая затем специализируется для cbow и skip-gram, соответственно)
class CustomTrainer
{
public:
  // конструктор
  CustomTrainer( std::shared_ptr< CustomLearningExampleProvider> learning_example_provider,
                 std::shared_ptr< CustomVocabulary > words_vocabulary,
                 std::shared_ptr< CustomVocabulary > contexts_vocabulary,
                 std::shared_ptr< CustomVocabulary > input_vocabulary,
                 std::shared_ptr< CustomVocabulary > output_vocabulary,
                 size_t embedding_size,
                 size_t epochs,
                 float learning_rate,
                 const std::string& optimization,
                 size_t negative_count )
  : lep(learning_example_provider)
  , w_vocabulary(words_vocabulary)
  , c_vocabulary(contexts_vocabulary)
  , in_vocabulary(input_vocabulary)
  , out_vocabulary(output_vocabulary)
  , layer1_size(embedding_size)
  , epoch_count(epochs)
  , alpha(learning_rate)
  , starting_alpha(learning_rate)
  , optimization_algo( loaUndefined )
  , negative(negative_count)
  , next_random_ns(0)
  {
    if (optimization == "hs")
      optimization_algo = loaHierarchicalSoftmax;
    else if (optimization == "ns")
      optimization_algo = loaNegativeSampling;
    // предварительный табличный расчет для логистической функции
    expTable = (float *)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));
    for (size_t i = 0; i < EXP_TABLE_SIZE; i++) {
      expTable[i] = exp((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
      expTable[i] = expTable[i] / (expTable[i] + 1);                    // Precompute f(x) = x / (x + 1)
    }
    // запомним количество обучающих примеров
    train_words = w_vocabulary->cn_sum();
  }
  // деструктор
  virtual ~CustomTrainer()
  {
    free(expTable);
    if (syn0)
      free_aligned(syn0);
    if (syn1)
      free_aligned(syn1);
    if (table)
      free(table);
  }
  // функция инициализации нейросети
  void init_net()
  {
    size_t in_vocab_size = in_vocabulary->size();
    size_t out_vocab_size = out_vocabulary->size();
    long long ap;
    unsigned long long next_random = 1;

    ap = posix_memalign((void **)&syn0, 128, (long long)in_vocab_size * layer1_size * sizeof(float));
    if (syn0 == nullptr || ap != 0) {std::cerr << "Memory allocation failed" << std::endl; exit(1);}
    for (size_t a = 0; a < in_vocab_size; ++a)
      for (size_t b = 0; b < layer1_size; ++b)
      {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (float)65536) - 0.5) / layer1_size;
      }

    ap = posix_memalign((void **)&syn1, 128, (long long)out_vocab_size * layer1_size * sizeof(float));
    if (syn1 == nullptr || ap != 0) {std::cerr << "Memory allocation failed" << std::endl; exit(1);}
    for (size_t a = 0; a < out_vocab_size; ++a)
      for (size_t b = 0; b < layer1_size; ++b)
        syn1[a * layer1_size + b] = 0;

    if (optimization_algo == loaHierarchicalSoftmax) // hierarchical softmax
      out_vocabulary->buildHuffmanTree();
    else if (optimization_algo == loaNegativeSampling) // negative sampling
    {}
    else
    {
      std::cerr << "Unknown learning optimization algorithm" << std::endl;
      exit(1);
    }
    start_learning_tp = std::chrono::steady_clock::now();
  } // method-end
  // обобщенная процедура обучения (точка входа для потоков)
  void train_entry_point( size_t thread_idx )
  {
    next_random_ns = thread_idx;
    // выделение памяти для хранения выхода скрытого слоя и величины ошибки
    float *neu1 = (float *)calloc(layer1_size, sizeof(float));
    float *neu1e = (float *)calloc(layer1_size, sizeof(float));
    // цикл по эпохам
    for (size_t epochIdx = 0; epochIdx < epoch_count; ++epochIdx)
    {
      if ( !lep->epoch_prepare(thread_idx) )
        return;
      long long word_count = 0, last_word_count = 0;
      // цикл по словам
      while (true)
      {
        // вывод прогресс-сообщений
        // и корректировка коэффициента скорости обучения (alpha)
        if (word_count - last_word_count > 10000)
        {
          word_count_actual += (word_count - last_word_count);
          last_word_count = word_count;
          //if ( debug_mode > 1 )
          {
            std::chrono::steady_clock::time_point current_learning_tp = std::chrono::steady_clock::now();
            std::chrono::duration< double, std::ratio<1> > learning_seconds = current_learning_tp - start_learning_tp;
            printf("%cAlpha: %f  Progress: %.2f%%  Words/sec: %.2fk  ", 13, alpha,
              word_count_actual / (float)(epoch_count * train_words + 1) * 100,
              word_count_actual / (learning_seconds.count() * 1000) );
            fflush(stdout);
          }
          alpha = starting_alpha * (1 - word_count_actual / (float)(epoch_count * train_words + 1));
          if ( alpha < starting_alpha * 0.0001 )
            alpha = starting_alpha * 0.0001;
        }
        // читаем очередной обучающий пример
        auto learning_example = lep->get(thread_idx);
        word_count = lep->getWordsCount(thread_idx);
        if (!learning_example) break; // признак окончания эпохи (все обучающие примеры перебраны)
        // используем обучающий пример для обучения нейросети
        learning_model( learning_example.value(), neu1, neu1e );
      } // for all learning examples
      word_count_actual += (word_count - last_word_count);
      if ( !lep->epoch_unprepare(thread_idx) )
        return;
    } // for all epochs
    free(neu1);
    free(neu1e);
  } // method-end: train_entry_point
  // функция, реализующая конкретную модель обучения
  virtual void learning_model(const LearningExample& le, float *neu1, float *neu1e ) = 0;
  // функция, реализующая сохранение эмбеддингов
  void saveEmbeddings(const std::string& filename) const
  {
    FILE *fo = fopen(filename.c_str(), "wb");
    fprintf(fo, "%lu %lu\n", w_vocabulary->size(), layer1_size);
    saveEmbeddingsBin_helper(fo, w_vocabulary, syn0);
    fclose(fo);
  } // method-end
  // функция сохранения обоих весовых матриц в файл
  void backup(const std::string& filename) const
  {
    FILE *fo = fopen(filename.c_str(), "wb");
    fprintf(fo, "%lu %lu\n", w_vocabulary->size(), layer1_size);
    // сохраняем весовую матрицу между входным и скрытым слоем
    saveEmbeddingsBin_helper(fo, w_vocabulary, syn0);
    // сохраняем весовую матрицу между скрытым и выходным слоем
    saveEmbeddingsBin_helper(fo, w_vocabulary, syn1);
    fclose(fo);
  } // method-end

protected:
  std::shared_ptr< CustomLearningExampleProvider> lep;
  std::shared_ptr< CustomVocabulary > w_vocabulary;
  std::shared_ptr< CustomVocabulary > c_vocabulary;
  std::shared_ptr< CustomVocabulary > in_vocabulary;
  std::shared_ptr< CustomVocabulary > out_vocabulary;
  // размерность скрытого слоя (она же размерность эмбеддинга)
  size_t layer1_size;
  // количество эпох обучения
  size_t epoch_count;
  // learning rate
  float alpha;
  // начальный learning rate
  float starting_alpha;
  // алгоритм оптимизации (hierarchical softmax либо negative sampling)
  LearningOptimizationAlgo optimization_algo;
  // количество отрицательных примеров на каждый положительный при оптимизации методом negative sampling
  size_t negative;
  // матрицы весов между слоями input-hidden и hidden-output
  float *syn0 = nullptr, *syn1 = nullptr;
  // табличное представление логистической функции в области определения [-MAX_EXP; +MAX_EXP]
  float *expTable = nullptr;
  // noise distribution for negative sampling
  const size_t table_size = 1e8; // 100 млн.
  int *table = nullptr;
  // служебное поле для генерации случайних чисел
  unsigned long long next_random_ns;
  // функция инициализации распределения, имитирующего шум, для метода оптимизации negative sampling  -- для словаря слов
  void InitUnigramTable_w()
  {
    double train_words_pow = 0;
    double d1, power = 0.75;
    table = (int *)malloc(table_size * sizeof(int));
    // вычисляем нормирующую сумму  (за слагаемое берется абсолютная частота слова/контекста в степени 3/4)
    for (size_t a = 0; a < w_vocabulary->size(); ++a)
      train_words_pow += pow(w_vocabulary->idx_to_data(a).cn, power);
    // заполняем таблицу распределения, имитирующего шум
    size_t i = 0;
    d1 = pow(w_vocabulary->idx_to_data(i).cn, power) / train_words_pow;
    for (size_t a = 0; a < table_size; ++a)
    {
      table[a] = i;
      if (a / (double)table_size > d1)
      {
        i++;
        d1 += pow(w_vocabulary->idx_to_data(i).cn, power) / train_words_pow;
      }
      if (i >= w_vocabulary->size())
        i = w_vocabulary->size() - 1;
    }
  } // method-end
//  // функция инициализации распределения, имитирующего шум, для метода оптимизации negative sampling  -- для словаря контекстов
//  void InitUnigramTable_c()
//  {
//    double train_words_pow = 0;
//    double d1, power = 0.75;
//    table = (int *)malloc(table_size * sizeof(int));
//    // вычисляем нормирующую сумму  (за слагаемое берется абсолютная частота слова/контекста в степени 3/4)
//    for (size_t a = 0; a < c_vocabulary->size(); ++a)
//      train_words_pow += pow(c_vocabulary->idx_to_data(a).cn, power);
//    // заполняем таблицу распределения, имитирующего шум
//    size_t i = 0;
//    d1 = pow(c_vocabulary->idx_to_data(i).cn, power) / train_words_pow;
//    for (size_t a = 0; a < table_size; ++a)
//    {
//      table[a] = i;
//      if (a / (double)table_size > d1)
//      {
//        i++;
//        d1 += pow(c_vocabulary->idx_to_data(i).cn, power) / train_words_pow;
//      }
//      if (i >= c_vocabulary->size())
//        i = c_vocabulary->size() - 1;
//    }
//  } // method-end
private:
  uint64_t train_words = 0;
  uint64_t word_count_actual = 0;
  std::chrono::steady_clock::time_point start_learning_tp;

  void saveEmbeddingsBin_helper(FILE *fo, std::shared_ptr< CustomVocabulary > vocabulary, float *weight_matrix) const
  {
    for (size_t a = 0; a < vocabulary->size(); ++a)
    {
      fprintf(fo, "%s ", vocabulary->idx_to_data(a).word.c_str());
      for (size_t b = 0; b < layer1_size; ++b)
        fwrite(&weight_matrix[a * layer1_size + b], sizeof(float), 1, fo);
      fprintf(fo, "\n");
    }
  }
  void saveEmbeddingsTxt_helper(FILE *fo, std::shared_ptr< CustomVocabulary > vocabulary, float *weight_matrix) const
  {
    for (size_t a = 0; a < vocabulary->size(); ++a)
    {
      fprintf(fo, "%s", vocabulary->idx_to_data(a).word.c_str());
      for (size_t b = 0; b < layer1_size; ++b)
        fprintf(fo, " %lf", weight_matrix[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  }
}; // class-decl-end


#endif /* TRAINER_H_ */
