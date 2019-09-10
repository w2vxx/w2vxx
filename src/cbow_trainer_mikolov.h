#ifndef SRC_CBOW_TRAINER_MIKOLOV_H_
#define SRC_CBOW_TRAINER_MIKOLOV_H_

#include <algorithm>
#include <functional>
#include <numeric>
//#include <execution>

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
  void learning_model(const LearningExample& le, float *neu1, float *neu1e)
  {
    if (le.context.size() == 0) return;
    // зануляем текущие значения выходов нейронов скрытого слоя и текущие значения ошибок
    std::fill(neu1, neu1+layer1_size, 0.0);
    std::fill(neu1e, neu1e+layer1_size, 0.0);
    // вычисляем выход скрытого слоя ( in --> hidden )
    // в cbow он вычисляется как "средний" вектор слов контекста (так называемый "проекционный" слой)
    for (auto&& ctx_idx : le.context)  // складываем все вектора слов контекста
      std::transform(neu1, neu1+layer1_size, syn0+ctx_idx*layer1_size, neu1, std::plus<float>());
    std::transform(neu1, neu1+layer1_size, neu1, std::bind(std::divides<float>(), std::placeholders::_1, le.context.size())); // нормируем по числу слов контекста
    //
    if (optimization_algo == loaHierarchicalSoftmax)  // hierarchical softmax
    {
      auto&& current_word_data = w_vocabulary->idx_to_data(le.word);
      const size_t huffman_code_len = current_word_data.huffman_code_float.size();
      for (size_t d = 0; d < huffman_code_len; ++d)
      {
        // вычисляем смещение вектора, соответствующего очередному узлу в дереве Хаффмана
        float *nodeVectorPtr = syn1 + current_word_data.huffman_path[d] * layer1_size;
        // вычисляем выход соответствующего нейрона выходного слоя (hidden -> output)
        // он вычисляется как сигма-функция от скалярного произведения векторов: весового вектора, соответствующего текущему узлу в дереве Хаффмана, и вектора, соответствующего выходу скрытого слоя
        //float f = std::transform_reduce(std::execution::par, neu1, neu1+layer1_size, nodeVectorPtr, 0.0, std::plus<>(), std::multiplies<>());
        float f = std::inner_product(neu1, neu1+layer1_size, nodeVectorPtr, 0.0);
        if (f <= -MAX_EXP || f >= MAX_EXP) continue;
        else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
        // 'g' is the gradient multiplied by the learning rate
        float g = (1.0 - current_word_data.huffman_code_float[d] - f) * alpha;
        // Propagate errors output -> hidden
        std::transform(neu1e, neu1e+layer1_size, nodeVectorPtr, neu1e, [g](float a, float b) -> float {return a + g*b;});
        // Learn weights hidden -> output
        std::transform(nodeVectorPtr, nodeVectorPtr+layer1_size, neu1, nodeVectorPtr, [g](float a, float b) -> float {return a + g*b;});
      }
    }
    else if (optimization_algo == loaNegativeSampling) // negative sampling
    {
      size_t target;
      int label; // знаковое целое (!)
      float g = 0.0;
      for (size_t d = 0; d <= negative; ++d)
      {
        if (d == 0) // на первой итерации рассматриваем положительный пример
        {
          target = le.word;
          label = 1;
        }
        else // на остальных итерациях рассматриваем отрицательные примеры (шум)
        {
          next_random_ns = next_random_ns * (unsigned long long)25214903917 + 11;
          target = table[(next_random_ns >> 16) % table_size];
          if (target == 0) target = next_random_ns % (w_vocabulary->size() - 1) + 1;
          if (target == le.word) continue;
          label = 0;
        }
        // вычисляем смещение вектора, соответствующего очередному положительному/отрицательному примеру
        float *targetVectorPtr = syn1 + target * layer1_size;
        // вычисляем выход соответствующего нейрона выходного слоя (hidden -> output)
        float f = std::inner_product(neu1, neu1+layer1_size, targetVectorPtr, 0.0);
        // вычислим градиент умноженный на коэффициент скорости обучения
        if (f > MAX_EXP) g = (label - 1) * alpha;
        else if (f < -MAX_EXP) g = (label - 0) * alpha;
        else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
        // Propagate errors output -> hidden
        std::transform(neu1e, neu1e+layer1_size, targetVectorPtr, neu1e, [g](float a, float b) -> float {return a + g*b;});
        // Learn weights hidden -> output
        std::transform(targetVectorPtr, targetVectorPtr+layer1_size, neu1, targetVectorPtr, [g](float a, float b) -> float {return a + g*b;});
      }
    }
    // коррекция весов между входным и скрытым слоем (обратное распространение ошибки на участке hidden -> in)
    for (auto&& ctx_idx : le.context)
    {
      size_t ctxOffset = ctx_idx * layer1_size;
      std::transform(syn0+ctxOffset, syn0+ctxOffset+layer1_size, neu1e, syn0+ctxOffset, std::plus<float>());
    }
  } // method-end
}; // class-end


#endif /* SRC_CBOW_TRAINER_MIKOLOV_H_ */
