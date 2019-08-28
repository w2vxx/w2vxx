#ifndef LEARNING_EXAMPLE_PROVIDER_H_
#define LEARNING_EXAMPLE_PROVIDER_H_

#include <vector>
#include <optional>
#include <cstring>       // for std::strerror


// структура, представляющая обучающий пример
struct LearningExample
{
  size_t word;                     // индекс слова (в словаре слов)
  std::vector<size_t> context;     // индексы контекстов (в словаре контекстов) -- в оригинальном word2vec "словарь контекстов" == "словарь слов"
};



// Базовый класс поставщика обучающих примеров ("итератор" по обучающему множеству).
// Выдает обучающие примеры в терминах индексов в словарях (полностью закрывает собой слова-строки).
class CustomLearningExampleProvider
{
public:
  // конструктор
  CustomLearningExampleProvider(size_t threadsCount)
  : threads_count(threadsCount)
  {
  }
  // деструктор
  virtual ~CustomLearningExampleProvider()
  {
  }
  // подготовительные действия, выполняемые перед каждой эпохой обучения
  virtual bool epoch_prepare(size_t threadIndex) = 0;
  // заключительные действия, выполняемые после каждой эпохой обучения
  virtual bool epoch_unprepare(size_t threadIndex) = 0;
  // получение очередного обучающего примера
  virtual std::optional<LearningExample> get(size_t threadIndex) = 0;
  // получение количества слов, фактически считанных из обучающего множества (т.е. без учета сабсэмплинга)
  virtual uint64_t getWordsCount(size_t threadIndex) const = 0;
protected:
  // количество потоков управления (thread), параллельно работающих с поставщиком обучающих примеров
  size_t threads_count;
  // получение размера файла
  uint64_t get_file_size(const std::string& filename)
  {
    // TODO: в будущем использовать std::experimental::filesystem::file_size
    std::ifstream ifs(filename, std::ios::binary|std::ios::ate);
    if ( !ifs.good() )
        throw std::runtime_error(std::strerror(errno));
    return ifs.tellg();
  }
};


#endif /* LEARNING_EXAMPLE_PROVIDER_H_ */
