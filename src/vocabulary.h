#ifndef VOCABULARY_H_
#define VOCABULARY_H_

#include <string>
#include <vector>
#include <limits>
#include <algorithm>
#include <numeric>
#include <list>
#include <iostream>


// данные словаря
struct VocabularyData
{
  // слово или контекст
  std::string word;
  // абсолютная частота
  uint64_t cn;
  // код Хаффмана для данного слова/контекста (для алгоритма Hierarchical Softmax)
  std::string huffman_code;
  // код Хаффмана, записанный в float-формате
  std::vector<float> huffman_code_float;
  // путь в дереве Хаффмана (от корня к листу), соответствующий данному слову (для алгоритма Hierarchical Softmax)
  // элементами пути являются индексы внутренних узлов в дереве Хаффмана
  std::vector<int> huffman_path;
  // конструктор
  VocabularyData(const std::string& theWord, const uint64_t theFrequency)
  : word(theWord), cn(theFrequency)
  {}
};


// базовый класс словаря
class CustomVocabulary
{
public:
  // конструктор
  CustomVocabulary()
  {
  }
  // деструктор
  virtual ~CustomVocabulary()
  {
  }
  // получение индекса в словаре по тексту слова/контекста
  virtual size_t word_to_idx(const std::string& word) const = 0;
  // получение данных словаря по индексу
  inline const VocabularyData& idx_to_data(size_t word_idx) const
  {
    // без валидации word_idx для скорости
    return vocabulary[word_idx];
  }
  // получение размера словаря
  size_t size() const
  {
    return vocabulary.size();
  }
  // вычисление суммы абсолютных частот слов словаря
  uint64_t cn_sum() const
  {
    //return std::reduce(vocabulary.cbegin(), vocabulary.cend(), 0, [](const uint64_t& sum, const VocabularyData& r) -> uint64_t { return sum + r.cn; });
    //return std::accumulate(vocabulary.cbegin(), vocabulary.cend(), 0, [](const uint64_t& sum, const VocabularyData& r) -> uint64_t { return sum + r.cn; });
    return std::accumulate( vocabulary.cbegin(), vocabulary.cend(),
                            static_cast<uint64_t>(0),
                            [](const uint64_t& sum, const VocabularyData& r) -> uint64_t { return sum + r.cn; } );
  }
  // построение дерева Хаффмана, вычисление кодов Хаффмана и путей для каждого слова/контекста
  // !!! изнчально предполагается, что вектор vocabulary отсортирован по убыванию cn
  void buildHuffmanTree()
  {
    if (size() == 0) return;
    // в векторе count хранится таблица частот для всего дерева
    // начало вектора соответствует листьям (кодируемым элементам, т.е. словам/контекстам); далее идут промежуточные узлы дерева; вершине дерева будет соответствовать последний элемент вектора
    std::vector<uint64_t> count( size()*2-1, std::numeric_limits<uint64_t>::max() );
    std::transform(vocabulary.begin(), vocabulary.end(), count.begin(), [](const VocabularyData& data) -> uint64_t {return data.cn;});
    // в векторе binary хранится метка (0 или 1), присвоенная дуге, ведущей к родителю данного узла
    std::vector<bool> binary( size()*2-1, false );
    // в векторе parent хранится индекс узла, родительского по отношению к данному
    std::vector<size_t> parent( size()*2-1 );
    // построение дерева
    int pos1 = size() - 1;     // индекс, пробегающий листья дерева, в ходе его построения
    int pos2 = size();         // индекс, пробегающий промежуточные узлы дерева, в ходе его построения
    for (size_t idx = 0; idx < (size() - 1); ++idx)
    {
      // лямбда-функция для поиска очередного узла с наименьшей частотой
      auto min_node = [&pos1, &pos2, &count]() -> size_t
                      {
                        size_t result;
                        if (pos1 >= 0)
                        {
                          if (count[pos1] < count[pos2])
                          {
                            result = pos1;
                            pos1--;
                          } else {
                            result = pos2;
                            pos2++;
                          }
                        } else {
                          result = pos2;
                          pos2++;
                        }
                        return result;
                      };
      // отыщем два узла с нименьшей частотой
      size_t min1i = min_node();
      size_t min2i = min_node();
      count[size() + idx] = count[min1i] + count[min2i];
      parent[min1i] = size() + idx;
      parent[min2i] = size() + idx;
      binary[min2i] = true;
    }
    // присваиваем Хаффман-коды каждому слову/контексту словаря
    for (size_t idx = 0; idx < size(); ++idx)
    {
      size_t idx_in_path = idx;  // индекс очередного узла в пути от листу к корню дерева
      size_t path_len = 0;       // накопленная к настоящему времени длина пути (количество дуг)
      std::list<char> code;          // накопитель кода Хаффмана
      std::list<int> path_indexes;   // накопитель пути в дереве Хаффмана (в итоге здесь окажется путь от корня к листу)
      while (true)
      {
        code.push_front( binary[idx_in_path] ? '1' : '0' );
        path_indexes.push_front(idx_in_path);
        path_len++;
        idx_in_path = parent[idx_in_path];
        if (idx_in_path == size() * 2 - 2)
          break;
      }
      path_indexes.push_front( size() * 2 - 2 ); // вершину дерева добавляем в начало пути
      vocabulary[idx].huffman_code.resize( code.size() );
      std::copy( code.cbegin(), code.cend(), vocabulary[idx].huffman_code.begin() );
      vocabulary[idx].huffman_code_float.resize( code.size() );
      std::transform( code.cbegin(), code.cend(), vocabulary[idx].huffman_code_float.begin(), [](const char c) -> float {return (c == '1') ? 1.0 : 0.0;} );
      // индексы в пути переиндексируются таким образом, чтобы листья имели отрицательные индексы, а промежуточные вершины неотрицательные (и их индексация начиналась с 0)
      vocabulary[idx].huffman_path.resize( path_indexes.size() );
      std::transform( path_indexes.cbegin(), path_indexes.cend(), vocabulary[idx].huffman_path.begin(), [this](const int curIdx) -> int {return curIdx - size();} );
    }
  } // method-end
protected:
  std::vector<VocabularyData> vocabulary;
};


#endif /* VOCABULARY_H_ */
