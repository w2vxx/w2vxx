#ifndef ORIGINAL_WORD2VEC_VOCABULARY_H_
#define ORIGINAL_WORD2VEC_VOCABULARY_H_

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <regex>
#include <limits>
#include "vocabulary.h"

class OriginalWord2VecVocabulary : public CustomVocabulary
{
public:
  // конструктор
  OriginalWord2VecVocabulary()
  : CustomVocabulary()
  {
    vocabulary_hash.reserve(21000000);  // изначально устанавливаем размер хэш-отображения таковым, чтобы эффективно хранить 21 млн. элементов
  }
  // деструктор
  virtual ~OriginalWord2VecVocabulary()
  {
  }
  // функция загрузки словаря из файла
  // предполагается, что словарь отсортирован по убыванию частоты встречаемости слов
  bool load(const std::string& filename)
  {
    // считываем словарь из файла
    std::ifstream ifs( filename );
    if (!ifs.good())
    {
      std::cerr << "Can't open vocabulary file: " << filename << std::endl;
      return false;
    }
    std::string buf;
    while ( std::getline(ifs, buf).good() )
    {
      // каждая запись словаря содержит слово (строку) и абсолютную частоту встречаемости данного слова в корпусе (на основе которого построен словарь)
      // элементы словарной записи разделены пробелами
      const std::regex space_re("\\s+");
      std::vector<std::string> vocabulary_record_components {
          std::sregex_token_iterator(buf.cbegin(), buf.cend(), space_re, -1),
          std::sregex_token_iterator()
      };
      if (vocabulary_record_components.size() != 2)
      {
        std::cerr << "Vocabulary loading error: " << filename << std::endl;
        std::cerr << "Invalid record: " << buf << std::endl;
        return false;
      }
      vocabulary_hash[vocabulary_record_components[0]] = vocabulary.size(); // сразу строим хэш-отображение для поиска индекса слова в словаре по слову (строке)
      vocabulary.emplace_back( vocabulary_record_components[0], std::stoull(vocabulary_record_components[1]) );
    }
    return true;
  }
  // получение индекса в словаре по тексту слова
  size_t word_to_idx(const std::string& word) const
  {
    auto it = vocabulary_hash.find(word);
    if (it == vocabulary_hash.end())
      return std::numeric_limits<size_t>::max();
    else
      return it->second;
  }
private:
  // хэш-отображение слов в их индексы в словаре (для быстрого поиска)
  std::unordered_map<std::string, size_t> vocabulary_hash;
};

#endif /* ORIGINAL_WORD2VEC_VOCABULARY_H_ */
