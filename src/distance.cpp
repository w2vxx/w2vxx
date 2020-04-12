#include <string>
//#include <cstring>       // for std::strerror
#include <iostream>
#include <vector>
#include <fstream>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <map>

bool loadModel(const std::string& model_filename, uint64_t& words, uint64_t& size, std::vector<std::string>& vocab, float*& embeddings)
{
  // открываем файл модели
  std::ifstream ifs(model_filename.c_str(), std::ios::binary);
  if ( !ifs.good() )
  {
    std::cerr << "Input file not found" << std::endl;
    return false;
  }
  std::string buf;
  // считыавем размер матрицы
  ifs >> words;
  ifs >> size;
  std::getline(ifs,buf); // считываем конец строки
  // выделяем память для эмбеддингов
  embeddings = (float *) malloc( words * size * sizeof(float) );
  if (embeddings == nullptr)
  {
    std::cerr << "Cannot allocate memory: " << (words * size * sizeof(float) / 1048576) << " MB" << std::endl;
    std::cerr << "    Words: " << words << std::endl;
    std::cerr << "    Embedding size: " << size << std::endl;
    return false;
  }
  // загрузка словаря и векторов
  for (uint64_t w = 0; w < words; ++w)
  {
    std::getline(ifs, buf, ' '); // читаем слово (до пробела)
    vocab.push_back(buf);
    float* eOffset = embeddings + w*size;
    ifs.read( reinterpret_cast<char*>( eOffset ), sizeof(float)*size ); // читаем вектор
    // нормируем вектор (все компоненты в диапазон [-1; +1]
    float len = std::sqrt( std::inner_product(eOffset, eOffset+size, eOffset, 0.0) );
    if (len == 0)
    {
      std::cerr << "Embedding normalization error: Division by zero" << std::endl;
      free(embeddings);
      return false;
    }
    std::transform(eOffset, eOffset+size, eOffset, [len](float a) -> float {return a/len;});
    std::getline(ifs,buf); // считываем конец строки
  }
  return true;
}


int main(int argc, char **argv)
{
  // разбор параметров
  if (argc < 2)
  {
    std::cout << "Usage: ./distance <FILE> [N]" << std::endl
              << "    where FILE contains word vectors in the BINARY FORMAT" << std::endl
              << "          N -- number of closest words that will be shown (default: 40)" << std::endl;
    return -1;
  }

  // создаем контейнер для словаря модели
  std::vector<std::string> vocab;
  // декларируем хранилище для векторов
  float *embeddings = nullptr;
  uint64_t words = 0, size = 0;
  if ( ! loadModel(argv[1], words, size, vocab, embeddings) )
    return -1;

  // определяем, сколько ближайших выводить в результат
  size_t n = 40;
  if (argc >= 3)
  {
    try { n = std::stoul(argv[2]); } catch (...) {}
  }

  // в цикле считываем слова и ищем для них ближайшие (по косинусной мере) в векторной модели
  while (true)
  {
    // запрашиваем слово у пользователя
    std::string word;
    std::cout << "Enter word (EXIT to break): ";
    std::cout.flush();
    std::cin >> word;
    if (word == "EXIT") break;
    if (word.length() > 100)
    {
      std::cout << "  the word is too long..." << std::endl;
      continue;
    }
    // ищем слово в словаре (проверим, что оно есть и получим индекс)
    size_t widx = 0;
    for ( ; widx < words; ++widx )
      if (vocab[widx] == word)
        break;
    if (widx == words)
    {
      std::cout << "  out of dictionary word..." << std::endl;
      continue;
    }
    float* wiOffset = embeddings + widx*size;
    std::cout << "                                       word | cosine similarity" << std::endl
              << "  -------------------------------------------------------------" << std::endl;
    std::multimap<float, std::string> best;
    for (size_t i = 0; i < words; ++i)
    {
      if (i == widx) continue;
      float* iOffset = embeddings + i*size;
      float dist = std::inner_product(iOffset, iOffset+size, wiOffset, 0.0);
      if (best.size() < n)
        best.insert( std::pair<float, std::string>(dist, vocab[i]) );
      else
      {
        auto minIt = best.begin();
        if (dist > minIt->first)
        {
          best.erase(minIt);
          best.insert( std::pair<float, std::string>(dist, vocab[i]) );
        }
      }
    }
    // выводим результат поиска
    for (auto it = best.crbegin(), itEnd = best.crend(); it != itEnd; ++it)
    {
      std::string alignedWord = (it->second.length() >= 41) ? it->second : (std::string(41-it->second.length(), ' ') + it->second);
      std::cout << "  " << alignedWord << "   " << it->first <<std::endl;
    }
  } // infinite loop

  free(embeddings);
  return 0;
}
