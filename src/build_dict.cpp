#include <string>
#include <cstring>       // for std::strerror
#include <map>
#include <unordered_map>
#include "simple_profiler.h"
#include "build_dict_command_line_parameters.h"

const size_t MAX_STRING = 100;

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


int main(int argc, char **argv)
{
  // выполняем разбор параметров командной строки
  BuildDictCommandLineParameters cmdLineParams;
  cmdLineParams.parse(argc, argv);
  cmdLineParams.dbg_cout();

  if (!cmdLineParams.isDefined("-save-vocab") || !cmdLineParams.isDefined("-train"))
    return 0;

  SimpleProfiler global_profiler;

  // открываем файл с тренировочными данными
  FILE *fi = fopen(cmdLineParams.getAsString("-train").c_str(), "rb");
  if ( fi == NULL )
  {
    std::cout << "Train-file open: error: " << std::strerror(errno) << std::endl;
    return 0;
  }

  // создаем контейенер для словаря
  //std::map<std::string, uint64_t> dict;
  std::unordered_map<std::string, uint64_t> dict;
  // создаем счетчик для переводов строк
  uint64_t eolCount = 0;
  // создаем счетчик слов (для вывода прогресса)
  uint64_t wordsCnt = 0;
  // порог отсечения при сокращении словаря
  size_t min_reduce = 1;
  // читаем тренировочные данные
  while (true)
  {
    std::string word;
    read_word(fi, word);
    if (feof(fi)) break;
    ++wordsCnt;
    if (wordsCnt % 100000 == 0)
    {
      std::cout << '\r' << (wordsCnt / 1000) << " K     ";
      std::cout.flush();
    }
    if (word == "</s>")
    {
      ++eolCount;
      continue;
    }
    auto it = dict.find(word);
    if (it == dict.end())
      dict[word] = 1;
    else
      ++it->second;
    if (dict.size() > 21000000)
    {
      std::cout << std::endl << "Reduce!" << std::endl;
      auto it = dict.begin();
      while (it != dict.end())    //TODO: в будущем заменить на std::experimental::erase_if (возможно на std::remove_if)
      {
        if (it->second > min_reduce)
          ++it;
        else
          it = dict.erase(it);
      }
      ++min_reduce;
    }
  }
  fclose(fi);
  // выполняем отсечение по min-count
  size_t min_count = cmdLineParams.getAsInt("-min-count");
  std::cout << std::endl << "min-count reduce!" << std::endl;
  auto it = dict.begin();
  while (it != dict.end())    //TODO: в будущем заменить на std::experimental::erase_if (возможно на std::remove_if)
  {
    if (it->second >= min_count)
      ++it;
    else
      it = dict.erase(it);
  }
  wordsCnt = eolCount;
  for (auto& i : dict)
    wordsCnt += i.second;
  std::cout << "Vocab size: " << (dict.size() + 1) << std::endl;
  std::cout << "Words in train file: " << wordsCnt << std::endl;
  // пересортируем в порядке убывания частоты
  std::multimap<uint64_t, std::string> revDict;
  for (auto& i : dict)
    revDict.insert( std::pair<uint64_t, std::string>(i.second, i.first) );
  // сохраняем словаь в файл
  FILE *fo = fopen(cmdLineParams.getAsString("-save-vocab").c_str(), "wb");
  fprintf(fo, "%s %lu\n", "</s>", eolCount);
  for (auto it = revDict.crbegin(), itEnd = revDict.crend(); it != itEnd; ++it)
    fprintf(fo, "%s %lu\n", it->second.c_str(), it->first);
  fclose(fo);

  return 0;
}
