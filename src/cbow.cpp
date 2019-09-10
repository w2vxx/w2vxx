#include <memory>
#include <string>
#include <thread>
#include <chrono>
#include "simple_profiler.h"
#include "cbow_command_line_parameters.h"
#include "original_word2vec_vocabulary.h"
#include "original_word2vec_le_provider.h"
#include "cbow_trainer_mikolov.h"



int main(int argc, char **argv)
{
  // выполняем разбор параметров командной строки
  CbowCommandLineParameters cmdLineParams;
  cmdLineParams.parse(argc, argv);
  cmdLineParams.dbg_cout();

  if (!cmdLineParams.isDefined("-words-vocab") || !cmdLineParams.isDefined("-train") || !cmdLineParams.isDefined("-output"))
    return 0;

  SimpleProfiler global_profiler;

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
//  if (cmdLineParams.isDefined("-backup"))
//    trainer.backup( cmdLineParams.getAsString("-backup") );

  return 0;
}
