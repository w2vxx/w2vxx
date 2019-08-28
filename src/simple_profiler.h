#ifndef SIMPLE_PROFILER_H_
#define SIMPLE_PROFILER_H_

#include <chrono>
#include <iostream>


class SimpleProfiler
{
public:
  SimpleProfiler()
  {
    start_tp = std::chrono::steady_clock::now();
  }
  ~SimpleProfiler()
  {
    auto finish_tp = std::chrono::steady_clock::now();
//    std::chrono::duration<double, std::milli> ms = finish_tp - start_tp;
//    std::cout << std::fixed << "  time elapsed: " << ms.count() << " ms" << std::endl;
    std::chrono::duration< double, std::ratio<1> > seconds = finish_tp - start_tp;
    std::cout << std::fixed << "  time elapsed: " << seconds.count() << " seconds" << std::endl;
  }
private:
  std::chrono::steady_clock::time_point start_tp;
};



#endif /* SIMPLE_PROFILER_H_ */
