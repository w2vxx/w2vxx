#ifndef BUILD_DICT_COMMAND_LINE_PARAMETERS_H_
#define BUILD_DICT_COMMAND_LINE_PARAMETERS_H_

#include "command_line_parameters.h"

class BuildDictCommandLineParameters : public CommandLineParameters
{
public:
  BuildDictCommandLineParameters()
  {
    // initialize params mapping with std::initialzer_list<T>
    params_ = {
        {"-save-vocab",   {"The vocabulary will be saved to <file>", std::nullopt, std::nullopt}},
        {"-min-count",    {"This will discard words that appear less than <int> times", "100", std::nullopt}},
        {"-train",        {"Use text data from <file> to train the model", std::nullopt, std::nullopt}}
    };
  }
};

#endif /* BUILD_DICT_COMMAND_LINE_PARAMETERS_H_ */
