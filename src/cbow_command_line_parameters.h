#ifndef CBOW_COMMAND_LINE_PARAMETERS_H_
#define CBOW_COMMAND_LINE_PARAMETERS_H_

#include "command_line_parameters.h"

class CbowCommandLineParameters : public CommandLineParameters
{
public:
  CbowCommandLineParameters()
  {
    // initialize params mapping with std::initialzer_list<T>
    params_ = {
        {"-words-vocab",  {"The words vocabulary will be read from <file>", std::nullopt, std::nullopt}},
        {"-ctx-vocab",    {"The contexts vocabulary will be read from <file>", std::nullopt, std::nullopt}},
        {"-min-count",    {"This will discard words that appear less than <int> times", "5", std::nullopt}},
        {"-train",        {"Use text data from <file> to train the model", std::nullopt, std::nullopt}},
        {"-backup",       {"Save neural network weights to <file>", std::nullopt, std::nullopt}},
        {"-restore",      {"Restore neural network weights from <file>", std::nullopt, std::nullopt}},
        {"-output",       {"Use <file> to save the resulting word vectors", std::nullopt, std::nullopt}},
        {"-size",         {"Set size of word vectors", "100", std::nullopt}},
        {"-window",       {"Set max skip length between words", "5", std::nullopt}},
        {"-sample",       {"Set threshold for occurrence of words. Those that appear with higher frequency in the training data will be randomly down-sampled", "1e-3", std::nullopt}},
        {"-optimization", {"Optimization method: hierarchical softmax (hs) or negative sampling (ns)", "ns", std::nullopt}},
        {"-negative",     {"Number of negative examples", "5", std::nullopt}},
        {"-alpha",        {"Set the starting learning rate", "0.05", std::nullopt}},
        {"-iter",         {"Run more training iterations", "5", std::nullopt}},
        {"-threads",      {"Use <int> threads", "12", std::nullopt}}
    };
  }
};

#endif /* CBOW_COMMAND_LINE_PARAMETERS_H_ */
