#ifndef COMMAND_LINE_PARAMETERS_H_
#define COMMAND_LINE_PARAMETERS_H_

#include <string>
#include <optional>
#include <map>
#include <vector>
#include <iostream>

// Тип для имен параметров
typedef std::string ParameterName;

// Тип для описания параметров и хранения их значений
struct ParameterData
{
  std::string description;
  std::optional<std::string> default_value;
  std::optional<std::string> defined_value;
};


// Базовый класс для доступа к параметрам командной строки
class CommandLineParameters
{
public:
  // конструктор
  CommandLineParameters()
  {
  }
  // парсер параметров в формате main(...)
  bool parse(int argc, char **argv)
  {
    const std::vector<std::string> args_vector(argv, argv + argc);
    std::string last_acceptable_arg;
    for (auto&& arg : args_vector)
    {
      if ( !last_acceptable_arg.empty() )
      {
        params_[last_acceptable_arg].defined_value = arg;
        last_acceptable_arg.clear();
      }
      else
      {
        auto it = params_.find(arg);
        if (it != params_.end())
          last_acceptable_arg = arg;
      }
    }
    return true;
  }
  // отладочный вывод в консоль всех текущих значений
  void dbg_cout() const
  {
    const std::string MARGIN_LEFT(2, ' ');
    for (auto&& [key, value] : params_)
    {
      auto&& default_value = value.default_value.value_or( std::string() );
      std::cout << MARGIN_LEFT;
      std::cout << key << " = " << value.defined_value.value_or( "<none>" );
      if (!default_value.empty())
        std::cout << "  (default: " << default_value << ")";
      std::cout << std::endl;
    }
  }
  // получение строкового параметра
  std::string getAsString(const std::string& paramName) const
  {
    return getInternal(paramName).value_or( std::string() );
  }
  // получение знакового целочисленного параметра
  int getAsInt(const std::string& paramName) const
  {
    auto internalResult = getInternal(paramName);
    if (!internalResult)
      return 0;
    else
      return std::stoi(internalResult.value());
  }
  // получение вещественного параметра
  float getAsFloat(const std::string& paramName) const
  {
    auto internalResult = getInternal(paramName);
    if (!internalResult)
      return 0;
    else
      return std::stof(internalResult.value());
  }
  // проверка, является ли параметр известным приложению
  bool isAcceptable(const std::string& paramName) const
  {
    auto it = params_.find(paramName);
    return (it != params_.end());
  }
  // проверка, было ли определено значение параметра
  bool isDefined(const std::string& paramName) const
  {
    auto it = params_.find(paramName);
    if (it == params_.end())
      return false;
    return it->second.defined_value.has_value();
  }
protected:
  // хранилище описаний и значений параметров
  std::map<ParameterName, ParameterData> params_;
private:
  std::optional<std::string> getInternal(const std::string& paramName) const
  {
    auto it = params_.find(paramName);
    if (it == params_.end())
      return std::nullopt;
    if (it->second.defined_value)
      return it->second.defined_value;
    else if (it->second.default_value)
      return it->second.default_value;
    else
      return std::nullopt;
  }
};


#endif /* COMMAND_LINE_PARAMETERS_H_ */
