
/*
 * Created: 05-06-2018
 * Author: Jonas R. Glesaaen (jonas@glesaaen.com)
 */

#ifndef FUNCTION_ALIAS_HPP
#define FUNCTION_ALIAS_HPP

#include <type_traits>
#include <utility>

#define OPENQCD_GLOBAL_FUNCTION_ALIAS(fn)                                      \
  template <typename... Args>                                                  \
  inline auto fn(Args &&... args)                                              \
      ->decltype(openqcd__##fn(std::forward<Args>(args)...))                   \
  {                                                                            \
    return openqcd__##fn(std::forward<Args>(args)...);                         \
  }

#define OPENQCD_MODULE_FUNCTION_ALIAS(fn, mod)                                 \
  template <typename... Args>                                                  \
  inline auto fn(Args &&... args)                                              \
      ->decltype(openqcd_##mod##__##fn(std::forward<Args>(args)...))           \
  {                                                                            \
    return openqcd_##mod##__##fn(std::forward<Args>(args)...);                 \
  }

#define OPENQCD_GLOBAL_VARIABLE_ALIAS(var)                                     \
  inline auto var()                                                            \
      ->std::add_const<                                                        \
          std::add_lvalue_reference<decltype(openqcd__##var)>::type>::type     \
  {                                                                            \
    return openqcd__##var;                                                     \
  }

#endif /* FUNCTION_ALIAS_HPP */
