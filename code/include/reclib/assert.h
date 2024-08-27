#ifndef RECLIB_ASSERT_H
#define RECLIB_ASSERT_H

#include <assert.h>

#include <string>

#ifndef RECLIB_DEBUG_MODE
// disables all assertions from <assert.h>
#define NDEBUG
#endif

#ifndef NDEBUG
#define _RECLIB_ASSERT(x)                                                     \
  do {                                                                        \
    if (!(x)) {                                                               \
      std::cerr << "reclib assertion FAILED: \"" << #x << "\" (" << (bool)(x) \
                << ")\n  at " << __FILE__ << " line " << __LINE__ << "\n";    \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)
#define _RECLIB_ASSERT_EQ(x, y)                                               \
  do {                                                                        \
    if ((x) != (y)) {                                                         \
      std::cerr << "reclib assertion FAILED: " << #x << " == " << #y << " ("  \
                << (x) << " != " << (y) << ")\n  at " << __FILE__ << " line " \
                << __LINE__ << "\n";                                          \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)
#define _RECLIB_ASSERT_NE(x, y)                                               \
  do {                                                                        \
    if ((x) == (y)) {                                                         \
      std::cerr << "reclib assertion FAILED: " << #x << " != " << #y << " ("  \
                << (x) << " == " << (y) << ")\n  at " << __FILE__ << " line " \
                << __LINE__ << "\n";                                          \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)
#define _RECLIB_ASSERT_LE(x, y)                                              \
  do {                                                                       \
    if ((x) > (y)) {                                                         \
      std::cerr << "reclib assertion FAILED: " << #x << " <= " << #y << " (" \
                << (x) << " > " << (y) << ")\n  at " << __FILE__ << " line " \
                << __LINE__ << "\n";                                         \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)
#define _RECLIB_ASSERT_GE(x, y)                                              \
  do {                                                                       \
    if ((x) < (y)) {                                                         \
      std::cerr << "reclib assertion FAILED: " << #x << " >= " << #y << " (" \
                << (x) << " < " << (y) << ")\n  at " << __FILE__ << " line " \
                << __LINE__ << "\n";                                         \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)
#define _RECLIB_ASSERT_LT(x, y)                                               \
  do {                                                                        \
    if ((x) >= (y)) {                                                         \
      std::cerr << "reclib assertion FAILED: " << #x << " < " << #y << " ("   \
                << (x) << " >= " << (y) << ")\n  at " << __FILE__ << " line " \
                << __LINE__ << "\n";                                          \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)
#define _RECLIB_ASSERT_GT(x, y)                                               \
  do {                                                                        \
    if ((x) <= (y)) {                                                         \
      std::cerr << "reclib assertion FAILED: " << #x << " > " << #y << " ("   \
                << (x) << " <= " << (y) << ")\n  at " << __FILE__ << " line " \
                << __LINE__ << "\n";                                          \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)
#else
#define _RECLIB_ASSERT(x)
#define _RECLIB_ASSERT_EQ(x, y)
#define _RECLIB_ASSERT_GE(x, y)
#define _RECLIB_ASSERT_NE(x, y)
#define _RECLIB_ASSERT_LE(x, y)
#define _RECLIB_ASSERT_LT(x, y)
#endif

#endif