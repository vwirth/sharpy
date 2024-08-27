#ifndef RECLIB_UTILS_REINTERPRET_H
#define RECLIB_UTILS_REINTERPRET_H

namespace reclib {

template <typename To, typename From>
inline std::shared_ptr<To> reinterpret_pointer_cast(
    std::shared_ptr<From> const& ptr) noexcept {
  To* raw_ptr = reinterpret_cast<To*>(ptr.get());
  _RECLIB_ASSERT_NE(ptr.get(), (To*)nullptr);
  return std::shared_ptr<To>(ptr, raw_ptr);
}

}  // namespace reclib

#endif
