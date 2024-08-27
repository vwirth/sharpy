#ifndef RECLIB_OPENGL_NAMED_HANDLE_H
#define RECLIB_OPENGL_NAMED_HANDLE_H

#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "reclib/assert.h"
#include "reclib/platform.h"
#include "reclib/utils/reinterpret.h"

namespace reclib {
namespace opengl {

template <typename T, typename = int>
struct HasName : std::false_type {};
template <typename T>
struct HasName<T, decltype((void)T::name, 0)> : std::true_type {};

template <typename T>
class _API NamedHandle {
 public:
  // "default" construct
  NamedHandle() : ptr(nullptr) {}

  // create new object and store handle in map for later retrieval
  template <class... Args>
  NamedHandle(const std::string& name, Args&&... args)
      : ptr(std::make_shared<T>(name, args...)) {
    static_assert(HasName<T>::value,
                  "Template type T is required to have a member \"name\"!");
    static_assert(std::is_same<decltype(T::name), std::string>::value ||
                      std::is_same<decltype(T::name), const std::string>::value,
                  "bad type bro");

#ifndef NDEBUG
    if (map.count(ptr->name)) {
      std::cout << "[ERROR] NamedHandle<" << T::type_to_str() << "> of name '"
                << ptr->name << "' already exists." << std::endl;
    }
#endif
    assert(
        !map.count(ptr->name));  // check if key unique in NamedHandle<T>::map
    const std::lock_guard<std::mutex> lock(mutex);
    map[ptr->name] = *this;
  }

  template <typename S>
  NamedHandle(const NamedHandle<S> other) {
    static_assert(
        std::is_base_of<T, S>::value,
        "[ERROR] in NamedHandle constructor: S must be derived from T");

    ptr = other.ptr;
    other.erase(ptr->name);
    map[ptr->name] = *this;
  }

  // template <typename S>
  // NamedHandle(std::shared_ptr<S> other) {
  //   static_assert(
  //       std::is_base_of<T, S>::value,
  //       "[ERROR] in NamedHandle constructor: S must be derived from T");

  //   ptr = other;
  //   map[ptr->name] = *this;
  // }

  virtual ~NamedHandle() {
    // if (initialized() && ptr.use_count() == 1) {
    //   if (map.count(ptr->name) > 0) {
    //     erase(ptr->name);
    //   }
    // }
  }

  // copy / move
  NamedHandle(const NamedHandle<T>& other) = default;
  NamedHandle(NamedHandle<T>&& other) = default;
  inline NamedHandle<T>& operator=(const NamedHandle<T>& other) = default;
  inline NamedHandle<T>& operator=(NamedHandle<T>&& other) = default;

  // operators for pointer-like usage
  inline explicit operator bool() const { return ptr.operator bool(); }
  inline T* operator->() { return ptr.operator->(); }
  inline const T* operator->() const { return ptr.operator->(); }
  inline T& operator*() { return *ptr; }
  inline const T& operator*() const { return *ptr; }

  inline bool initialized() const { return ptr != nullptr; }

  template <typename To>
  inline std::shared_ptr<To> cast() {
    return reclib::reinterpret_pointer_cast<To>(ptr);
  }

  // check if mapping for given name exists
  static bool valid(const std::string& name) {
    const std::lock_guard<std::mutex> lock(mutex);
    return map.count(name);
  }
  // return mapped handle for given name
  static NamedHandle<T> find(const std::string& name) {
#ifndef NDEBUG
    if (!valid(name)) {
      std::cout << "[ERROR] NamedHandle<" << T::type_to_str() << "> of name '"
                << name << "' does not exist." << std::endl;
    }
#endif
    assert(valid(name));
    const std::lock_guard<std::mutex> lock(mutex);
    return map[name];
  }

  // remove element from map for given name
  static void erase(const std::string& name) {
    if (valid(name)) {
      const std::lock_guard<std::mutex> lock(mutex);
      map.erase(name);
    }
  }
  // clear saved handles and free unsused memory
  static void clear() {
    const std::lock_guard<std::mutex> lock(mutex);
    map.clear();
  }

  // erase the element from the map and free the
  // underlying pointer
  void free(bool force = false) {
    const std::lock_guard<std::mutex> lock(mutex);
    if (!initialized()) return;

    if (map.count(ptr->name) && (ptr.use_count() <= 2 || force)) {
      map.erase(ptr->name);
    } else if (map.count(ptr->name) == 0) {
      throw std::runtime_error("Namedhandle " + ptr->name +
                               " was erased from map before free.");
    } else {
      // std::cout << "do not free " << ptr->name
      //           << " as still used: " << ptr.use_count() << std::endl;
    }
    ptr.reset();
  }

  // iterators to iterate over all entries
  static typename std::map<std::string, NamedHandle<T>>::iterator begin() {
    return map.begin();
  }
  static typename std::map<std::string, NamedHandle<T>>::iterator end() {
    return map.end();
  }

  std::shared_ptr<T> ptr;
  static std::mutex mutex;
  static std::map<std::string, NamedHandle<T>> map;
};

// definition of static members (compiler magic)
template <typename T>
std::mutex NamedHandle<T>::mutex;
template <typename T>
std::map<std::string, NamedHandle<T>> NamedHandle<T>::map;

}  // namespace opengl
}  // namespace reclib

#endif
