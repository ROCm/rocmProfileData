// Copyright (C) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Scope-object helper for rlog ranges.
//
// See OPTIMIZATIONS.md for the rationale and measurements behind every design
// choice in this header. The short version:
//
//   * The application must do NO work when no tool is attached. Guard on a
//     caller-cached std::atomic<bool>, never on rlog::isActive() (library call).
//
//   * Expensive range arguments must be passed as a CALLABLE, not a string.
//     Function arguments are evaluated at the call site, before the constructor
//     is entered, so a guard inside the constructor is far too late.
//     Measured: eager string args cost 509 ns per range with logging OFF; the
//     deferred form costs 0 (within noise of an empty loop).
//
//   * The guard is sampled TWICE, independently: once at push, once at pop.
//     It is deliberately NOT latched. If tracing resumes mid-range, the pop is
//     still delivered; that orphan pop is the only evidence the range existed,
//     and it is what lets the tool keep the descendants correctly nested.
//     A latched object emits nothing and silently re-parents every descendant
//     to depth 0.
//
// Usage:
//
//   static std::atomic<bool> appActive{false};
//   static void onActiveChanged() {
//       appActive.store(rlog::isActive(), std::memory_order_relaxed);
//   }
//   // once at startup:
//   rlog::init();
//   rlog::registerActiveCallback(onActiveChanged);
//
//   // cheap args (literal / already-computed):
//   rlog::Range r(appActive, "matmul", "static");
//
//   // expensive args - pass a lambda, it is only invoked when logging:
//   rlog::Range r(appActive, "matmul",
//                 [&]{ return rlog::fmt("m=%d n=%d", m, n); });
//
//   // the lambda may run ANY code; rlog::fmt is only a convenience.
//   // It must return std::string or const char*. Other shapes that work:
//   rlog::Range r(appActive, "matmul", [&]{ return std::to_string(n); });
//   rlog::Range r(appActive, "matmul", [&]{ std::ostringstream o; o << n; return o.str(); });
//   rlog::Range r(appActive, "matmul", [&]{ return describeTensor(t); });
//   rlog::Range r(appActive, "matmul", [&]{ std::string s; for (int i : v) s += ...; return s; });
//
//   // A returned const char* must outlive the push (literal/static is fine;
//   // a pointer into a temporary that dies with the lambda is not).
//
// Requirements that must not be broken (each was measured):
//   1. The argument constructor must be a TEMPLATE on the callable type, and
//      the callable must never be stored. std::function anywhere in the path
//      forces construction plus an unconditional indirect call while inactive.
//   2. The class itself is not a template, so C++14 can deduce nothing and
//      needs nothing deduced: `rlog::Range r(...)` just works, no helper fn.
//   3. domain/category/apiname are string literals in practice and are held as
//      bare const char*. They must outlive the Range.

#pragma once

#include <rlog/client.h>

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <string>
#include <type_traits>

namespace rlog {

// Convenience printf-style formatter for use INSIDE an argument lambda.
// Never call this outside the lambda - that would defeat the whole point.
inline std::string fmt(const char *format, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;

inline std::string fmt(const char *format, ...)
{
    char buf[1024];
    va_list ap;
    va_start(ap, format);
    int n = vsnprintf(buf, sizeof buf, format, ap);
    va_end(ap);
    if (n < 0)
        return std::string();
    if (static_cast<size_t>(n) < sizeof buf)
        return std::string(buf, n);

    std::string big(static_cast<size_t>(n) + 1, '\0');
    va_start(ap, format);
    vsnprintf(&big[0], big.size(), format, ap);
    va_end(ap);
    big.resize(static_cast<size_t>(n));
    return big;
}


class Range
{
public:
    // ---- eager form: arguments already exist and cost nothing to pass ----

    Range(const std::atomic<bool> &gate, const char *domain, const char *category,
          const char *apiname, const char *args)
    : m_gate(&gate)
    {
        if (!gate.load(std::memory_order_relaxed))
            return;
        rlog::rangePush(domain, category, apiname, args);
    }

    Range(const std::atomic<bool> &gate, const char *apiname, const char *args)
    : m_gate(&gate)
    {
        if (!gate.load(std::memory_order_relaxed))
            return;
        rlog::rangePush(apiname, args);
    }

    // ---- deferred form: argument callable, invoked only while logging ----
    //
    // The enable_if keeps a plain `const char *` from selecting this overload.
    // ArgFn must be callable with no arguments and return something with
    // .c_str() (std::string) or convertible to const char*.

    template <class ArgFn,
              class = typename std::enable_if<
                  !std::is_convertible<ArgFn, const char *>::value>::type>
    Range(const std::atomic<bool> &gate, const char *domain, const char *category,
          const char *apiname, ArgFn &&argfn)
    : m_gate(&gate)
    {
        if (!gate.load(std::memory_order_relaxed))
            return;
        rlog::rangePush(domain, category, apiname, cstr(argfn()));
    }

    template <class ArgFn,
              class = typename std::enable_if<
                  !std::is_convertible<ArgFn, const char *>::value>::type>
    Range(const std::atomic<bool> &gate, const char *apiname, ArgFn &&argfn)
    : m_gate(&gate)
    {
        if (!gate.load(std::memory_order_relaxed))
            return;
        rlog::rangePush(apiname, cstr(argfn()));
    }

    // ---- no-argument form ----

    Range(const std::atomic<bool> &gate, const char *apiname)
    : m_gate(&gate)
    {
        if (!gate.load(std::memory_order_relaxed))
            return;
        rlog::rangePush(apiname, "");
    }

    // Deliberately re-samples the gate rather than latching the push decision.
    // See the header comment: the "unbalanced" call sequence this can produce
    // is intentional and is what makes pause/resume traces correct.
    ~Range()
    {
        if (m_gate->load(std::memory_order_relaxed))
            rlog::rangePop();
    }

    Range(const Range &) = delete;
    Range &operator=(const Range &) = delete;
    Range(Range &&) = delete;
    Range &operator=(Range &&) = delete;

private:
    static const char *cstr(const std::string &s) { return s.c_str(); }
    static const char *cstr(const char *s) { return s; }

    const std::atomic<bool> *m_gate;
};


// Optional convenience: binds the gate once so call sites do not repeat it.
//
//   static rlog::Scope tracer(appActive, "MyApp", "compute");
//   ...
//   auto r = tracer.range("matmul", [&]{ return rlog::fmt("n=%d", n); });
//
// Note the returned Range is immovable, so this requires C++17 guaranteed copy
// elision. Under C++14 construct rlog::Range directly instead.
#if __cplusplus >= 201703L
class Scope
{
public:
    Scope(const std::atomic<bool> &gate, const char *domain, const char *category)
    : m_gate(&gate), m_domain(domain), m_category(category) {}

    template <class Arg>
    Range range(const char *apiname, Arg &&arg) const
    {
        return Range(*m_gate, m_domain, m_category, apiname,
                     std::forward<Arg>(arg));
    }

private:
    const std::atomic<bool> *m_gate;
    const char *m_domain;
    const char *m_category;
};
#endif

} // namespace rlog
