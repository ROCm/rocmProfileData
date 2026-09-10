# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import ctypes
import ctypes.util

_CALLBACK = ctypes.CFUNCTYPE(None)


class RlogClient:

    def __init__(self, lib_path="librlog.so"):
        self._lib = ctypes.CDLL(lib_path)

        self._lib.rlog_mark.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                        ctypes.c_char_p, ctypes.c_char_p]
        self._lib.rlog_mark.restype = None

        self._lib.rlog_rangePush.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                             ctypes.c_char_p, ctypes.c_char_p]
        self._lib.rlog_rangePush.restype = None

        self._lib.rlog_rangePop.argtypes = []
        self._lib.rlog_rangePop.restype = None

        self._lib.rlog_isActive.argtypes = []
        self._lib.rlog_isActive.restype = ctypes.c_bool

        self._lib.rlog_registerActiveCallback.argtypes = [_CALLBACK]
        self._lib.rlog_registerActiveCallback.restype = None

        self._lib.rlog_getProperty.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                               ctypes.c_char_p]
        self._lib.rlog_getProperty.restype = ctypes.c_char_p

        self._default_domain = b""
        self._default_category = b""

        self._callbacks = []

        self.is_logging = False
        self._active_cb = _CALLBACK(self._on_active_changed)
        self._callbacks.append(self._active_cb)
        self._lib.rlog_registerActiveCallback(self._active_cb)

    def _on_active_changed(self):
        self.is_logging = self._lib.rlog_isActive()

    @staticmethod
    def _encode(s):
        if s is None:
            return None
        return s.encode("utf-8") if isinstance(s, str) else s

    def set_default_domain(self, domain):
        self._default_domain = self._encode(domain) or b""

    def set_default_category(self, category):
        self._default_category = self._encode(category) or b""

    def mark(self, apiname, args, domain=None, category=None):
        d = self._encode(domain) if domain is not None else self._default_domain
        c = self._encode(category) if category is not None else self._default_category
        self._lib.rlog_mark(d, c, self._encode(apiname), self._encode(args))

    def range_push(self, apiname, args, domain=None, category=None):
        d = self._encode(domain) if domain is not None else self._default_domain
        c = self._encode(category) if category is not None else self._default_category
        self._lib.rlog_rangePush(d, c, self._encode(apiname), self._encode(args))

    def range_pop(self):
        self._lib.rlog_rangePop()

    def is_active(self):
        return self._lib.rlog_isActive()

    def register_active_callback(self, cb):
        wrapped = _CALLBACK(cb)
        self._callbacks.append(wrapped)
        self._lib.rlog_registerActiveCallback(wrapped)

    def get_property(self, domain, prop, default_value):
        result = self._lib.rlog_getProperty(
            self._encode(domain), self._encode(prop), self._encode(default_value))
        if result is None:
            return default_value
        return result.decode("utf-8")

    # -- scope helpers --------------------------------------------------
    # See OPTIMIZATIONS.md. Prefer the decorator to the context manager:
    # a decorator's arguments are the wrapped function's own arguments,
    # already evaluated by the caller, so there is nothing to defer. A
    # context manager evaluates its arguments before __enter__, which
    # reintroduces the eager-argument cost.
    #
    # Both sample is_logging twice, independently, and deliberately do NOT
    # latch the push decision. If tracing resumes mid-range the pop is
    # still delivered; that orphan pop is the only evidence the range
    # existed and is what lets the tool keep descendants correctly nested.

    def range(self, apiname, args=None, domain=None, category=None):
        """Context manager for a range.

        `args` may be a str, or a zero-argument callable that is only
        invoked while logging is active. Use the callable form whenever
        producing the string costs anything.
        """
        return _RangeScope(self, apiname, args, domain, category)

    def range_decorator(self, apiname=None, args=None, domain=None, category=None):
        """Decorator that wraps a function in a range.

        `apiname` defaults to the function's qualified name. `args` may be
        a str, or a callable receiving the same arguments as the wrapped
        function, invoked only while logging is active.
        """
        import functools

        def decorate(fn):
            name = apiname if apiname is not None else fn.__qualname__

            @functools.wraps(fn)
            def wrapper(*a, **kw):
                if self.is_logging:
                    text = args(*a, **kw) if callable(args) else (args or "")
                    self.range_push(name, text, domain=domain, category=category)
                try:
                    return fn(*a, **kw)
                finally:
                    # Sampled again on purpose - do not latch.
                    if self.is_logging:
                        self.range_pop()

            return wrapper

        return decorate


class _RangeScope:
    __slots__ = ("_c", "_apiname", "_args", "_domain", "_category")

    def __init__(self, client, apiname, args, domain, category):
        self._c = client
        self._apiname = apiname
        self._args = args
        self._domain = domain
        self._category = category

    def __enter__(self):
        if self._c.is_logging:
            args = self._args
            text = args() if callable(args) else (args or "")
            self._c.range_push(self._apiname, text,
                               domain=self._domain, category=self._category)
        return self

    def __exit__(self, *exc):
        # Sampled again on purpose - do not latch. See OPTIMIZATIONS.md.
        if self._c.is_logging:
            self._c.range_pop()
        return False
