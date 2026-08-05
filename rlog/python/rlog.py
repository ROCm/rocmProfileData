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
