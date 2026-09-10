#!/usr/bin/env python3
# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Python range benchmark: what an idle range costs when no tool is attached.

The C++ benchmark (benchmarks/bench_range.cpp) makes one point: eager argument
evaluation is the entire cost, and a lambda removes it. Python makes two, and
the second is the reason this file exists.

  1. Deferred arguments matter here too, for the same reason -- an f-string
     passed as an argument is built before the range is ever entered.

  2. Unlike C++, the scope machinery itself is not free. A guarded pair of raw
     calls is the floor; the decorator adds a *args/**kw wrapper on top of it,
     and the context manager adds an object allocation plus two method calls.
     No amount of deferring removes that. It is the price of the syntax.

Every measurement below is taken with NO tool attached, which is the case that
matters: it is what an instrumented application pays in production for the
privilege of being instrumentable.

Usage:
    python3 bench_range.py [path/to/librlog.so]
"""

import os
import sys
import timeit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rlog import RlogClient

N = 200_000
REPEAT = 5


def expensive(i):
    """Stand-in for a real argument string: formats five fields."""
    return f"m={i} n={i + 1} k={i + 2} alpha={1.0} beta={0.0}"


def build_variants(client):
    """Each variant is a callable running N iterations of one pattern.

    The loop lives inside the variant so the measurement does not include a
    per-iteration function call that none of the real call sites would pay.
    """

    def empty():
        for i in range(N):
            pass

    def unguarded():
        for i in range(N):
            client.range_push("op", "static args")
            client.range_pop()

    def guarded_literal():
        for i in range(N):
            if client.is_logging:
                client.range_push("op", "static args")
            if client.is_logging:
                client.range_pop()

    def guarded_deferred():
        for i in range(N):
            if client.is_logging:
                client.range_push("op", expensive(i))
            if client.is_logging:
                client.range_pop()

    def guarded_eager():
        for i in range(N):
            text = expensive(i)          # built before the guard is tested
            if client.is_logging:
                client.range_push("op", text)
            if client.is_logging:
                client.range_pop()

    @client.range_decorator(apiname="op", args="static args")
    def dec_literal_fn(i):
        pass

    @client.range_decorator(apiname="op", args=lambda i: expensive(i))
    def dec_deferred_fn(i):
        pass

    def dec_literal():
        for i in range(N):
            dec_literal_fn(i)

    def dec_deferred():
        for i in range(N):
            dec_deferred_fn(i)

    def ctx_eager():
        for i in range(N):
            with client.range("op", expensive(i)):
                pass

    def ctx_deferred():
        for i in range(N):
            with client.range("op", lambda: expensive(i)):
                pass

    return [
        ("empty loop", empty),
        ("raw calls, unguarded", unguarded),
        ("raw calls, guarded, literal", guarded_literal),
        ("raw calls, guarded, eager f-string", guarded_eager),
        ("raw calls, guarded, deferred", guarded_deferred),
        ("decorator, literal args", dec_literal),
        ("decorator, deferred lambda", dec_deferred),
        ("context manager, eager f-string", ctx_eager),
        ("context manager, deferred lambda", ctx_deferred),
    ]


def main():
    lib = sys.argv[1] if len(sys.argv) > 1 else "librlog.so"
    client = RlogClient(lib)

    state = "true" if client.is_logging else "false (baseline, no tool attached)"
    print(f"range: logging = {state}\n")
    print(f"  {N} iterations, best of {REPEAT}\n")

    baseline = None
    for label, fn in build_variants(client):
        best = min(timeit.repeat(fn, number=1, repeat=REPEAT))
        ns = best * 1e9 / N
        if baseline is None:
            baseline = ns
            delta = ""
        else:
            delta = f"  (+{ns - baseline:5.0f} ns over empty)"
        print(f"  {label:36s} {best * 1e3:7.1f} ms  ({ns:5.0f} ns/range){delta}")


if __name__ == "__main__":
    main()
