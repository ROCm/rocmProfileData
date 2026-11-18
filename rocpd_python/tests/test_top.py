# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sqlite3
import unittest

from rocpd.schema import RocpdSchema


class TopOperationsTest(unittest.TestCase):
    def test_coincident_operations_on_different_gpus(self):
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        RocpdSchema().writeSchema(conn)
        conn.executemany("INSERT INTO rocpd_string (id, string) VALUES (?, ?)",
                         [(1, ""), (2, "gemm"), (3, "Copy"), (4, "KernelExecution")])
        conn.executemany("""
            INSERT INTO rocpd_op (gpuId, queueId, sequenceId, start, end, description_id, opType_id)
            VALUES (?, 0, ?, ?, ?, ?, ?)
        """, [(0, 1, 0, 4000, 2, 4), (1, 2, 0, 4000, 2, 4),
              (0, 3, 5000, 7000, 1, 3), (1, 4, 5000, 7000, 1, 3)])

        rows = conn.execute("SELECT Name, TotalCalls, TotalDuration_us, Percentage FROM top ORDER BY Name").fetchall()
        self.assertEqual([row[:3] for row in rows], [("Copy", 2, 4), ("gemm", 2, 8)])
        self.assertAlmostEqual(sum(row[3] for row in rows), 100.0)
