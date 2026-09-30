"""Regression tests for least-loaded dispatch spreading.

Before the fix, ``_dispatch_tasks`` scored workers only by heartbeat metrics
(which stay constant within a tick), so every pending task in one tick landed
on the same worker and the per-worker capacity cap was never re-checked.
"""

import collections
import shutil
import tempfile
import unittest

from backend.common import constants as C
from backend.common.config import ClusterConfig
from backend.common.logbus import LogBus
from backend.common.storage import Storage
from backend.master.fault_tolerance import FaultTolerance
from backend.master.job_manager import JobManager
from backend.master.metrics import Metrics
from backend.master.registry import WorkerRegistry
from backend.master.scheduler import MAX_TASKS_PER_WORKER, Scheduler
from backend.master.shuffle import ShuffleCoordinator


class TestDispatchSpreading(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.storage = Storage(self.tmp)
        self.config = ClusterConfig()
        self.logbus = LogBus(self.storage)
        self.jm = JobManager(self.storage, self.config, self.logbus)
        self.registry = WorkerRegistry(self.storage, self.config)
        for i in range(3):
            self.registry.register({
                "worker_id": f"w{i}", "name": f"w{i}", "host": "127.0.0.1",
                "port": 9000 + i, "cpu_cores": 4, "mem_total_mb": 1024,
            })
        shuffle = ShuffleCoordinator(self.storage, self.jm, self.registry, self.logbus)
        ft = FaultTolerance(self.storage, self.jm, self.config, self.logbus)
        self.sched = Scheduler(self.storage, self.jm, self.registry, shuffle, ft,
                               Metrics(self.storage), self.config, self.logbus)
        # Record dispatches instead of performing real HTTP calls.
        self.dispatched: list[tuple[str, str]] = []
        self.sched._dispatch = lambda job, task, worker, speculative=False: (
            self.dispatched.append((task.task_id, worker.worker_id)))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _submit(self, num_map: int):
        return self.jm.submit({
            "name": "t", "mapper": "wordcount_mapper", "reducer": "count_reducer",
            "num_map_tasks": num_map, "num_reduce_tasks": 1, "input_rows": 100,
            "params": {},
        })

    def test_tasks_spread_across_workers(self):
        job = self._submit(num_map=9)
        self.sched._dispatch_tasks(job, C.TASK_MAP)
        self.assertEqual(len(self.dispatched), 9)
        per_worker = collections.Counter(wid for _, wid in self.dispatched)
        self.assertEqual(sorted(per_worker.values()), [3, 3, 3])

    def test_per_worker_capacity_enforced_within_tick(self):
        job = self._submit(num_map=MAX_TASKS_PER_WORKER * 3 + 1)
        self.sched._dispatch_tasks(job, C.TASK_MAP)
        per_worker = collections.Counter(wid for _, wid in self.dispatched)
        self.assertEqual(len(self.dispatched), MAX_TASKS_PER_WORKER * 3)
        self.assertTrue(all(n <= MAX_TASKS_PER_WORKER for n in per_worker.values()))


if __name__ == "__main__":
    unittest.main()
