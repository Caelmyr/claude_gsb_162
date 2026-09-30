"""Regression tests: pending tasks must be balanced across workers.

Before the fix, ``_dispatch_tasks`` computed the available-worker list once
and scored workers purely by (stale) heartbeat metrics, so every pending map
task in a single tick went to the same worker.
"""

import shutil
import tempfile
import unittest
from unittest import mock

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


class _FakeResp:
    ok = True
    data = {"accepted": True}


class TestDispatchBalancing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        storage = Storage(self.tmp)
        logbus = LogBus(storage)
        config = ClusterConfig()
        self.jm = JobManager(storage, config, logbus)
        self.registry = WorkerRegistry(storage, config)
        shuffle = ShuffleCoordinator(storage, self.jm, self.registry, logbus)
        ft = FaultTolerance(storage, self.jm, config, logbus)
        self.sched = Scheduler(storage, self.jm, self.registry, shuffle, ft,
                               Metrics(storage), config, logbus)
        # Pretend every worker accepts every dispatch.
        self.sched.client = mock.Mock()
        self.sched.client.post.return_value = _FakeResp()
        for i in range(3):
            self.registry.register({
                "worker_id": f"w{i}", "name": f"w{i}",
                "host": "127.0.0.1", "port": 9000 + i,
                "cpu_cores": 8, "mem_total_mb": 1024,
            })

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _submit(self, num_map):
        return self.jm.submit({
            "name": "t", "mapper": "wordcount_mapper", "reducer": "count_reducer",
            "num_map_tasks": num_map, "num_reduce_tasks": 2,
            "input_rows": num_map * 100, "params": {},
        })

    def test_map_tasks_spread_across_workers_within_one_tick(self):
        job = self._submit(num_map=9)
        self.sched._dispatch_tasks(job, C.TASK_MAP)

        counts: dict[str, int] = {}
        for t in self.jm.tasks_for(job.job_id, C.TASK_MAP):
            self.assertEqual(t.status, C.TASK_ASSIGNED)
            counts[t.worker_id] = counts.get(t.worker_id, 0) + 1

        # Every worker is used and none exceeds the per-worker cap.
        self.assertEqual(len(counts), 3)
        self.assertEqual(sum(counts.values()), 9)
        for n in counts.values():
            self.assertLessEqual(n, MAX_TASKS_PER_WORKER)

    def test_excess_tasks_wait_for_capacity(self):
        # 12 map tasks but only 3 workers x MAX_TASKS_PER_WORKER slots: the
        # remainder must stay pending instead of piling onto one node.
        job = self._submit(num_map=3 * MAX_TASKS_PER_WORKER + 2)
        self.sched._dispatch_tasks(job, C.TASK_MAP)

        tasks = self.jm.tasks_for(job.job_id, C.TASK_MAP)
        assigned = [t for t in tasks if t.status == C.TASK_ASSIGNED]
        pending = [t for t in tasks if t.status == C.TASK_PENDING]
        self.assertEqual(len(assigned), 3 * MAX_TASKS_PER_WORKER)
        self.assertEqual(len(pending), 2)


if __name__ == "__main__":
    unittest.main()
