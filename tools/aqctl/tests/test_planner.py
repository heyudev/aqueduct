import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import planner  # noqa: E402
from planner import RunningContainer  # noqa: E402


class FakeSpec(object):
    """Stands in for InstanceSpec — diff only reads these four fields."""

    def __init__(self, host_name, container, spec_hash, gpus=(0,)):
        self.host_name = host_name
        self.container = container
        self.spec_hash = spec_hash
        self.gpus = gpus


def running(host, name, spec_hash, state="running"):
    return RunningContainer(
        host_name=host, name=name, model_key="primary", gpus="0",
        served_name="aqueduct-chat", spec_hash=spec_hash, state=state,
    )


class DiffTest(unittest.TestCase):
    def kinds(self, plan):
        return sorted((a.kind, a.container) for a in plan.actions)

    def test_nothing_running_means_create_everything(self):
        plan = planner.diff([FakeSpec("h1", "c1", "aaa")], [])
        self.assertEqual(self.kinds(plan), [(planner.CREATE, "c1")])
        self.assertFalse(plan.empty)

    def test_matching_hash_is_unchanged(self):
        plan = planner.diff(
            [FakeSpec("h1", "c1", "aaa")], [running("h1", "c1", "aaa")]
        )
        self.assertEqual(self.kinds(plan), [(planner.UNCHANGED, "c1")])
        self.assertTrue(plan.empty)

    def test_changed_hash_triggers_recreate(self):
        plan = planner.diff(
            [FakeSpec("h1", "c1", "bbb")], [running("h1", "c1", "aaa")]
        )
        self.assertEqual(self.kinds(plan), [(planner.RECREATE, "c1")])

    def test_undeclared_container_is_removed(self):
        plan = planner.diff([], [running("h1", "orphan", "aaa")])
        self.assertEqual(self.kinds(plan), [(planner.REMOVE, "orphan")])

    def test_stopped_container_with_matching_hash_is_restarted(self):
        plan = planner.diff(
            [FakeSpec("h1", "c1", "aaa")],
            [running("h1", "c1", "aaa", state="exited")],
        )
        self.assertEqual(self.kinds(plan), [(planner.RESTART, "c1")])

    def test_stopped_container_with_stale_hash_is_recreated(self):
        plan = planner.diff(
            [FakeSpec("h1", "c1", "bbb")],
            [running("h1", "c1", "aaa", state="exited")],
        )
        self.assertEqual(self.kinds(plan), [(planner.RECREATE, "c1")])

    def test_same_container_name_on_different_hosts_is_independent(self):
        plan = planner.diff(
            [FakeSpec("h1", "c1", "aaa"), FakeSpec("h2", "c1", "aaa")],
            [running("h1", "c1", "aaa")],
        )
        self.assertEqual(
            self.kinds(plan),
            [(planner.CREATE, "c1"), (planner.UNCHANGED, "c1")],
        )

    def test_moving_a_model_to_a_new_port_is_remove_plus_create(self):
        # Container names encode the port, so a port change cannot be an
        # in-place update — the old one has to go before the new one binds.
        plan = planner.diff(
            [FakeSpec("h1", "aqueduct-primary-8002", "aaa")],
            [running("h1", "aqueduct-primary-8001", "aaa")],
        )
        self.assertEqual(
            self.kinds(plan),
            [
                (planner.CREATE, "aqueduct-primary-8002"),
                (planner.REMOVE, "aqueduct-primary-8001"),
            ],
        )


class DisruptionTest(unittest.TestCase):
    def test_create_is_not_disruptive(self):
        plan = planner.diff([FakeSpec("h1", "c1", "aaa")], [])
        self.assertEqual(plan.disruptions, [])

    def test_recreate_is_flagged_as_disruptive(self):
        plan = planner.diff(
            [FakeSpec("h1", "c1", "bbb")], [running("h1", "c1", "aaa")]
        )
        self.assertEqual(len(plan.disruptions), 1)

    def test_plan_text_warns_about_downtime(self):
        plan = planner.diff(
            [FakeSpec("h1", "c1", "bbb")], [running("h1", "c1", "aaa")]
        )
        text = planner.format_plan(plan)
        self.assertIn("stop serving", text)
        self.assertIn("c1", text)


class ExecutionOrderTest(unittest.TestCase):
    def test_removals_precede_creations(self):
        # Freeing GPUs has to happen before anything tries to claim them.
        plan = planner.diff(
            [FakeSpec("h1", "new", "bbb")], [running("h1", "old", "aaa")]
        )
        removals, creations = planner.execution_order(plan)
        self.assertEqual([a.container for a in removals], ["old"])
        self.assertEqual([a.container for a in creations], ["new"])

    def test_recreate_appears_in_both_phases(self):
        plan = planner.diff(
            [FakeSpec("h1", "c1", "bbb")], [running("h1", "c1", "aaa")]
        )
        removals, creations = planner.execution_order(plan)
        self.assertEqual([a.container for a in removals], ["c1"])
        self.assertEqual([a.container for a in creations], ["c1"])


class FormatTest(unittest.TestCase):
    def test_no_changes_says_so(self):
        plan = planner.diff(
            [FakeSpec("h1", "c1", "aaa")], [running("h1", "c1", "aaa")]
        )
        self.assertIn("Nothing to do", planner.format_plan(plan))


if __name__ == "__main__":
    unittest.main()
