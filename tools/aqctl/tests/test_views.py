import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg  # noqa: E402
import spec as spec_mod  # noqa: E402
import views  # noqa: E402
from planner import RunningContainer  # noqa: E402

MODELS = {
    "models": {
        "primary": {
            "path": "/data/models/Primary", "served_name": "aqueduct-chat",
            "gpus": 4, "weight_gb": 120, "max_model_len": 65536,
        },
        "embed": {
            "path": "/data/models/Embed", "served_name": "aqueduct-embed",
            "gpus": 1, "weight_gb": 16, "task": "embedding",
        },
    }
}

CLUSTER = {
    "defaults": {"model_dir": "/data/models"},
    "hosts": [{
        "name": "gpu-01", "address": "10.0.0.11", "gpu_count": 8,
        "deployments": [
            {"model": "primary", "gpus": [0, 1, 2, 3], "port": 8001},
            {"model": "embed", "gpus": [4], "port": 8003},
        ],
    }],
}


def build():
    cluster = cfg.parse_cluster(CLUSTER, cfg.parse_models(MODELS))
    return cluster, spec_mod.build_all(cluster, "sk-test")


def container_for(spec, state="running", spec_hash=None):
    return RunningContainer(
        host_name=spec.host_name, name=spec.container, model_key=spec.model_key,
        gpus=",".join(str(g) for g in spec.gpus), served_name=spec.served_name,
        spec_hash=spec_hash if spec_hash is not None else spec.spec_hash,
        state=state,
    )


def view(states=None, health=None, metrics=None, series=None, unreachable=None):
    """Render the cluster view with per-container overrides."""
    cluster, specs = build()
    states = states or {}
    containers = {}
    for spec in specs:
        state = states.get(spec.model_key, "running")
        if state == "absent":
            continue
        if state == "stale":
            containers[(spec.host_name, spec.container)] = container_for(
                spec, "running", spec_hash="different")
        else:
            containers[(spec.host_name, spec.container)] = container_for(spec, state)
    return cluster, specs, views.cluster_view(
        cluster, specs, containers,
        health if health is not None else {s.uid: True for s in specs},
        metrics or {}, series=series, unreachable=unreachable or {},
    )


def instance(data, model):
    for host in data["hosts"]:
        for inst in host["instances"]:
            if inst["model"] == model:
                return inst
    raise AssertionError("no instance for " + model)


class GpuOccupancyTest(unittest.TestCase):
    def test_declared_gpus_are_attributed_to_their_model(self):
        _, _, data = view()
        gpus = data["hosts"][0]["gpus"]
        self.assertEqual([g["model"] for g in gpus[:4]], ["primary"] * 4)
        self.assertEqual(gpus[4]["model"], "embed")

    def test_unassigned_gpus_report_as_free(self):
        _, _, data = view()
        free = [g for g in data["hosts"][0]["gpus"] if g["state"] == "free"]
        self.assertEqual([g["id"] for g in free], [5, 6, 7])

    def test_summary_counts_used_versus_total(self):
        _, _, data = view()
        self.assertEqual(data["summary"]["gpus_used"], 5)
        self.assertEqual(data["summary"]["gpus_total"], 8)


class InstanceStateTest(unittest.TestCase):
    def test_running_and_healthy(self):
        _, _, data = view()
        inst = instance(data, "primary")
        self.assertEqual(inst["state"], "running")
        self.assertTrue(inst["healthy"])

    def test_missing_container_is_not_deployed(self):
        _, _, data = view(states={"primary": "absent"})
        self.assertEqual(instance(data, "primary")["state"], "not_deployed")

    def test_stopped_container(self):
        _, _, data = view(states={"primary": "exited"})
        self.assertEqual(instance(data, "primary")["state"], "stopped")

    def test_hash_mismatch_is_stale_config(self):
        _, _, data = view(states={"primary": "stale"})
        self.assertEqual(instance(data, "primary")["state"], "stale_config")

    def test_running_but_failing_health_is_not_healthy(self):
        # The container is up but /health does not answer — a distinction the
        # console must show, since the two look identical in `docker ps`.
        _, specs, data = view(health={s.uid: False for s in specs_of()})
        inst = instance(data, "primary")
        self.assertEqual(inst["state"], "running")
        self.assertFalse(inst["healthy"])
        self.assertEqual(data["summary"]["healthy"], 0)

    def test_unreachable_host_reports_unknown_not_missing(self):
        # Claiming "not deployed" for a host we could not query would be a
        # guess presented as fact.
        _, _, data = view(unreachable={"gpu-01": "ssh refused"})
        host = data["hosts"][0]
        self.assertFalse(host["reachable"])
        self.assertEqual(host["error"], "ssh refused")
        self.assertEqual(instance(data, "primary")["state"], "unknown")
        self.assertEqual(data["summary"]["hosts_unreachable"], 1)


def specs_of():
    return build()[1]


class UndeclaredTest(unittest.TestCase):
    def test_container_not_in_declaration_is_flagged(self):
        cluster, specs = build()
        containers = {(s.host_name, s.container): container_for(s) for s in specs}
        stray = RunningContainer(
            host_name="gpu-01", name="aqueduct-old-8009", model_key="old",
            gpus="6,7", served_name="aqueduct-old", spec_hash="x", state="running")
        containers[("gpu-01", stray.name)] = stray

        data = views.cluster_view(
            cluster, specs, containers,
            {s.uid: True for s in specs}, {})
        undeclared = data["hosts"][0]["undeclared"]
        self.assertEqual([u["container"] for u in undeclared], ["aqueduct-old-8009"])


class MetricsInlineTest(unittest.TestCase):
    def test_window_rides_along_with_its_instance(self):
        # The load curve belongs on the same row as the GPU assignment, not on
        # a separate metrics page.
        window = [{"t": 1, "values": {"running": 2}},
                  {"t": 2, "values": {"running": 5}}]
        _, specs, data = view(
            metrics={"gpu-01/aqueduct-primary-8001": {"running": 5}},
            series={"gpu-01/aqueduct-primary-8001": window})
        inst = instance(data, "primary")
        self.assertEqual(inst["metrics"]["running"], 5)
        self.assertEqual(len(inst["window"]), 2)

    def test_instance_without_samples_gets_an_empty_window(self):
        _, _, data = view()
        self.assertEqual(instance(data, "primary")["window"], [])


class ModelsViewTest(unittest.TestCase):
    def render(self, health=None):
        cluster, specs = build()
        return views.models_view(
            cluster, specs,
            health if health is not None else {s.uid: True for s in specs},
            "http://gw:4000/v1")

    def by_name(self, data, name):
        return next(m for m in data["models"] if m["served_name"] == name)

    def test_healthy_model_is_available(self):
        data = self.render()
        self.assertEqual(self.by_name(data, "aqueduct-chat")["status"], "available")

    def test_all_replicas_down_is_down(self):
        cluster, specs = build()
        data = views.models_view(
            cluster, specs, {s.uid: False for s in specs}, "http://gw:4000/v1")
        self.assertEqual(self.by_name(data, "aqueduct-chat")["status"], "down")

    def test_partial_replicas_is_degraded(self):
        two_hosts = {
            "defaults": {"model_dir": "/data/models"},
            "hosts": [
                {"name": "gpu-01", "address": "10.0.0.11", "gpu_count": 8,
                 "deployments": [{"model": "primary", "gpus": [0, 1, 2, 3],
                                  "port": 8001}]},
                {"name": "gpu-02", "address": "10.0.0.12", "gpu_count": 8,
                 "deployments": [{"model": "primary", "gpus": [0, 1, 2, 3],
                                  "port": 8001}]},
            ],
        }
        cluster = cfg.parse_cluster(two_hosts, cfg.parse_models(MODELS))
        specs = spec_mod.build_all(cluster, "sk-test")
        # Same container name on both hosts — only the uid distinguishes them.
        health = {specs[0].uid: True}
        data = views.models_view(cluster, specs, health, "http://gw:4000/v1")
        chat = self.by_name(data, "aqueduct-chat")
        self.assertEqual(chat["replicas"], 2)
        self.assertEqual(chat["status"], "degraded")
        self.assertEqual(chat["hosts"], ["gpu-01", "gpu-02"])

    def test_model_defined_but_never_placed(self):
        doc = dict(MODELS)
        doc["models"] = dict(MODELS["models"])
        doc["models"]["unused"] = {
            "path": "/data/models/Unused", "served_name": "aqueduct-unused",
            "gpus": 1, "weight_gb": 8, "max_model_len": 4096,
        }
        cluster = cfg.parse_cluster(CLUSTER, cfg.parse_models(doc))
        specs = spec_mod.build_all(cluster, "sk-test")
        data = views.models_view(
            cluster, specs, {s.uid: True for s in specs}, "http://gw:4000/v1")
        self.assertEqual(
            self.by_name(data, "aqueduct-unused")["status"], "not_deployed")

    def test_embedding_model_is_labelled_by_kind(self):
        data = self.render()
        self.assertEqual(self.by_name(data, "aqueduct-embed")["kind"], "embedding")

    def test_parameter_matrix_is_included(self):
        self.assertTrue(len(self.render()["parameters"]) > 5)


class UsageViewTest(unittest.TestCase):
    def test_percent_used_is_computed_from_budget(self):
        data = views.usage_view([
            {"key_alias": "aqueduct-search", "spend": 50, "max_budget": 200},
        ])
        self.assertEqual(data["keys"][0]["percent_used"], 25.0)

    def test_key_near_its_budget_needs_attention(self):
        data = views.usage_view([
            {"key_alias": "runaway", "spend": 90, "max_budget": 100},
            {"key_alias": "fine", "spend": 5, "max_budget": 100},
        ])
        self.assertEqual([k["alias"] for k in data["needs_attention"]], ["runaway"])

    def test_key_without_a_budget_has_no_percentage(self):
        data = views.usage_view([{"key_alias": "unlimited", "spend": 12}])
        self.assertIsNone(data["keys"][0]["percent_used"])
        self.assertEqual(data["needs_attention"], [])

    def test_keys_are_sorted_by_spend(self):
        data = views.usage_view([
            {"key_alias": "small", "spend": 1},
            {"key_alias": "big", "spend": 99},
        ])
        self.assertEqual([k["alias"] for k in data["keys"]], ["big", "small"])

    def test_totals_add_up(self):
        data = views.usage_view([
            {"key_alias": "a", "spend": 1.5}, {"key_alias": "b", "spend": 2.25},
        ])
        self.assertEqual(data["total_spend"], 3.75)

    def test_gateway_error_is_surfaced_not_swallowed(self):
        data = views.usage_view([], error="gateway unreachable")
        self.assertEqual(data["error"], "gateway unreachable")
        self.assertEqual(data["keys"], [])

    def test_unnamed_key_still_renders(self):
        data = views.usage_view([{"spend": 3}])
        self.assertEqual(data["keys"][0]["alias"], "(unnamed)")



class UndeclaredGpuTest(unittest.TestCase):
    """A container outside the declaration still holds its cards."""

    def _view_with_stray(self, gpus_label, state="running"):
        cluster, specs = build()
        containers = {(s.host_name, s.container): container_for(s) for s in specs}
        containers[("gpu-01", "aqueduct-legacy-8009")] = RunningContainer(
            host_name="gpu-01", name="aqueduct-legacy-8009", model_key="legacy",
            gpus=gpus_label, served_name="aqueduct-legacy", spec_hash="old",
            state=state)
        return views.cluster_view(
            cluster, specs, containers, {s.uid: True for s in specs}, {})

    def test_stray_container_marks_its_gpus_occupied(self):
        # Showing these as free would advertise capacity that is not there,
        # and the next apply would fail trying to claim them.
        data = self._view_with_stray("7")
        gpu7 = data["hosts"][0]["gpus"][7]
        self.assertEqual(gpu7["state"], views.UNDECLARED)
        self.assertEqual(data["summary"]["gpus_used"], 6)

    def test_stopped_stray_does_not_hold_gpus(self):
        data = self._view_with_stray("7", state="exited")
        self.assertEqual(data["hosts"][0]["gpus"][7]["state"], "free")

    def test_stray_never_overrides_a_declared_assignment(self):
        # If both claim GPU 0, the declaration is what the operator asked for.
        data = self._view_with_stray("0,7")
        self.assertEqual(data["hosts"][0]["gpus"][0]["model"], "primary")
        self.assertEqual(data["hosts"][0]["gpus"][7]["state"], views.UNDECLARED)

    def test_unparseable_gpu_label_is_tolerated(self):
        # A container we did not create may carry no usable label.
        data = self._view_with_stray("")
        self.assertEqual(data["hosts"][0]["undeclared"][0]["gpus"], [])
        self.assertEqual(data["summary"]["gpus_used"], 5)

    def test_gpu_label_parsing(self):
        self.assertEqual(views.parse_gpu_label("0,1,2,3"), [0, 1, 2, 3])
        self.assertEqual(views.parse_gpu_label(" 4 , 5 "), [4, 5])
        self.assertEqual(views.parse_gpu_label("all"), [])
        self.assertEqual(views.parse_gpu_label(None), [])


class MetricRoundingTest(unittest.TestCase):
    """vLLM reports request counts as float gauges."""

    def test_float_gauges_survive_the_view_layer(self):
        # The view passes them through; rounding is the renderer's job, but the
        # values must arrive intact so it has something to round.
        _, _, data = view(metrics={"gpu-01/aqueduct-primary-8001":
                                   {"running": 5.6, "waiting": 0.0}})
        self.assertAlmostEqual(instance(data, "primary")["metrics"]["running"], 5.6)

if __name__ == "__main__":
    unittest.main()
