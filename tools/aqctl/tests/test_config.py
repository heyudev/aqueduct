import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg  # noqa: E402


def models_doc(**overrides):
    model = {
        "path": "/data/models/Primary",
        "served_name": "aqueduct-chat",
        "gpus": 8,
        "weight_gb": 235,
        "max_model_len": 65536,
    }
    model.update(overrides)
    return {"models": {"primary": model}}


def cluster_doc(deployments=None, gpu_count=8, **host_overrides):
    host = {
        "name": "gpu-01",
        "address": "10.0.0.11",
        "gpu_count": gpu_count,
        "deployments": deployments
        if deployments is not None
        else [{"model": "primary", "gpus": [0, 1, 2, 3, 4, 5, 6, 7], "port": 8001}],
    }
    host.update(host_overrides)
    return {"defaults": {"model_dir": "/data/models"}, "hosts": [host]}


def build(models=None, cluster=None):
    parsed_models = cfg.parse_models(models or models_doc())
    return cfg.parse_cluster(cluster or cluster_doc(), parsed_models)


class ParsingTest(unittest.TestCase):
    def test_defaults_flow_into_hosts(self):
        c = build()
        self.assertEqual(c.hosts[0].model_dir, "/data/models")
        self.assertEqual(c.hosts[0].gpu_memory_gb, cfg.DEFAULT_GPU_MEMORY_GB)

    def test_host_overrides_defaults(self):
        doc = cluster_doc(image="custom:tag")
        c = build(cluster=doc)
        self.assertEqual(c.hosts[0].image, "custom:tag")

    def test_missing_required_field_is_an_error(self):
        doc = models_doc()
        del doc["models"]["primary"]["served_name"]
        with self.assertRaises(cfg.ConfigError):
            cfg.parse_models(doc)

    def test_extra_args_as_string_is_rejected(self):
        with self.assertRaises(cfg.ConfigError):
            cfg.parse_models(models_doc(extra_args="--enable-expert-parallel"))

    def test_instances_pairs_hosts_with_models(self):
        c = build()
        pairs = list(c.instances())
        self.assertEqual(len(pairs), 1)
        host, dep, model = pairs[0]
        self.assertEqual(host.name, "gpu-01")
        self.assertEqual(model.key, "primary")
        self.assertEqual(dep.port, 8001)


class ValidationTest(unittest.TestCase):
    def assertValid(self, cluster):
        errors = cfg.validate(cluster)
        self.assertEqual(errors, [], "expected valid, got: %s" % errors)

    def assertErrorMatching(self, cluster, fragment):
        errors = cfg.validate(cluster)
        self.assertTrue(
            any(fragment in e for e in errors),
            "expected an error containing %r, got: %s" % (fragment, errors),
        )

    def test_reference_config_is_valid(self):
        self.assertValid(build())

    def test_overlapping_gpus_rejected(self):
        models = models_doc()
        models["models"]["fast"] = {
            "path": "/data/models/Fast",
            "served_name": "aqueduct-fast",
            "gpus": 2,
            "weight_gb": 30,
            "max_model_len": 32768,
        }
        doc = cluster_doc(
            deployments=[
                {"model": "primary", "gpus": [0, 1, 2, 3], "port": 8001},
                {"model": "fast", "gpus": [3, 4], "port": 8002},
            ]
        )
        models["models"]["primary"]["gpus"] = 4
        models["models"]["primary"]["weight_gb"] = 120
        self.assertErrorMatching(build(models, doc), "already assigned")

    def test_gpu_out_of_range_rejected(self):
        doc = cluster_doc(
            deployments=[{"model": "primary", "gpus": [4, 5, 6, 7, 8, 9, 10, 11],
                          "port": 8001}]
        )
        self.assertErrorMatching(build(cluster=doc), "outside the host's range")

    def test_duplicate_port_rejected(self):
        models = models_doc(gpus=4, weight_gb=120)
        models["models"]["primary"]["gpus"] = 4
        doc = cluster_doc(
            deployments=[
                {"model": "primary", "gpus": [0, 1, 2, 3], "port": 8001},
                {"model": "primary", "gpus": [4, 5, 6, 7], "port": 8001},
            ]
        )
        self.assertErrorMatching(build(models, doc), "already used")

    def test_gpu_count_must_match_model_requirement(self):
        doc = cluster_doc(
            deployments=[{"model": "primary", "gpus": [0, 1, 2, 3], "port": 8001}]
        )
        self.assertErrorMatching(build(cluster=doc), "requires 8")

    def test_non_power_of_two_tp_rejected(self):
        models = models_doc(gpus=6, weight_gb=120)
        doc = cluster_doc(
            deployments=[{"model": "primary", "gpus": [0, 1, 2, 3, 4, 5], "port": 8001}]
        )
        self.assertErrorMatching(build(models, doc), "tensor-parallel size")

    def test_unknown_model_rejected(self):
        doc = cluster_doc(
            deployments=[{"model": "nope", "gpus": [0], "port": 8001}]
        )
        self.assertErrorMatching(build(cluster=doc), "no such model")

    def test_weights_larger_than_memory_rejected(self):
        models = models_doc(weight_gb=900)
        self.assertErrorMatching(build(models), "weights need")

    def test_insufficient_kv_cache_headroom_rejected(self):
        # Fits, but leaves too little for KV cache to be usable.
        usable = 8 * 96 * cfg.DEFAULT_MEMORY_UTIL
        models = models_doc(weight_gb=usable - 10)
        self.assertErrorMatching(build(models), "no usable")

    def test_671b_class_model_rejected_on_one_host(self):
        # The real reason DeepSeek-V3-class models are out of scope for a
        # single 8x96GB host: the weights technically fit, but only ~36GB is
        # left for KV cache. Catching this at validation is the whole point of
        # the headroom rule — it would otherwise start and then be useless.
        models = models_doc(weight_gb=671)
        self.assertErrorMatching(build(models), "no usable concurrency")

    def test_model_path_outside_model_dir_rejected(self):
        models = models_doc(path="/elsewhere/Primary")
        self.assertErrorMatching(build(models), "outside the host's model_dir")

    def test_generative_model_needs_max_model_len(self):
        models = models_doc()
        del models["models"]["primary"]["max_model_len"]
        self.assertErrorMatching(build(models), "max_model_len is required")

    def test_embedding_model_needs_no_max_model_len(self):
        doc = {
            "models": {
                "embed": {
                    "path": "/data/models/Embed",
                    "served_name": "aqueduct-embed",
                    "gpus": 1,
                    "weight_gb": 16,
                    "task": "embedding",
                }
            }
        }
        cluster = cluster_doc(
            deployments=[{"model": "embed", "gpus": [0], "port": 8003}]
        )
        self.assertValid(build(doc, cluster))

    def test_two_models_claiming_one_served_name_rejected(self):
        models = models_doc()
        models["models"]["primary_v2"] = dict(models["models"]["primary"])
        self.assertErrorMatching(build(models), "claimed by multiple models")

    def test_duplicate_host_names_rejected(self):
        doc = cluster_doc()
        doc["hosts"].append(dict(doc["hosts"][0]))
        self.assertErrorMatching(build(cluster=doc), "duplicate host name")



class NominalCostTest(unittest.TestCase):
    """Budgets only work if models have a price — see config.DEFAULT_COST_*."""

    def test_models_get_a_positive_default_price(self):
        models = cfg.parse_models(models_doc())
        self.assertGreater(models["primary"].cost_per_1k_input, 0)
        self.assertGreater(models["primary"].cost_per_1k_output, 0)

    def test_output_costs_more_than_input(self):
        # Decoding occupies the GPU far longer than prefill does.
        m = cfg.parse_models(models_doc())["primary"]
        self.assertGreater(m.cost_per_1k_output, m.cost_per_1k_input)

    def test_explicit_price_overrides_the_default(self):
        models = cfg.parse_models(models_doc(cost_per_1k_input=5.0))
        self.assertEqual(models["primary"].cost_per_1k_input, 5.0)

    def test_zero_price_is_rejected(self):
        # A model priced at zero accrues no spend, so its budget never fires.
        errors = cfg.validate(build(models_doc(cost_per_1k_input=0)))
        self.assertTrue(any("never accrues" in e for e in errors), errors)

if __name__ == "__main__":
    unittest.main()
