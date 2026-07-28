import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg  # noqa: E402
import spec as spec_mod  # noqa: E402

MODELS = {
    "models": {
        "primary": {
            "path": "/data/models/Primary",
            "served_name": "aqueduct-chat",
            "gpus": 4,
            "weight_gb": 120,
            "max_model_len": 65536,
            "kv_cache_dtype": "fp8",
            "max_num_batched_tokens": 4096,
            "max_num_seqs": 64,
            "extra_args": ["--enable-expert-parallel"],
        },
        "embed": {
            "path": "/data/models/Embed",
            "served_name": "aqueduct-embed",
            "gpus": 1,
            "weight_gb": 16,
            "task": "embedding",
        },
    }
}

CLUSTER = {
    "defaults": {"model_dir": "/data/models", "image": "vllm/vllm-openai:v1"},
    "hosts": [
        {
            "name": "gpu-01",
            "address": "10.0.0.11",
            "gpu_count": 8,
            "deployments": [
                {"model": "primary", "gpus": [0, 1, 2, 3], "port": 8001},
                {"model": "embed", "gpus": [4], "port": 8003},
            ],
        }
    ],
}


def build_cluster():
    return cfg.parse_cluster(CLUSTER, cfg.parse_models(MODELS))


def spec_for(model_key, api_key="sk-test"):
    cluster = build_cluster()
    for host, dep, model in cluster.instances():
        if dep.model == model_key:
            return spec_mod.build(host, dep, model, api_key)
    raise KeyError(model_key)


def arg_value(args, flag):
    return args[args.index(flag) + 1]


class RunArgsTest(unittest.TestCase):
    def test_tensor_parallel_matches_assigned_gpu_count(self):
        args = list(spec_for("primary").run_args)
        self.assertEqual(arg_value(args, "--tensor-parallel-size"), "4")

    def test_container_is_pinned_to_the_declared_gpus(self):
        args = list(spec_for("primary").run_args)
        self.assertEqual(arg_value(args, "--gpus"), "device=0,1,2,3")

    def test_port_is_published_from_the_fixed_container_port(self):
        args = list(spec_for("primary").run_args)
        self.assertEqual(arg_value(args, "-p"), "0.0.0.0:8001:8000")

    def test_model_dir_is_mounted_at_the_same_path(self):
        # The declared model path is absolute on the host; mounting it at the
        # identical path inside means no translation and matching log output.
        args = list(spec_for("primary").run_args)
        self.assertEqual(arg_value(args, "-v"), "/data/models:/data/models:ro")
        self.assertEqual(arg_value(args, "--model"), "/data/models/Primary")

    def test_shared_memory_is_set(self):
        # The docker default of 64MB makes tensor-parallel startup hang.
        args = list(spec_for("primary").run_args)
        self.assertEqual(arg_value(args, "--shm-size"), "64gb")

    def test_h20_tuning_flags_are_present(self):
        args = list(spec_for("primary").run_args)
        self.assertIn("--enable-prefix-caching", args)
        self.assertIn("--enable-chunked-prefill", args)
        self.assertEqual(arg_value(args, "--max-num-batched-tokens"), "4096")
        self.assertEqual(arg_value(args, "--kv-cache-dtype"), "fp8")

    def test_extra_args_are_appended(self):
        self.assertIn("--enable-expert-parallel", spec_for("primary").run_args)

    def test_labels_carry_identity_and_hash(self):
        args = list(spec_for("primary").run_args)
        labels = [args[i + 1] for i, a in enumerate(args) if a == "-l"]
        self.assertIn("aqueduct.managed=true", labels)
        self.assertIn("aqueduct.model=primary", labels)
        self.assertIn("aqueduct.gpus=0,1,2,3", labels)
        self.assertIn("aqueduct.served-name=aqueduct-chat", labels)
        self.assertTrue(any(l.startswith("aqueduct.spec-hash=") for l in labels))


class EmbeddingModelTest(unittest.TestCase):
    def test_pooling_flags_replace_generation_flags(self):
        args = list(spec_for("embed").run_args)
        self.assertIn("--runner", args)
        self.assertEqual(arg_value(args, "--convert"), "embed")
        # Context length and scheduling flags do not apply to pooling models.
        self.assertNotIn("--max-model-len", args)
        self.assertNotIn("--enable-chunked-prefill", args)


class SpecHashTest(unittest.TestCase):
    def test_identical_declarations_hash_identically(self):
        self.assertEqual(spec_for("primary").spec_hash,
                         spec_for("primary").spec_hash)

    def test_api_key_rotation_does_not_change_the_hash(self):
        # Otherwise rotating the shared secret would recreate every instance
        # in the cluster.
        self.assertEqual(
            spec_for("primary", api_key="sk-one").spec_hash,
            spec_for("primary", api_key="sk-two").spec_hash,
        )

    def test_api_key_still_reaches_the_container(self):
        args = list(spec_for("primary", api_key="sk-one").run_args)
        self.assertEqual(arg_value(args, "--api-key"), "sk-one")

    def _hash_with_model_change(self, **overrides):
        import copy

        doc = copy.deepcopy(MODELS)
        doc["models"]["primary"].update(overrides)
        cluster = cfg.parse_cluster(CLUSTER, cfg.parse_models(doc))
        for host, dep, model in cluster.instances():
            if dep.model == "primary":
                return spec_mod.build(host, dep, model, "sk-test").spec_hash
        raise AssertionError

    def test_changing_a_tuning_parameter_changes_the_hash(self):
        baseline = spec_for("primary").spec_hash
        self.assertNotEqual(baseline, self._hash_with_model_change(max_model_len=32768))
        self.assertNotEqual(baseline, self._hash_with_model_change(kv_cache_dtype="auto"))
        self.assertNotEqual(baseline, self._hash_with_model_change(max_num_seqs=128))

    def test_changing_the_image_changes_the_hash(self):
        import copy

        doc = copy.deepcopy(CLUSTER)
        doc["defaults"]["image"] = "vllm/vllm-openai:v2"
        cluster = cfg.parse_cluster(doc, cfg.parse_models(MODELS))
        host, dep, model = next(
            t for t in cluster.instances() if t[1].model == "primary")
        changed = spec_mod.build(host, dep, model, "sk-test").spec_hash
        self.assertNotEqual(spec_for("primary").spec_hash, changed)

    def test_changing_assigned_gpus_changes_the_hash(self):
        import copy

        doc = copy.deepcopy(CLUSTER)
        doc["hosts"][0]["deployments"][0]["gpus"] = [4, 5, 6, 7]
        doc["hosts"][0]["deployments"][1]["gpus"] = [0]
        cluster = cfg.parse_cluster(doc, cfg.parse_models(MODELS))
        host, dep, model = next(
            t for t in cluster.instances() if t[1].model == "primary")
        changed = spec_mod.build(host, dep, model, "sk-test").spec_hash
        self.assertNotEqual(spec_for("primary").spec_hash, changed)


class AddressingTest(unittest.TestCase):
    def test_urls_point_at_the_host_not_localhost(self):
        instance = spec_for("primary")
        self.assertEqual(instance.api_base, "http://10.0.0.11:8001/v1")
        self.assertEqual(instance.health_url, "http://10.0.0.11:8001/health")

    def test_docker_host_uses_ssh(self):
        self.assertEqual(spec_for("primary").docker_host, "ssh://root@10.0.0.11")

    def test_container_name_includes_port(self):
        self.assertEqual(spec_for("primary").container, "aqueduct-primary-8001")


if __name__ == "__main__":
    unittest.main()
