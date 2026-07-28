import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

import gateway  # noqa: E402
import remote  # noqa: E402


class FakeSpec(object):
    def __init__(self, served_name, host_name, address, port):
        self.served_name = served_name
        self.host_name = host_name
        self.host_address = address
        self.port = port

    @property
    def api_base(self):
        return "http://%s:%d/v1" % (self.host_address, self.port)


BASE = {
    "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
    "litellm_settings": {"drop_params": True},
}


class RenderTest(unittest.TestCase):
    def render(self, specs):
        return yaml.safe_load(gateway.render(specs, BASE))

    def test_replicas_share_one_model_name(self):
        # Same model_name across entries is how LiteLLM load-balances; this is
        # the mechanism that makes a second machine add capacity.
        doc = self.render([
            FakeSpec("aqueduct-chat", "gpu-01", "10.0.0.11", 8001),
            FakeSpec("aqueduct-chat", "gpu-02", "10.0.0.12", 8001),
        ])
        names = [e["model_name"] for e in doc["model_list"]]
        self.assertEqual(names, ["aqueduct-chat", "aqueduct-chat"])
        bases = sorted(e["litellm_params"]["api_base"] for e in doc["model_list"])
        self.assertEqual(
            bases, ["http://10.0.0.11:8001/v1", "http://10.0.0.12:8001/v1"]
        )

    def test_distinct_models_become_distinct_entries(self):
        doc = self.render([
            FakeSpec("aqueduct-chat", "gpu-01", "10.0.0.11", 8001),
            FakeSpec("aqueduct-fast", "gpu-01", "10.0.0.11", 8002),
        ])
        self.assertEqual(
            sorted(e["model_name"] for e in doc["model_list"]),
            ["aqueduct-chat", "aqueduct-fast"],
        )

    def test_base_settings_are_preserved(self):
        doc = self.render([FakeSpec("aqueduct-chat", "gpu-01", "10.0.0.11", 8001)])
        self.assertEqual(doc["litellm_settings"]["drop_params"], True)
        self.assertIn("general_settings", doc)

    def test_api_key_is_an_env_reference_not_a_value(self):
        doc = self.render([FakeSpec("aqueduct-chat", "gpu-01", "10.0.0.11", 8001)])
        self.assertEqual(
            doc["model_list"][0]["litellm_params"]["api_key"],
            "os.environ/VLLM_API_KEY",
        )

    def test_generated_file_is_marked_as_such(self):
        text = gateway.render([FakeSpec("aqueduct-chat", "g", "10.0.0.11", 1)], BASE)
        self.assertIn("DO NOT EDIT", text)

    def test_output_is_deterministic(self):
        # Otherwise every apply produces a spurious git diff.
        specs = [
            FakeSpec("aqueduct-fast", "gpu-02", "10.0.0.12", 8002),
            FakeSpec("aqueduct-chat", "gpu-01", "10.0.0.11", 8001),
        ]
        first = gateway.render(specs, BASE)
        second = gateway.render(list(reversed(specs)), BASE)
        self.assertEqual(first, second)

    def test_empty_cluster_yields_empty_model_list(self):
        doc = self.render([])
        self.assertEqual(doc["model_list"], [])


class LoadBaseTest(unittest.TestCase):
    def test_base_containing_model_list_is_rejected(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write("model_list:\n  - model_name: x\n")
            path = handle.name
        try:
            with self.assertRaises(ValueError):
                gateway.load_base(path)
        finally:
            os.unlink(path)


class ParsePsTest(unittest.TestCase):
    """docker ps output parsing — the one piece of remote.py that is pure."""

    def test_parses_tab_separated_fields(self):
        out = "aqueduct-primary-8001\trunning\tprimary\t0,1,2,3\taqueduct-chat\tabc123\n"
        containers = remote.parse_ps("gpu-01", out)
        self.assertEqual(len(containers), 1)
        c = containers[0]
        self.assertEqual(c.name, "aqueduct-primary-8001")
        self.assertEqual(c.model_key, "primary")
        self.assertEqual(c.gpus, "0,1,2,3")
        self.assertEqual(c.spec_hash, "abc123")
        self.assertTrue(c.is_running)

    def test_stopped_container_is_not_running(self):
        out = "aqueduct-primary-8001\texited\tprimary\t0\taqueduct-chat\tabc123\n"
        self.assertFalse(remote.parse_ps("gpu-01", out)[0].is_running)

    def test_blank_and_short_lines_are_skipped(self):
        out = "\n  \nincomplete\tline\n"
        self.assertEqual(remote.parse_ps("gpu-01", out), [])

    def test_container_without_hash_label_parses_with_empty_hash(self):
        # An externally created container carrying our managed label but no
        # hash must still parse, so plan can decide to replace it.
        out = "stray\trunning\t\t\t\t\n"
        containers = remote.parse_ps("gpu-01", out)
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].spec_hash, "")


if __name__ == "__main__":
    unittest.main()
