"""Generating the gateway's model list from the cluster declaration.

The gateway config has two parts with different lifecycles: policy
(general_settings, litellm_settings, router_settings) is hand-maintained in
config.base.yaml, while the model list follows whatever is deployed and is
generated from the same declaration that deploys it.

Keeping them in one generated file is what LiteLLM expects; keeping the source
of each part separate is what stops someone hand-editing a generated section
and losing the change on the next apply.
"""

import yaml

HEADER = """\
# ============================================================================
# GENERATED FILE — DO NOT EDIT
#
# The model_list section is generated from cluster.yaml + models.yaml by
#   aqctl gen-gateway
# Everything else comes from gateway/config.base.yaml.
#
# To change routing or which models exist, edit cluster.yaml and re-run apply.
# To change gateway policy (timeouts, retries, logging), edit config.base.yaml.
# ============================================================================
"""


def build_model_list(specs):
    """One entry per instance. Same served_name across entries means LiteLLM
    load-balances between them, which is exactly how replicas should behave.
    """
    entries = []
    for spec in sorted(specs, key=lambda s: (s.served_name, s.host_name, s.port)):
        params = {
            "model": "openai/%s" % spec.served_name,
            "api_base": spec.api_base,
            "api_key": "os.environ/VLLM_API_KEY",
        }
        entry = {
            "model_name": spec.served_name,
            "litellm_params": params,
            # Not read by LiteLLM; here so someone reading the generated file
            # can tell which machine an entry points at.
            "model_info": {"aqueduct_host": spec.host_name},
        }
        entries.append(entry)
    return entries


def render(specs, base_config):
    """Return the full gateway config as YAML text."""
    config = dict(base_config or {})
    config["model_list"] = build_model_list(specs)

    # model_list first: it is what people look for when they open this file.
    ordered = {"model_list": config.pop("model_list")}
    for key in sorted(config):
        ordered[key] = config[key]

    body = yaml.safe_dump(
        ordered, sort_keys=False, allow_unicode=True, default_flow_style=False
    )
    return HEADER + "\n" + body


def load_base(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if "model_list" in data:
        raise ValueError(
            "%s must not define model_list — that section is generated" % path
        )
    return data
