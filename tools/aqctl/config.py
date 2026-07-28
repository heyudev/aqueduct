"""Loading and validation of the cluster declaration.

Two files describe the whole deployment:

  models.yaml   what a model needs in order to run
  cluster.yaml  which host runs which models on which GPUs

Everything here is pure: it reads YAML and returns objects, or a list of
errors. Nothing touches the network, so it is cheap to test and safe to run
before any machine exists.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

# Tensor-parallel size has to divide the model's attention head count. In
# practice powers of two are the only safe choices; TP=6 and friends fail to
# start with an error that does not point at the cause.
VALID_TP_SIZES = (1, 2, 4, 8)

# Weights are not the only thing in GPU memory. If a model's weights leave less
# than this much room, the instance starts but has no usable concurrency —
# better to reject the config than to debug it under load.
MIN_KV_CACHE_GB = 40.0

DEFAULT_GPU_MEMORY_GB = 96.0
DEFAULT_MEMORY_UTIL = 0.92
DEFAULT_HEALTH_TIMEOUT = 900
DEFAULT_IMAGE = "vllm/vllm-openai:latest"
DEFAULT_SSH_USER = "root"
DEFAULT_MODEL_DIR = "/data/models"
DEFAULT_BIND = "0.0.0.0"


class ConfigError(Exception):
    """Raised when the declaration cannot be parsed at all."""


@dataclass(frozen=True)
class Model:
    """One entry from models.yaml — how to serve a given model."""

    key: str
    path: str
    served_name: str
    gpus: int
    weight_gb: float
    max_model_len: Optional[int] = None
    kv_cache_dtype: str = "auto"
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: Optional[int] = None
    gpu_memory_utilization: float = DEFAULT_MEMORY_UTIL
    task: Optional[str] = None
    extra_args: tuple = ()
    health_timeout: int = DEFAULT_HEALTH_TIMEOUT


@dataclass(frozen=True)
class Deployment:
    """One vLLM instance on one host."""

    model: str
    gpus: tuple
    port: int


@dataclass(frozen=True)
class Host:
    name: str
    address: str
    gpu_count: int
    ssh_user: str
    model_dir: str
    image: str
    bind: str
    shm_size: str
    gpu_memory_gb: float
    deployments: tuple = ()


@dataclass
class Cluster:
    models: Dict[str, Model] = field(default_factory=dict)
    hosts: List[Host] = field(default_factory=list)

    def instances(self):
        """Every (host, deployment, model) triple in the cluster."""
        for host in self.hosts:
            for dep in host.deployments:
                yield host, dep, self.models[dep.model]


def _require(mapping, key, ctx):
    if key not in mapping or mapping[key] is None:
        raise ConfigError("%s: missing required field '%s'" % (ctx, key))
    return mapping[key]


def _load_yaml(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError:
        raise ConfigError("file not found: %s" % path)
    except yaml.YAMLError as exc:
        raise ConfigError("%s: invalid YAML: %s" % (path, exc))
    if not isinstance(data, dict):
        raise ConfigError("%s: expected a mapping at the top level" % path)
    return data


def parse_models(data, source="models.yaml"):
    raw = data.get("models")
    if not isinstance(raw, dict) or not raw:
        raise ConfigError("%s: 'models' must be a non-empty mapping" % source)

    models = {}
    for key, spec in raw.items():
        if not isinstance(spec, dict):
            raise ConfigError("%s: model '%s' must be a mapping" % (source, key))
        ctx = "%s: model '%s'" % (source, key)
        extra = spec.get("extra_args") or []
        if isinstance(extra, str):
            raise ConfigError("%s: extra_args must be a list, not a string" % ctx)
        models[key] = Model(
            key=key,
            path=str(_require(spec, "path", ctx)),
            served_name=str(_require(spec, "served_name", ctx)),
            gpus=int(_require(spec, "gpus", ctx)),
            weight_gb=float(_require(spec, "weight_gb", ctx)),
            max_model_len=spec.get("max_model_len"),
            kv_cache_dtype=spec.get("kv_cache_dtype", "auto"),
            max_num_batched_tokens=spec.get("max_num_batched_tokens"),
            max_num_seqs=spec.get("max_num_seqs"),
            gpu_memory_utilization=float(
                spec.get("gpu_memory_utilization", DEFAULT_MEMORY_UTIL)
            ),
            task=spec.get("task"),
            extra_args=tuple(str(a) for a in extra),
            health_timeout=int(spec.get("health_timeout", DEFAULT_HEALTH_TIMEOUT)),
        )
    return models


def parse_cluster(data, models, source="cluster.yaml"):
    defaults = data.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ConfigError("%s: 'defaults' must be a mapping" % source)

    raw_hosts = data.get("hosts")
    if not isinstance(raw_hosts, list) or not raw_hosts:
        raise ConfigError("%s: 'hosts' must be a non-empty list" % source)

    hosts = []
    for entry in raw_hosts:
        if not isinstance(entry, dict):
            raise ConfigError("%s: each host must be a mapping" % source)
        name = str(_require(entry, "name", source))
        ctx = "%s: host '%s'" % (source, name)

        deployments = []
        for dep in entry.get("deployments") or []:
            if not isinstance(dep, dict):
                raise ConfigError("%s: each deployment must be a mapping" % ctx)
            gpus = _require(dep, "gpus", ctx)
            if not isinstance(gpus, list) or not gpus:
                raise ConfigError("%s: 'gpus' must be a non-empty list" % ctx)
            deployments.append(
                Deployment(
                    model=str(_require(dep, "model", ctx)),
                    gpus=tuple(int(g) for g in gpus),
                    port=int(_require(dep, "port", ctx)),
                )
            )

        hosts.append(
            Host(
                name=name,
                address=str(_require(entry, "address", ctx)),
                gpu_count=int(_require(entry, "gpu_count", ctx)),
                ssh_user=str(entry.get("ssh_user", defaults.get("ssh_user", DEFAULT_SSH_USER))),
                model_dir=str(entry.get("model_dir", defaults.get("model_dir", DEFAULT_MODEL_DIR))),
                image=str(entry.get("image", defaults.get("image", DEFAULT_IMAGE))),
                bind=str(entry.get("bind", defaults.get("bind", DEFAULT_BIND))),
                shm_size=str(entry.get("shm_size", defaults.get("shm_size", "64gb"))),
                gpu_memory_gb=float(
                    entry.get(
                        "gpu_memory_gb",
                        defaults.get("gpu_memory_gb", DEFAULT_GPU_MEMORY_GB),
                    )
                ),
                deployments=tuple(deployments),
            )
        )

    return Cluster(models=models, hosts=hosts)


def load(cluster_path, models_path):
    models = parse_models(_load_yaml(models_path), models_path)
    return parse_cluster(_load_yaml(cluster_path), models, cluster_path)


def validate(cluster):
    """Return a list of human-readable problems. Empty list means valid.

    These checks exist because each of them, if left to deployment time, shows
    up as either a container that will not start or an OOM under load — both
    far more expensive to diagnose than a rejected config.
    """
    errors = []

    for model in cluster.models.values():
        if model.gpus not in VALID_TP_SIZES:
            errors.append(
                "model '%s': gpus=%d is not one of %s; tensor-parallel size must "
                "divide the attention head count"
                % (model.key, model.gpus, list(VALID_TP_SIZES))
            )
        if model.weight_gb <= 0:
            errors.append("model '%s': weight_gb must be positive" % model.key)
        if not 0 < model.gpu_memory_utilization <= 1:
            errors.append(
                "model '%s': gpu_memory_utilization must be in (0, 1]" % model.key
            )
        if model.task is None and model.max_model_len is None:
            errors.append(
                "model '%s': max_model_len is required for generative models"
                % model.key
            )

    # Same served_name across hosts means the gateway load-balances between
    # them. Divergent parameters would make identical requests behave
    # differently depending on which replica answered.
    by_served = {}
    for model in cluster.models.values():
        by_served.setdefault(model.served_name, []).append(model)
    for served, group in by_served.items():
        if len(group) > 1:
            keys = sorted(m.key for m in group)
            errors.append(
                "served_name '%s' is claimed by multiple models (%s); replicas "
                "must reference one model definition" % (served, ", ".join(keys))
            )

    seen_hosts = set()
    for host in cluster.hosts:
        if host.name in seen_hosts:
            errors.append("duplicate host name '%s'" % host.name)
        seen_hosts.add(host.name)

        used_gpus = {}
        used_ports = {}
        for dep in host.deployments:
            ctx = "host '%s' deployment '%s'" % (host.name, dep.model)

            model = cluster.models.get(dep.model)
            if model is None:
                errors.append("%s: no such model in models.yaml" % ctx)
                continue

            if len(dep.gpus) != model.gpus:
                errors.append(
                    "%s: assigned %d GPUs but the model requires %d"
                    % (ctx, len(dep.gpus), model.gpus)
                )

            if len(set(dep.gpus)) != len(dep.gpus):
                errors.append("%s: repeats a GPU id in its own assignment" % ctx)

            for gpu in dep.gpus:
                if not 0 <= gpu < host.gpu_count:
                    errors.append(
                        "%s: GPU %d is outside the host's range 0-%d"
                        % (ctx, gpu, host.gpu_count - 1)
                    )
                elif gpu in used_gpus:
                    errors.append(
                        "%s: GPU %d is already assigned to '%s'; a GPU belongs "
                        "to exactly one instance" % (ctx, gpu, used_gpus[gpu])
                    )
                else:
                    used_gpus[gpu] = dep.model

            if dep.port in used_ports:
                errors.append(
                    "%s: port %d is already used by '%s' on this host"
                    % (ctx, dep.port, used_ports[dep.port])
                )
            else:
                used_ports[dep.port] = dep.model

            usable = len(dep.gpus) * host.gpu_memory_gb * model.gpu_memory_utilization
            headroom = usable - model.weight_gb
            if headroom < 0:
                errors.append(
                    "%s: weights need %.0fGB but only %.0fGB is addressable on "
                    "%d GPU(s)" % (ctx, model.weight_gb, usable, len(dep.gpus))
                )
            elif headroom < MIN_KV_CACHE_GB:
                errors.append(
                    "%s: only %.0fGB left for KV cache after %.0fGB of weights "
                    "(want at least %.0fGB); this starts but has no usable "
                    "concurrency" % (ctx, headroom, model.weight_gb, MIN_KV_CACHE_GB)
                )

            if not model.path.startswith(host.model_dir.rstrip("/") + "/"):
                errors.append(
                    "%s: model path '%s' is outside the host's model_dir '%s', "
                    "so it will not be visible inside the container"
                    % (ctx, model.path, host.model_dir)
                )

    return errors
