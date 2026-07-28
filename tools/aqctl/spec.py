"""Turning a declaration into a concrete container specification.

An InstanceSpec is everything needed to run one vLLM container, plus a hash of
that same information. The hash is what `plan` compares against the label on a
running container to decide whether it needs recreating — comparing hashes is
reliable in a way that parsing a running container's command line is not.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import List, Tuple

LABEL_MANAGED = "aqueduct.managed"
LABEL_MODEL = "aqueduct.model"
LABEL_GPUS = "aqueduct.gpus"
LABEL_HASH = "aqueduct.spec-hash"
LABEL_SERVED = "aqueduct.served-name"

MANAGED_FILTER = "%s=true" % LABEL_MANAGED


@dataclass(frozen=True)
class InstanceSpec:
    host_name: str
    host_address: str
    docker_host: str
    container: str
    model_key: str
    served_name: str
    gpus: Tuple[int, ...]
    port: int
    health_timeout: int
    image: str
    run_args: Tuple[str, ...]
    spec_hash: str

    @property
    def health_url(self):
        return "http://%s:%d/health" % (self.host_address, self.port)

    @property
    def api_base(self):
        return "http://%s:%d/v1" % (self.host_address, self.port)


def docker_host(host):
    return "ssh://%s@%s" % (host.ssh_user, host.address)


def container_name(model_key, port):
    # Port makes it unique on a host even when the same model runs twice.
    return "aqueduct-%s-%d" % (model_key, port)


def vllm_args(model, gpus, api_key):
    """The arguments passed to the vLLM server, after the image name.

    The tuning defaults mirror inference/vllm/models/*.env — see docs/hardware.md
    for why they differ from what an H100 deployment would use.
    """
    args = [
        "--model", model.path,
        "--served-model-name", model.served_name,
        "--tensor-parallel-size", str(len(gpus)),
        "--gpu-memory-utilization", str(model.gpu_memory_utilization),
        "--host", "0.0.0.0",
        "--port", "8000",
        "--api-key", api_key,
        "--disable-log-requests",
    ]

    if model.task == "embedding":
        # Pooling models take no context-length or scheduling flags.
        args += ["--runner", "pooling", "--convert", "embed"]
    else:
        args += ["--max-model-len", str(model.max_model_len)]
        args += ["--kv-cache-dtype", model.kv_cache_dtype]
        args += ["--enable-prefix-caching", "--enable-chunked-prefill"]
        if model.max_num_batched_tokens is not None:
            args += ["--max-num-batched-tokens", str(model.max_num_batched_tokens)]
        if model.max_num_seqs is not None:
            args += ["--max-num-seqs", str(model.max_num_seqs)]

    args += list(model.extra_args)
    return args


def _hash_payload(host, model, dep, image, server_args):
    """What the spec hash covers: anything whose change requires a restart.

    Deliberately excludes the API key — rotating it should not force every
    instance in the cluster to be recreated.
    """
    return {
        "image": image,
        "gpus": list(dep.gpus),
        "port": dep.port,
        "shm_size": host.shm_size,
        "bind": host.bind,
        "model_dir": host.model_dir,
        "args": [a for a in server_args if a != "--api-key"],
    }


def build(host, dep, model, api_key):
    server_args = vllm_args(model, dep.gpus, api_key)

    # The API key is in server_args but excluded from the hash payload; strip
    # its value too so a rotation does not show up as a spec change.
    hashed_args = list(server_args)
    if "--api-key" in hashed_args:
        idx = hashed_args.index("--api-key")
        del hashed_args[idx : idx + 2]

    payload = _hash_payload(host, model, dep, host.image, hashed_args)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]

    name = container_name(dep.model, dep.port)
    gpu_list = ",".join(str(g) for g in dep.gpus)
    mount = "%s:%s:ro" % (host.model_dir, host.model_dir)

    run_args = [
        "run", "-d",
        "--name", name,
        "--restart", "unless-stopped",
        # Pin this container to exactly the declared cards.
        "--gpus", "device=%s" % gpu_list,
        "--ipc", "host",
        # Tensor-parallel workers exchange tensors through shared memory; the
        # docker default of 64MB makes startup hang with no error.
        "--shm-size", host.shm_size,
        "--ulimit", "memlock=-1",
        "--ulimit", "stack=67108864",
        "-p", "%s:%d:8000" % (host.bind, dep.port),
        "-v", mount,
        "-e", "VLLM_USE_V1=1",
        # Single host: all communication is over NVLink, InfiniBand is not used.
        "-e", "NCCL_IB_DISABLE=1",
        "-e", "NCCL_P2P_DISABLE=0",
        "-l", "%s=true" % LABEL_MANAGED,
        "-l", "%s=%s" % (LABEL_MODEL, dep.model),
        "-l", "%s=%s" % (LABEL_GPUS, gpu_list),
        "-l", "%s=%s" % (LABEL_SERVED, model.served_name),
        "-l", "%s=%s" % (LABEL_HASH, digest),
        host.image,
    ] + server_args

    return InstanceSpec(
        host_name=host.name,
        host_address=host.address,
        docker_host=docker_host(host),
        container=name,
        model_key=dep.model,
        served_name=model.served_name,
        gpus=tuple(dep.gpus),
        port=dep.port,
        health_timeout=model.health_timeout,
        image=host.image,
        run_args=tuple(run_args),
        spec_hash=digest,
    )


def build_all(cluster, api_key):
    """Every instance the declaration asks for, in declaration order."""
    specs = []
    for host, dep, model in cluster.instances():
        specs.append(build(host, dep, model, api_key))
    return specs
