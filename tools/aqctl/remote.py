"""Talking to the GPU hosts.

Everything goes through `docker -H ssh://user@host`, so the hosts need nothing
installed beyond docker and an SSH key — there is no agent of ours to operate.

This module is the only place that does IO. It is kept thin on purpose: the
decisions live in planner.py, which is testable without a machine.
"""

import json
import subprocess
import time
import urllib.error
import urllib.request

from planner import RunningContainer
from spec import (
    LABEL_GPUS,
    LABEL_HASH,
    LABEL_MODEL,
    LABEL_SERVED,
    MANAGED_FILTER,
)

DEFAULT_TIMEOUT = 60


class RemoteError(Exception):
    pass


def _docker(docker_host, args, timeout=DEFAULT_TIMEOUT, check=True):
    cmd = ["docker", "-H", docker_host] + list(args)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except FileNotFoundError:
        raise RemoteError("docker CLI not found on this machine")
    except subprocess.TimeoutExpired:
        raise RemoteError(
            "timed out after %ds: docker -H %s %s"
            % (timeout, docker_host, " ".join(args[:2]))
        )
    if check and proc.returncode != 0:
        raise RemoteError(
            "docker -H %s %s failed: %s"
            % (docker_host, " ".join(args[:2]), (proc.stderr or proc.stdout).strip())
        )
    return proc


# Tab-separated so parsing stays trivial; container names and hashes never
# contain tabs.
_PS_FORMAT = "\t".join(
    [
        "{{.Names}}",
        "{{.State}}",
        '{{.Label "%s"}}' % LABEL_MODEL,
        '{{.Label "%s"}}' % LABEL_GPUS,
        '{{.Label "%s"}}' % LABEL_SERVED,
        '{{.Label "%s"}}' % LABEL_HASH,
    ]
)


def parse_ps(host_name, stdout):
    """Parse `docker ps` output into RunningContainer objects.

    Split out from the subprocess call so it can be tested directly.
    """
    containers = []
    for line in stdout.splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 6:
            continue
        name, state, model_key, gpus, served, spec_hash = fields[:6]
        containers.append(
            RunningContainer(
                host_name=host_name,
                name=name,
                model_key=model_key,
                gpus=gpus,
                served_name=served,
                spec_hash=spec_hash,
                state=state,
            )
        )
    return containers


def list_containers(host, timeout=30):
    """Containers on a host that we manage. Includes stopped ones.

    The timeout is short: this runs against every host before plan/status can
    print anything, so an unreachable machine should be reported quickly rather
    than stalling the whole command.
    """
    from spec import docker_host as _dh

    proc = _docker(
        _dh(host),
        ["ps", "-a", "--filter", "label=" + MANAGED_FILTER, "--format", _PS_FORMAT],
        timeout=timeout,
    )
    return parse_ps(host.name, proc.stdout)


def create(spec):
    _docker(spec.docker_host, list(spec.run_args), timeout=180)


def remove(docker_host, container):
    # Stop then remove. A vLLM container holding several hundred GB of weights
    # can take a while to shut down; -t gives it room before SIGKILL.
    _docker(docker_host, ["stop", "-t", "30", container], timeout=120, check=False)
    _docker(docker_host, ["rm", "-f", container], timeout=60, check=False)


def start(docker_host, container):
    _docker(docker_host, ["start", container], timeout=60)


def logs(docker_host, container, tail=200, follow=False):
    """Stream logs to this terminal. Blocks when follow=True."""
    cmd = ["docker", "-H", docker_host, "logs", "--tail", str(tail)]
    if follow:
        cmd.append("-f")
    cmd.append(container)
    subprocess.call(cmd)


def check_health(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def wait_healthy(specs, on_event=None, poll_interval=15, clock=time.time,
                 sleep=time.sleep):
    """Poll /health until every instance answers or its own timeout expires.

    Each model carries its own timeout because load times differ by an order of
    magnitude — a 235B model can take half an hour where a small one takes two
    minutes. A single global timeout would either be uselessly long or would
    report healthy instances as failed.

    Returns (ready, failed) lists of specs.
    """
    pending = {s.container: s for s in specs}
    started = clock()
    ready = []
    failed = []

    while pending:
        for container in sorted(pending):
            spec = pending[container]
            if check_health(spec.health_url):
                ready.append(spec)
                del pending[container]
                if on_event:
                    on_event("ready", spec, clock() - started)
            elif clock() - started > spec.health_timeout:
                failed.append(spec)
                del pending[container]
                if on_event:
                    on_event("timeout", spec, clock() - started)
        if pending:
            if on_event:
                on_event("waiting", None, clock() - started)
            sleep(poll_interval)

    return ready, failed


def probe_metrics(spec, timeout=5):
    """Fetch a few vLLM gauges for `status`. Returns {} when unavailable."""
    url = "http://%s:%d/metrics" % (spec.host_address, spec.port)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError):
        return {}

    wanted = {
        "vllm:num_requests_running": "running",
        "vllm:num_requests_waiting": "waiting",
        "vllm:gpu_cache_usage_perc": "kv_cache",
    }
    found = {}
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        for metric, label in wanted.items():
            if line.startswith(metric):
                try:
                    found[label] = float(line.rsplit(" ", 1)[1])
                except (ValueError, IndexError):
                    pass
    return found
