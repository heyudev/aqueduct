"""Shaping collected data into the JSON the console renders.

Every function here is pure: inputs in, dict out. The collector does the IO and
hands the results over. Keeping the shaping separate means the interesting
logic — deciding an instance is unhealthy, working out which GPUs are free,
aggregating replicas into a logical model — is testable without a machine.
"""

from spec import container_name

# Instance states, in the order a reader cares about them.
RUNNING = "running"
STALE = "stale_config"      # running, but not what the declaration asks for
STOPPED = "stopped"
MISSING = "not_deployed"
UNREACHABLE = "unknown"     # host could not be queried; state is genuinely unknown
UNDECLARED = "undeclared"   # occupying GPUs but absent from cluster.yaml


def parse_gpu_label(value):
    """GPU ids off a container's aqueduct.gpus label. Tolerates junk.

    A container we did not create, or one from an older version, may carry no
    usable label — in that case we know it exists but not what it holds.
    """
    ids = []
    for part in (value or "").split(","):
        part = part.strip()
        if part.isdigit():
            ids.append(int(part))
    return ids

# Mirrors the parameter matrix in docs/design/lld.md §1.3. Kept here so the
# portal shows callers the same contract the design document states; update
# both together.
PARAMETER_SUPPORT = [
    {"name": "messages", "support": "yes",
     "note": "system / user / assistant / tool roles"},
    {"name": "stream", "support": "yes", "note": "SSE"},
    {"name": "temperature, top_p, max_tokens", "support": "yes", "note": ""},
    {"name": "stop", "support": "yes", "note": ""},
    {"name": "tools, tool_choice", "support": "yes",
     "note": "depends on the underlying model"},
    {"name": "response_format", "support": "partial",
     "note": "JSON mode via vLLM structured output; verify behaviour"},
    {"name": "seed", "support": "partial", "note": "not reproducible across versions"},
    {"name": "n", "support": "no", "note": "multiplies compute cost; disabled"},
    {"name": "logprobs", "support": "unconfirmed", "note": "not promised in phase 1"},
    {"name": "frequency_penalty, presence_penalty", "support": "yes", "note": ""},
]


def instance_state(spec, container, healthy):
    """Classify one instance. `container` is None when nothing is deployed."""
    if container is None:
        return MISSING
    if not container.is_running:
        return STOPPED
    if container.spec_hash != spec.spec_hash:
        return STALE
    return RUNNING


def cluster_view(cluster, specs, containers, health, metrics, series=None,
                 unreachable=(), generated_at=0):
    """The cluster tab: hosts, GPU occupancy, instance health, live load.

    The short metric window rides along with each instance rather than living
    on a tab of its own. A separate metrics page would be a worse Grafana;
    what Grafana cannot do is put the load curve on the same row as the GPU
    assignment it belongs to, which is what you want while diagnosing.

    containers  -- {(host_name, container_name): RunningContainer}
    health      -- {uid: bool}
    metrics     -- {uid: {...}}
    series      -- {uid: [{"t": ..., "values": {...}}, ...]}
    unreachable -- {host_name: error string}

    Per-instance data is keyed by spec.uid, never by container name — see the
    note on InstanceSpec.uid.
    """
    series = series or {}
    spec_by_key = {(s.host_name, s.container): s for s in specs}
    hosts = []
    totals = {"instances": 0, "healthy": 0, "gpus_used": 0, "gpus_total": 0,
              "hosts_unreachable": 0}

    for host in cluster.hosts:
        error = unreachable.get(host.name) if unreachable else None
        reachable = error is None
        if not reachable:
            totals["hosts_unreachable"] += 1

        gpu_owner = {}
        instances = []

        for dep in host.deployments:
            name = container_name(dep.model, dep.port)
            spec = spec_by_key.get((host.name, name))
            container = containers.get((host.name, name))
            model = cluster.models[dep.model]

            uid = spec.uid if spec else "%s/%s" % (host.name, name)

            if not reachable:
                state = UNREACHABLE
            else:
                state = instance_state(spec, container, health.get(uid))

            is_healthy = bool(health.get(uid)) and state == RUNNING
            usable = len(dep.gpus) * host.gpu_memory_gb * model.gpu_memory_utilization

            instances.append({
                "uid": uid,
                "container": name,
                "model": dep.model,
                "served_name": model.served_name,
                "gpus": list(dep.gpus),
                "port": dep.port,
                "state": state,
                "healthy": is_healthy,
                "endpoint": spec.api_base if spec else None,
                "weight_gb": model.weight_gb,
                "kv_headroom_gb": round(usable - model.weight_gb),
                "max_model_len": model.max_model_len,
                "metrics": metrics.get(uid) or {},
                "window": series.get(uid, []),
            })

            for gpu in dep.gpus:
                gpu_owner[gpu] = {"model": dep.model, "state": state}

            totals["instances"] += 1
            if is_healthy:
                totals["healthy"] += 1

        declared = {container_name(d.model, d.port) for d in host.deployments}
        undeclared = []
        for (h, n), container in sorted(containers.items()):
            if h != host.name or n in declared:
                continue
            gpu_ids = parse_gpu_label(container.gpus)
            undeclared.append({
                "container": container.name,
                "state": container.state,
                "gpus": gpu_ids,
            })
            # A container outside the declaration still holds its cards. Showing
            # those as free would advertise capacity that is not there — and the
            # next apply would fail trying to claim them.
            if container.is_running:
                for gpu in gpu_ids:
                    if 0 <= gpu < host.gpu_count and gpu not in gpu_owner:
                        gpu_owner[gpu] = {"model": container.name,
                                          "state": UNDECLARED}

        gpus = []
        for gpu_id in range(host.gpu_count):
            owner = gpu_owner.get(gpu_id)
            gpus.append({
                "id": gpu_id,
                "model": owner["model"] if owner else None,
                "state": owner["state"] if owner else "free",
            })

        totals["gpus_used"] += len(gpu_owner)
        totals["gpus_total"] += host.gpu_count

        hosts.append({
            "name": host.name,
            "address": host.address,
            "gpu_count": host.gpu_count,
            "reachable": reachable,
            "error": error,
            "gpus": gpus,
            "instances": instances,
            "undeclared": undeclared,
        })

    return {"generated_at": generated_at, "hosts": hosts, "summary": totals}


def models_view(cluster, specs, health, gateway_url, generated_at=0):
    """The portal tab: what a caller can use, and whether it is up.

    Replicas of one logical model are collapsed into a single row — callers
    address the logical name and should not have to think about machines.
    """
    by_served = {}
    for spec in specs:
        entry = by_served.setdefault(spec.served_name, {
            "served_name": spec.served_name,
            "replicas": 0,
            "healthy_replicas": 0,
            "hosts": [],
        })
        entry["replicas"] += 1
        if health.get(spec.uid):
            entry["healthy_replicas"] += 1
        entry["hosts"].append(spec.host_name)

    models = []
    for model in cluster.models.values():
        agg = by_served.get(model.served_name)
        if agg is None:
            # Defined in models.yaml but not assigned to any host yet.
            models.append({
                "served_name": model.served_name,
                "kind": "embedding" if model.task == "embedding" else "chat",
                "max_input_tokens": model.max_model_len,
                "replicas": 0,
                "healthy_replicas": 0,
                "hosts": [],
                "status": "not_deployed",
            })
            continue

        if agg["healthy_replicas"] == 0:
            status = "down"
        elif agg["healthy_replicas"] < agg["replicas"]:
            status = "degraded"
        else:
            status = "available"

        models.append({
            "served_name": model.served_name,
            "kind": "embedding" if model.task == "embedding" else "chat",
            "max_input_tokens": model.max_model_len,
            "replicas": agg["replicas"],
            "healthy_replicas": agg["healthy_replicas"],
            "hosts": sorted(set(agg["hosts"])),
            "status": status,
        })

    models.sort(key=lambda m: m["served_name"])
    return {
        "generated_at": generated_at,
        "gateway_url": gateway_url,
        "models": models,
        "parameters": PARAMETER_SUPPORT,
    }


# A key using this much of its budget is worth surfacing before it hits the
# ceiling and starts getting 429s.
BUDGET_WARN_PERCENT = 80.0


def usage_view(spend_rows, generated_at=0, error=None, admin_url=None):
    """The usage tab: a read-only summary, not a management screen.

    LiteLLM's own admin UI mints and revokes keys and shows the full ledger; it
    owns this data and does more with it. What it does not do is sit next to
    the cluster view, so what earns its place here is spotting the thing an
    operator checks daily — a key whose consumption has run away — and then
    handing off.
    """
    keys = []
    total_spend = 0.0
    for row in spend_rows or []:
        spend = float(row.get("spend") or 0)
        budget = row.get("max_budget")
        total_spend += spend
        pct = None
        if budget:
            try:
                pct = round(spend / float(budget) * 100, 1)
            except (TypeError, ZeroDivisionError):
                pct = None
        keys.append({
            "alias": row.get("key_alias") or row.get("key_name") or "(unnamed)",
            "spend": round(spend, 4),
            "max_budget": budget,
            "percent_used": pct,
            "models": row.get("models") or [],
            "budget_duration": row.get("budget_duration"),
            "attention": pct is not None and pct >= BUDGET_WARN_PERCENT,
        })

    keys.sort(key=lambda k: k["spend"], reverse=True)
    return {
        "generated_at": generated_at,
        "keys": keys,
        "total_spend": round(total_spend, 4),
        "needs_attention": [k for k in keys if k["attention"]],
        "admin_url": admin_url,
        "error": error,
    }
