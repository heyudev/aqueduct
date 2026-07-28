"""aqctl — reconcile the GPU cluster with its declaration.

Commands:
  validate      check the declaration; touches no machines
  status        what is actually running, per host and per GPU
  plan          what apply would change
  apply         make reality match the declaration
  gen-gateway   regenerate the gateway config from the declaration
  logs          tail one instance's logs
"""

import argparse
import concurrent.futures
import os
import sys

import config as cfg
import gateway
import planner
import remote
import spec as spec_mod

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_CLUSTER = os.path.join(REPO_ROOT, "cluster.yaml")
DEFAULT_MODELS = os.path.join(REPO_ROOT, "models.yaml")
DEFAULT_ENV = os.path.join(REPO_ROOT, ".env")
GATEWAY_BASE = os.path.join(REPO_ROOT, "gateway", "config.base.yaml")
GATEWAY_OUT = os.path.join(REPO_ROOT, "gateway", "config.yaml")


def read_env(path):
    """Minimal .env reader — enough for KEY=value, no shell semantics."""
    values = {}
    if not os.path.exists(path):
        return values
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_cluster(args):
    try:
        cluster = cfg.load(args.cluster, args.models)
    except cfg.ConfigError as exc:
        sys.exit("Config error: %s" % exc)

    errors = cfg.validate(cluster)
    if errors:
        print("Declaration is invalid:\n", file=sys.stderr)
        for err in errors:
            print("  - %s" % err, file=sys.stderr)
        sys.exit(1)
    return cluster


def api_key(args):
    env = read_env(args.env)
    key = os.environ.get("VLLM_API_KEY") or env.get("VLLM_API_KEY")
    if not key:
        sys.exit(
            "VLLM_API_KEY is not set. Put it in %s or export it." % args.env
        )
    return key


def collect_running(cluster):
    """Query every host in parallel. A host that cannot be reached is reported
    but does not abort the run — you still want to see the rest of the cluster.
    """
    running = []
    unreachable = []

    def probe(host):
        return host, remote.list_containers(host)

    workers = max(1, min(8, len(cluster.hosts)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(probe, h): h for h in cluster.hosts}
        for future in concurrent.futures.as_completed(futures):
            host = futures[future]
            try:
                _, containers = future.result()
                running.extend(containers)
            except remote.RemoteError as exc:
                unreachable.append((host, str(exc)))

    return running, unreachable


def report_unreachable(unreachable):
    if not unreachable:
        return
    print("", file=sys.stderr)
    for host, message in unreachable:
        print("WARNING: %s (%s) unreachable: %s" % (host.name, host.address, message),
              file=sys.stderr)
    print("", file=sys.stderr)


# --------------------------------------------------------------------------
# commands


def cmd_validate(args):
    cluster = load_cluster(args)
    total = sum(len(h.deployments) for h in cluster.hosts)
    print("Declaration is valid: %d host(s), %d instance(s), %d model(s) defined."
          % (len(cluster.hosts), total, len(cluster.models)))

    for host in cluster.hosts:
        used = sorted(g for d in host.deployments for g in d.gpus)
        free = [g for g in range(host.gpu_count) if g not in used]
        print("\n%s (%s)" % (host.name, host.address))
        for dep in host.deployments:
            model = cluster.models[dep.model]
            usable = len(dep.gpus) * host.gpu_memory_gb * model.gpu_memory_utilization
            print(
                "  gpus %-14s %-10s port %-6d weights %.0fGB, ~%.0fGB for KV cache"
                % (",".join(str(g) for g in dep.gpus), dep.model, dep.port,
                   model.weight_gb, usable - model.weight_gb)
            )
        if free:
            print("  gpus %-14s (free)" % ",".join(str(g) for g in free))
    return 0


def cmd_status(args):
    cluster = load_cluster(args)
    specs = spec_mod.build_all(cluster, api_key(args))
    running, unreachable = collect_running(cluster)
    report_unreachable(unreachable)

    by_key = {(c.host_name, c.name): c for c in running}
    spec_by_key = {(s.host_name, s.container): s for s in specs}

    for host in cluster.hosts:
        print("%s (%s)" % (host.name, host.address))
        occupied = {}
        for dep in host.deployments:
            for gpu in dep.gpus:
                occupied[gpu] = dep.model

        for dep in host.deployments:
            key = (host.name, spec_mod.container_name(dep.model, dep.port))
            current = by_key.get(key)
            spec = spec_by_key.get(key)

            if current is None:
                state = "NOT DEPLOYED"
            elif not current.is_running:
                state = current.state.upper()
            elif spec and current.spec_hash != spec.spec_hash:
                state = "running (STALE CONFIG)"
            else:
                state = "running"

            line = "  gpus %-14s %-10s port %-6d %s" % (
                ",".join(str(g) for g in dep.gpus), dep.model, dep.port, state)

            if current is not None and current.is_running and spec is not None:
                metrics = remote.probe_metrics(spec)
                if metrics:
                    line += "   [in-flight %d, queued %d, kv %.0f%%]" % (
                        metrics.get("running", 0),
                        metrics.get("waiting", 0),
                        metrics.get("kv_cache", 0) * 100,
                    )
                else:
                    line += "   [metrics unavailable]"
            print(line)

        free = [g for g in range(host.gpu_count) if g not in occupied]
        if free:
            print("  gpus %-14s (free)" % ",".join(str(g) for g in free))

        # Containers we manage that the declaration no longer mentions.
        declared = {spec_mod.container_name(d.model, d.port) for d in host.deployments}
        for container in running:
            if container.host_name == host.name and container.name not in declared:
                print("  %-19s %-10s UNDECLARED (%s) — apply will remove it"
                      % ("", container.name, container.state))
        print("")

    return 0


def _build_plan(args, cluster):
    specs = spec_mod.build_all(cluster, api_key(args))
    running, unreachable = collect_running(cluster)
    report_unreachable(unreachable)

    if unreachable:
        if len(unreachable) == len(cluster.hosts):
            # Every host is unreachable, so we know nothing about actual state.
            # Reporting "nothing to do" here would be a lie that reads like
            # reassurance.
            sys.exit(
                "No host could be reached — cannot determine what is running. "
                "Check SSH access and that docker is installed on the hosts."
            )
        # Diffing against a host we could not read would propose recreating
        # everything on it, which is exactly the wrong move.
        unreachable_names = {h.name for h, _ in unreachable}
        specs = [s for s in specs if s.host_name not in unreachable_names]
        print("Skipping unreachable host(s); this plan covers the rest.\n",
              file=sys.stderr)

    return specs, planner.diff(specs, running)


def cmd_plan(args):
    cluster = load_cluster(args)
    _, plan = _build_plan(args, cluster)
    print(planner.format_plan(plan))
    return 0


def cmd_apply(args):
    cluster = load_cluster(args)
    _, plan = _build_plan(args, cluster)

    print(planner.format_plan(plan))
    if plan.empty:
        return 0

    if not args.yes:
        print("")
        answer = input("Apply these changes? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            return 1

    removals, creations = planner.execution_order(plan)

    # Removals first — they release the GPUs the creations need.
    for action in removals:
        target = action.spec.docker_host if action.spec else _docker_host_for(
            cluster, action.host_name)
        print("removing %s on %s" % (action.container, action.host_name))
        remote.remove(target, action.container)

    started = []
    for action in creations:
        print("starting %s on %s" % (action.container, action.host_name))
        try:
            if action.kind == planner.RESTART:
                remote.start(action.spec.docker_host, action.container)
            else:
                remote.create(action.spec)
            started.append(action.spec)
        except remote.RemoteError as exc:
            print("  FAILED: %s" % exc, file=sys.stderr)

    if started:
        print("\nWaiting for %d instance(s) to become healthy." % len(started))
        print("Loading several hundred GB of weights takes minutes to tens of "
              "minutes; timeouts come from each model's health_timeout.\n")

        def on_event(kind, instance, elapsed):
            if kind == "ready":
                print("  ready   %s after %.0fs" % (instance.container, elapsed))
            elif kind == "timeout":
                print("  TIMEOUT %s after %.0fs — check: aqctl logs %s %s"
                      % (instance.container, elapsed, instance.host_name,
                         instance.model_key), file=sys.stderr)

        ready, failed = remote.wait_healthy(started, on_event=on_event)
        print("\n%d ready, %d failed." % (len(ready), len(failed)))
        if failed:
            print("Instances that timed out may still be loading — the container "
                  "is not stopped. Check logs before retrying.", file=sys.stderr)

    # Generate from the full declaration, not from the subset this run could
    # reach. A host that was unreachable over SSH may still be serving traffic
    # fine; dropping it from the gateway would take working capacity offline.
    # LiteLLM's own health checks remove backends that are genuinely dead.
    write_gateway_config(spec_mod.build_all(cluster, api_key(args)))
    print("\nWrote %s" % os.path.relpath(GATEWAY_OUT, REPO_ROOT))
    print("Apply it with: make restart-gateway")
    return 0


def _docker_host_for(cluster, host_name):
    for host in cluster.hosts:
        if host.name == host_name:
            return spec_mod.docker_host(host)
    raise KeyError(host_name)


def write_gateway_config(specs):
    try:
        base = gateway.load_base(GATEWAY_BASE)
    except FileNotFoundError:
        sys.exit("Missing %s — the hand-maintained half of the gateway config."
                 % GATEWAY_BASE)
    except ValueError as exc:
        sys.exit(str(exc))

    text = gateway.render(specs, base)
    with open(GATEWAY_OUT, "w", encoding="utf-8") as handle:
        handle.write(text)


def cmd_gen_gateway(args):
    cluster = load_cluster(args)
    specs = spec_mod.build_all(cluster, api_key(args))
    write_gateway_config(specs)
    served = sorted({s.served_name for s in specs})
    print("Wrote %s: %d backend(s) across %d logical model(s) (%s)"
          % (os.path.relpath(GATEWAY_OUT, REPO_ROOT), len(specs), len(served),
             ", ".join(served)))
    return 0


def cmd_logs(args):
    cluster = load_cluster(args)
    for host in cluster.hosts:
        if host.name != args.host:
            continue
        for dep in host.deployments:
            if dep.model != args.model:
                continue
            remote.logs(
                spec_mod.docker_host(host),
                spec_mod.container_name(dep.model, dep.port),
                tail=args.tail,
                follow=args.follow,
            )
            return 0
    sys.exit("No deployment of '%s' on host '%s' in the declaration."
             % (args.model, args.host))


# --------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(
        prog="aqctl", description="Reconcile the Aqueduct GPU cluster.")
    parser.add_argument("--cluster", default=DEFAULT_CLUSTER)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--env", default=DEFAULT_ENV)

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("validate", help="check the declaration, touching no machines")
    sub.add_parser("status", help="what is actually running")
    sub.add_parser("plan", help="what apply would change")

    apply_parser = sub.add_parser("apply", help="make reality match the declaration")
    apply_parser.add_argument("-y", "--yes", action="store_true",
                              help="skip the confirmation prompt")

    sub.add_parser("gen-gateway", help="regenerate the gateway config")

    logs_parser = sub.add_parser("logs", help="tail one instance's logs")
    logs_parser.add_argument("host")
    logs_parser.add_argument("model")
    logs_parser.add_argument("--tail", type=int, default=200)
    logs_parser.add_argument("-f", "--follow", action="store_true")

    return parser


COMMANDS = {
    "validate": cmd_validate,
    "status": cmd_status,
    "plan": cmd_plan,
    "apply": cmd_apply,
    "gen-gateway": cmd_gen_gateway,
    "logs": cmd_logs,
}


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2
    try:
        return COMMANDS[args.command](args)
    except remote.RemoteError as exc:
        sys.exit("Remote error: %s" % exc)
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
