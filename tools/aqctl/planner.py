"""Diffing desired state against what is actually running.

Pure logic: takes desired InstanceSpecs and observed containers, returns the
actions needed to reconcile. No IO, so the interesting cases are all testable
without a GPU machine.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

CREATE = "create"
REMOVE = "remove"
RECREATE = "recreate"
RESTART = "restart"
UNCHANGED = "unchanged"

# Actions that interrupt an instance that was serving traffic. `plan` calls
# these out because recreating the primary model means tens of minutes of
# downtime while several hundred GB of weights reload.
DISRUPTIVE = (REMOVE, RECREATE, RESTART)


@dataclass(frozen=True)
class RunningContainer:
    """A container observed on a host, as reported by `docker ps`."""

    host_name: str
    name: str
    model_key: str
    gpus: str
    spec_hash: str
    served_name: str
    state: str  # "running", "exited", "created", ...

    @property
    def is_running(self):
        return self.state == "running"


@dataclass(frozen=True)
class Action:
    kind: str
    host_name: str
    container: str
    spec: Optional[object] = None       # InstanceSpec, absent for REMOVE
    existing: Optional[RunningContainer] = None
    reason: str = ""

    @property
    def disruptive(self):
        return self.kind in DISRUPTIVE


@dataclass
class Plan:
    actions: Tuple[Action, ...] = ()

    def of_kind(self, kind):
        return [a for a in self.actions if a.kind == kind]

    @property
    def changes(self):
        return [a for a in self.actions if a.kind != UNCHANGED]

    @property
    def disruptions(self):
        return [a for a in self.actions if a.disruptive]

    @property
    def empty(self):
        return not self.changes


def diff(specs, running):
    """Compute the plan.

    specs   -- desired InstanceSpec list
    running -- observed RunningContainer list

    Matching is by (host, container name). The container name encodes model and
    port, so a model moved to a different port is seen as one removal plus one
    creation, which is what actually has to happen.
    """
    desired = {(s.host_name, s.container): s for s in specs}
    actual = {(c.host_name, c.name): c for c in running}

    actions = []

    for key in sorted(desired):
        spec = desired[key]
        current = actual.get(key)

        if current is None:
            actions.append(
                Action(CREATE, spec.host_name, spec.container, spec=spec,
                       reason="not deployed")
            )
        elif current.spec_hash != spec.spec_hash:
            actions.append(
                Action(RECREATE, spec.host_name, spec.container, spec=spec,
                       existing=current,
                       reason="spec changed (%s -> %s)"
                              % (current.spec_hash or "unknown", spec.spec_hash))
            )
        elif not current.is_running:
            actions.append(
                Action(RESTART, spec.host_name, spec.container, spec=spec,
                       existing=current, reason="container is %s" % current.state)
            )
        else:
            actions.append(
                Action(UNCHANGED, spec.host_name, spec.container, spec=spec,
                       existing=current)
            )

    for key in sorted(actual):
        if key not in desired:
            current = actual[key]
            actions.append(
                Action(REMOVE, current.host_name, current.name, existing=current,
                       reason="not in declaration")
            )

    return Plan(actions=tuple(actions))


def execution_order(plan):
    """Actions in the order they must run.

    Removals first: they free the GPUs that creations need. Within a host the
    order matters; across hosts it does not.
    """
    removals = [a for a in plan.actions if a.kind in (REMOVE, RECREATE)]
    creations = [a for a in plan.actions if a.kind in (CREATE, RECREATE, RESTART)]
    return removals, creations


def format_plan(plan):
    """Render a plan for a human about to approve it."""
    if plan.empty:
        return "Nothing to do — the cluster matches the declaration."

    symbols = {CREATE: "+", REMOVE: "-", RECREATE: "~", RESTART: ">"}
    lines = []
    by_host = {}
    for action in plan.changes:
        by_host.setdefault(action.host_name, []).append(action)

    for host_name in sorted(by_host):
        lines.append("%s:" % host_name)
        for action in by_host[host_name]:
            gpus = ""
            if action.spec is not None:
                gpus = " gpus=%s" % ",".join(str(g) for g in action.spec.gpus)
            elif action.existing is not None:
                gpus = " gpus=%s" % action.existing.gpus
            lines.append(
                "  %s %-28s %-9s%s  (%s)"
                % (symbols.get(action.kind, "?"), action.container, action.kind,
                   gpus, action.reason)
            )
        lines.append("")

    counts = {}
    for action in plan.changes:
        counts[action.kind] = counts.get(action.kind, 0) + 1
    summary = ", ".join("%d to %s" % (n, k) for k, n in sorted(counts.items()))
    lines.append("Plan: %s." % summary)

    disruptions = plan.disruptions
    if disruptions:
        lines.append("")
        lines.append(
            "%d instance(s) will stop serving while this runs:" % len(disruptions)
        )
        for action in disruptions:
            lines.append("  %s on %s" % (action.container, action.host_name))
        lines.append(
            "Reloading weights takes minutes to tens of minutes. Apply one host "
            "at a time if other replicas must keep serving."
        )

    return "\n".join(lines)
