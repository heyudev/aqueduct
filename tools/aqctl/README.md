# aqctl

Reconciles the GPU cluster with `cluster.yaml` + `models.yaml`. Design and
rationale: [../../docs/design/orchestration.md](../../docs/design/orchestration.md).

## Usage

Normally invoked through the Makefile from the repo root:

```bash
make validate     # check the declaration; contacts no machines
make status       # what is actually running, per host and per GPU
make plan         # what apply would change
make apply        # make it so
make gen-gateway  # regenerate gateway/config.yaml
make logs HOST=gpu-01 MODEL=primary
make console      # read-only web console on :8080
```

Directly, when you need non-default paths:

```bash
python3 tools/aqctl --cluster other.yaml --models other-models.yaml plan
python3 tools/aqctl apply -y      # skip the confirmation prompt
```

## Requirements

- Python 3.8+ with PyYAML (`pip install -r tools/aqctl/requirements.txt`)
- `docker` CLI on this machine
- Passwordless SSH from this machine to every host in `cluster.yaml`
- Docker plus nvidia-container-toolkit on each GPU host (`make preflight`)

Nothing is installed on the GPU hosts — all remote work goes through
`docker -H ssh://user@host`.

## Layout

| File | Role |
|---|---|
| `config.py` | Parse and validate the declaration. Pure. |
| `spec.py` | Declaration → container spec + spec hash. Pure. |
| `planner.py` | Desired vs actual → actions. Pure. |
| `gateway.py` | Declaration → LiteLLM model list. Pure. |
| `remote.py` | The only module that does IO. Deliberately thin. |
| `views.py` | Collected data → the console's JSON. Pure. |
| `collector.py` | Background polling into an in-memory snapshot. |
| `server.py` | Read-only HTTP console. Standard library only. |
| `web/` | The console's HTML/CSS/JS. No build step, no CDN. |
| `cli.py` | Argument parsing and command bodies. |

Decisions live in the pure modules so they can be tested without a GPU machine;
`remote.py` is kept thin because it is the part that cannot be.

## Tests

```bash
make test
```

Covers validation rules, the diff algorithm, container spec construction, spec
hashing, gateway generation, and the console's view shaping. The SSH and docker
calls in `remote.py` are not covered — only its output parsing is. **Anything
involving a real host is unverified until the machines arrive.**

The console was checked against fabricated state covering every render path
(healthy, stale config, stopped, undeployed, unreachable host, undeclared
container, budget overrun) in both themes and at mobile width.
