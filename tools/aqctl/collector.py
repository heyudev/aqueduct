"""Background collection of cluster state for the console.

Two loops on different periods, because the sources have very different costs:

  slow  container inventory over SSH, and gateway spend — seconds per host
  fast  /health and /metrics over HTTP — milliseconds

HTTP requests to the console never touch a machine; they read the last
snapshot. That is the whole point of collecting in the background: a page load
must not block on an SSH handshake to a host that is down.
"""

import collections
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import remote

SLOW_INTERVAL = 30.0
FAST_INTERVAL = 10.0

# 180 samples at 10s is a 30-minute window. Enough to see a load spike or a
# restart; anything longer is Grafana's job.
WINDOW_SAMPLES = 180

METRIC_KEYS = ("running", "waiting", "kv_cache")


class Series(object):
    """Fixed-length time series per container. Not thread-safe on its own —
    the Collector holds a lock around it."""

    def __init__(self, maxlen=WINDOW_SAMPLES):
        self.maxlen = maxlen
        self._data = {}

    def append(self, container, values, timestamp):
        window = self._data.get(container)
        if window is None:
            window = collections.deque(maxlen=self.maxlen)
            self._data[container] = window
        window.append({"t": round(timestamp), "values": dict(values)})

    def drop_missing(self, keep):
        """Forget containers that are no longer declared, so the window does
        not grow without bound as models come and go."""
        for container in list(self._data):
            if container not in keep:
                del self._data[container]

    def as_dict(self):
        return {c: list(w) for c, w in self._data.items()}

    def __len__(self):
        return len(self._data)


def fetch_spend(gateway_url, master_key, timeout=5):
    """Per-key spend from the gateway. Returns (rows, error)."""
    url = gateway_url.rstrip("/") + "/spend/keys"
    request = urllib.request.Request(
        url, headers={"Authorization": "Bearer %s" % master_key})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return [], "gateway returned HTTP %s" % exc.code
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return [], "gateway unreachable: %s" % exc

    if isinstance(payload, dict):
        payload = payload.get("keys") or payload.get("data") or []
    if not isinstance(payload, list):
        return [], "unexpected response shape from gateway"
    return payload, None


class Collector(object):
    def __init__(self, cluster, specs, gateway_url, master_key=None,
                 slow_interval=SLOW_INTERVAL, fast_interval=FAST_INTERVAL,
                 clock=time.time):
        self.cluster = cluster
        self.specs = specs
        self.gateway_url = gateway_url
        self.master_key = master_key
        self.slow_interval = slow_interval
        self.fast_interval = fast_interval
        self.clock = clock

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads = []

        self.containers = {}
        self.unreachable = {}
        self.health = {}
        self.metrics = {}
        self.series = Series()
        self.spend_rows = []
        self.spend_error = None
        self.last_slow = 0.0
        self.last_fast = 0.0

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        for target in (self._slow_loop, self._fast_loop):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self):
        self._stop.set()

    def _loop(self, work, interval):
        while not self._stop.is_set():
            try:
                work()
            except Exception as exc:  # a collector crash must not kill the page
                print("[collector] %s failed: %s" % (work.__name__, exc))
            self._stop.wait(interval)

    def _slow_loop(self):
        self._loop(self.collect_containers, self.slow_interval)

    def _fast_loop(self):
        self._loop(self.collect_health, self.fast_interval)

    # -- collection --------------------------------------------------------

    def collect_containers(self):
        containers = {}
        unreachable = {}
        hosts = self.cluster.hosts
        if hosts:
            with ThreadPoolExecutor(max_workers=min(8, len(hosts))) as pool:
                for host, result in zip(hosts, pool.map(self._safe_probe, hosts)):
                    found, error = result
                    if error:
                        unreachable[host.name] = error
                    for container in found:
                        containers[(host.name, container.name)] = container

        with self._lock:
            self.containers = containers
            self.unreachable = unreachable
            self.last_slow = self.clock()

        if self.master_key:
            rows, error = fetch_spend(self.gateway_url, self.master_key)
            with self._lock:
                self.spend_rows = rows
                self.spend_error = error

    def _safe_probe(self, host):
        try:
            return remote.list_containers(host), None
        except remote.RemoteError as exc:
            return [], str(exc)

    def collect_health(self):
        if not self.specs:
            return

        def probe(spec):
            healthy = remote.check_health(spec.health_url, timeout=3)
            values = remote.probe_metrics(spec, timeout=3) if healthy else {}
            # Keyed by uid, not container name: replicas of one model on two
            # hosts share a container name and would overwrite each other.
            return spec.uid, healthy, values

        health = {}
        metrics = {}
        with ThreadPoolExecutor(max_workers=min(8, len(self.specs))) as pool:
            for uid, healthy, values in pool.map(probe, self.specs):
                health[uid] = healthy
                metrics[uid] = values

        now = self.clock()
        with self._lock:
            self.health = health
            self.metrics = metrics
            for uid, values in metrics.items():
                if values:
                    self.series.append(
                        uid,
                        {k: values.get(k, 0) for k in METRIC_KEYS},
                        now,
                    )
            self.series.drop_missing({s.uid for s in self.specs})
            self.last_fast = now

    # -- reading -----------------------------------------------------------

    def read(self):
        """A consistent copy of everything collected so far."""
        with self._lock:
            return {
                "containers": dict(self.containers),
                "unreachable": dict(self.unreachable),
                "health": dict(self.health),
                "metrics": dict(self.metrics),
                "series": self.series.as_dict(),
                "spend_rows": list(self.spend_rows),
                "spend_error": self.spend_error,
                "last_slow": self.last_slow,
                "last_fast": self.last_fast,
            }
