/* Aqueduct console — read-only.
 *
 * Polls the JSON API and renders. No mutation endpoints exist: changes go
 * through cluster.yaml and `aqctl apply`, so the deployment stays reproducible
 * from git.
 *
 * Values are inserted as text, never as markup — host names, model keys and
 * key aliases all come from configuration and are not assumed to be safe.
 */

const REFRESH_MS = 10000;

const state = {
  tab: "cluster",
  timer: null,
  lastOk: 0,
  failures: 0,
};

/* ---- helpers --------------------------------------------------------- */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function ago(seconds) {
  if (!seconds) return "尚未采集";
  const delta = Math.max(0, Date.now() / 1000 - seconds);
  if (delta < 60) return Math.round(delta) + " 秒前";
  if (delta < 3600) return Math.round(delta / 60) + " 分钟前";
  return Math.round(delta / 3600) + " 小时前";
}

function pct(value) {
  return value === undefined || value === null ? "—" : Math.round(value * 100) + "%";
}

async function getJSON(path) {
  const resp = await fetch(path, { cache: "no-store" });
  if (!resp.ok) throw new Error("HTTP " + resp.status);
  return resp.json();
}

/* Status vocabulary. Every state maps to a reserved status colour AND a
 * label — the colour never carries the meaning by itself. */
const STATES = {
  running:       { tone: "good",     label: "运行中" },
  stale_config:  { tone: "warning",  label: "配置已变更" },
  stopped:       { tone: "critical", label: "已停止" },
  not_deployed:  { tone: "serious",  label: "未部署" },
  unknown:       { tone: "",         label: "无法确认" },
  undeclared:    { tone: "serious",  label: "声明外占用" },
};

function statusChip(stateKey, healthy) {
  const spec = STATES[stateKey] || STATES.unknown;
  let tone = spec.tone;
  let label = spec.label;
  if (stateKey === "running" && !healthy) {
    tone = "warning";
    label = "无响应";
  }
  return el("span", "status " + tone, label);
}

/* A sparkline: one series, so no legend — the label beside it names it.
 * 2px stroke, no axes, no dots except the current value. */
function sparkline(points, width, height) {
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  width = width || 108;
  height = height || 26;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
  svg.setAttribute("class", "spark");
  svg.setAttribute("role", "img");

  if (!points || points.length < 2) {
    svg.setAttribute("aria-label", "数据点不足");
    return svg;
  }

  const max = Math.max(...points, 1);
  const pad = 3;
  const stepX = (width - pad * 2) / (points.length - 1);
  const scaleY = (v) => height - pad - (v / max) * (height - pad * 2);

  const path = points
    .map((v, i) => `${i === 0 ? "M" : "L"}${(pad + i * stepX).toFixed(1)},${scaleY(v).toFixed(1)}`)
    .join(" ");

  const line = document.createElementNS(NS, "path");
  line.setAttribute("d", path);
  line.setAttribute("fill", "none");
  line.setAttribute("stroke", "var(--series-1)");
  line.setAttribute("stroke-width", "2");
  line.setAttribute("stroke-linecap", "round");
  line.setAttribute("stroke-linejoin", "round");
  svg.appendChild(line);

  const dot = document.createElementNS(NS, "circle");
  dot.setAttribute("cx", (pad + (points.length - 1) * stepX).toFixed(1));
  dot.setAttribute("cy", scaleY(points[points.length - 1]).toFixed(1));
  dot.setAttribute("r", "2.5");
  dot.setAttribute("fill", "var(--series-1)");
  svg.appendChild(dot);

  svg.setAttribute("aria-label",
    `最近 ${points.length} 个采样点，当前 ${points[points.length - 1]}，峰值 ${max}`);
  return svg;
}

function statTile(label, value, unit, sub) {
  const tile = el("div", "stat");
  tile.appendChild(el("div", "stat-label", label));
  const v = el("div", "stat-value", value);
  if (unit) {
    const u = el("span", "unit", " " + unit);
    v.appendChild(u);
  }
  tile.appendChild(v);
  if (sub) tile.appendChild(el("div", "stat-sub", sub));
  return tile;
}

function table(headers, rows) {
  const wrap = el("div", "table-wrap");
  const t = el("table");
  const thead = el("thead");
  const htr = el("tr");
  headers.forEach((h) => {
    const th = el("th", h.num ? "num" : null, h.label);
    htr.appendChild(th);
  });
  thead.appendChild(htr);
  t.appendChild(thead);

  const tbody = el("tbody");
  rows.forEach((cells) => {
    const tr = el("tr");
    cells.forEach((cell, i) => {
      const td = el("td", headers[i] && headers[i].num ? "num" : null);
      if (cell instanceof Node) td.appendChild(cell);
      else td.textContent = cell === null || cell === undefined ? "—" : String(cell);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  t.appendChild(tbody);
  wrap.appendChild(t);
  return wrap;
}

function emptyState(message) {
  return el("div", "empty", message);
}

/* ---- cluster tab ----------------------------------------------------- */

function renderCluster(data) {
  const stats = document.getElementById("cluster-stats");
  clear(stats);

  const s = data.summary;
  stats.appendChild(statTile("机器", data.hosts.length, "台",
    s.hosts_unreachable ? s.hosts_unreachable + " 台连不上" : "全部可达"));
  stats.appendChild(statTile("实例健康", `${s.healthy}/${s.instances}`, "",
    s.healthy === s.instances ? "全部正常" : "有实例未就绪"));
  stats.appendChild(statTile("GPU 占用", `${s.gpus_used}/${s.gpus_total}`, "",
    `${s.gpus_total - s.gpus_used} 张空闲`));

  const inflight = data.hosts.reduce((sum, h) =>
    sum + h.instances.reduce((n, i) => n + (i.metrics.running || 0), 0), 0);
  const queued = data.hosts.reduce((sum, h) =>
    sum + h.instances.reduce((n, i) => n + (i.metrics.waiting || 0), 0), 0);
  stats.appendChild(statTile("在途请求", inflight, "",
    queued > 0 ? `${queued} 个排队中` : "无排队"));

  const list = document.getElementById("cluster-hosts");
  clear(list);

  if (!data.hosts.length) {
    list.appendChild(emptyState("cluster.yaml 里还没有声明任何机器。"));
    return;
  }

  data.hosts.forEach((host) => list.appendChild(renderHost(host)));
}

function renderHost(host) {
  const card = el("div", "host");

  const head = el("div", "host-head");
  const left = el("div");
  const name = el("span", "host-name", host.name);
  left.appendChild(name);
  left.appendChild(el("span", "host-addr", host.address));
  head.appendChild(left);

  const used = host.gpus.filter((g) => g.model).length;
  const right = el("div", "instance-right");
  right.appendChild(el("span", "instance-meta", `${used}/${host.gpu_count} 卡占用`));
  right.appendChild(host.reachable
    ? el("span", "status good", "可达")
    : el("span", "status critical", "连不上"));
  head.appendChild(right);
  card.appendChild(head);

  const grid = el("div", "gpu-grid");
  host.gpus.forEach((gpu) => {
    const tile = el("div", "gpu " + (gpu.model ? gpu.state : "free"));
    tile.appendChild(el("div", "gpu-id", gpu.id));
    tile.appendChild(el("div", "gpu-bar"));
    tile.appendChild(el("div", "gpu-tag", gpu.model || "空闲"));
    tile.title = gpu.model
      ? `GPU ${gpu.id} — ${gpu.model}（${(STATES[gpu.state] || STATES.unknown).label}）`
      : `GPU ${gpu.id} — 空闲`;
    if (gpu.state === "undeclared") {
      // The label is the container name, which is long; the tooltip carries it.
      tile.querySelector(".gpu-tag").textContent = "非声明";
    }
    grid.appendChild(tile);
  });
  card.appendChild(grid);

  if (!host.reachable) {
    const err = el("p", "host-error", "无法查询这台机器的实际状态，下面显示的是声明内容。");
    if (host.error) err.appendChild(el("code", null, host.error));
    card.appendChild(err);
  }

  const instances = el("div", "instances");
  host.instances.forEach((inst) => instances.appendChild(renderInstance(inst)));
  card.appendChild(instances);

  if (host.undeclared && host.undeclared.length) {
    const warn = el("div", "undeclared");
    warn.appendChild(el("strong", null, "声明外的容器："));
    warn.appendChild(document.createTextNode(
      " " + host.undeclared.map((u) => {
        const cards = u.gpus && u.gpus.length ? ` 占用 GPU ${u.gpus.join(",")}` : "";
        return `${u.container}（${u.state}${cards}）`;
      }).join("、") +
      "。它们不在 cluster.yaml 里，下次 apply 会移除。"));
    card.appendChild(warn);
  }

  return card;
}

function renderInstance(inst) {
  const row = el("div", "instance");

  const main = el("div", "instance-main");
  const title = el("div", "instance-title");
  title.appendChild(el("span", "instance-model", inst.model));
  title.appendChild(el("span", "arrow", "→"));
  title.appendChild(el("span", "served", inst.served_name));
  main.appendChild(title);

  const meta = el("div", "instance-meta");
  const bits = [
    `GPU ${inst.gpus.join(",")}`,
    `:${inst.port}`,
    `权重 ${inst.weight_gb}GB`,
    `KV 余量 ${inst.kv_headroom_gb}GB`,
  ];
  if (inst.max_model_len) bits.push(`上下文 ${inst.max_model_len / 1024}k`);
  bits.forEach((b, i) => {
    if (i) meta.appendChild(el("span", "sep", "·"));
    meta.appendChild(document.createTextNode(b));
  });
  main.appendChild(meta);
  row.appendChild(main);

  const right = el("div", "instance-right");
  const m = inst.metrics || {};
  const window_ = inst.window || [];

  if (inst.state === "running" && Object.keys(m).length) {
    // The load curve sits on the same row as the GPU assignment it belongs
    // to. Grafana holds the history; this is for "what is it doing now".
    const load = el("div", "metric-cell");
    load.appendChild(el("div", "metric-label", "在途 / 排队"));
    load.appendChild(el("div", "instance-meta",
      `${m.running || 0} / ${m.waiting || 0}`));
    if (window_.length > 1) {
      load.appendChild(sparkline(window_.map((p) => p.values.running || 0), 108, 24));
    }
    right.appendChild(load);

    const kv = el("div", "metric-cell");
    kv.appendChild(el("div", "metric-label", "KV cache " + pct(m.kv_cache)));
    const meter = el("div", "meter");
    const fill = el("span");
    const kvPct = Math.round((m.kv_cache || 0) * 100);
    fill.style.width = Math.min(100, kvPct) + "%";
    // KV cache near full is the concurrency ceiling, not a cosmetic detail.
    if (kvPct >= 95) fill.style.background = "var(--critical)";
    else if (kvPct >= 80) fill.style.background = "var(--warning)";
    meter.appendChild(fill);
    kv.appendChild(meter);
    if (window_.length > 1) {
      kv.appendChild(sparkline(window_.map((p) => p.values.kv_cache || 0), 108, 24));
    }
    right.appendChild(kv);
  }

  right.appendChild(statusChip(inst.state, inst.healthy));
  row.appendChild(right);
  return row;
}

/* ---- models tab ------------------------------------------------------ */

const SUPPORT_LABEL = {
  yes: "支持",
  partial: "部分支持",
  no: "不支持",
  unconfirmed: "未承诺",
};

function renderModels(data) {
  const snippet = document.getElementById("snippet");
  const first = data.models.find((m) => m.kind === "chat");
  clear(snippet);
  snippet.textContent =
    "from openai import OpenAI\n\n" +
    "client = OpenAI(\n" +
    `    base_url="${data.gateway_url}",\n` +
    '    api_key="sk-aq-...",\n' +
    ")\n" +
    "resp = client.chat.completions.create(\n" +
    `    model="${first ? first.served_name : "aqueduct-chat"}",\n` +
    '    messages=[{"role": "user", "content": "你好"}],\n' +
    "    max_tokens=512,\n" +
    ")";

  const host = document.getElementById("models-table");
  clear(host);

  if (!data.models.length) {
    host.appendChild(emptyState("models.yaml 里还没有定义模型。"));
  } else {
    const rows = data.models.map((m) => {
      const status = m.status === "available"
        ? el("span", "status good", "可用")
        : m.status === "degraded"
        ? el("span", "status warning", `降级 ${m.healthy_replicas}/${m.replicas}`)
        : m.status === "down"
        ? el("span", "status critical", "不可用")
        : el("span", "status serious", "未部署");

      const name = el("code", null, m.served_name);
      const hosts = el("span");
      (m.hosts.length ? m.hosts : ["—"]).forEach((h) => {
        hosts.appendChild(el("span", "pill", h));
      });

      return [
        name,
        m.kind === "embedding" ? "向量化" : "对话补全",
        m.max_input_tokens ? m.max_input_tokens / 1024 + "k" : "—",
        m.replicas,
        hosts,
        status,
      ];
    });

    host.appendChild(table(
      [
        { label: "逻辑模型名" },
        { label: "类型" },
        { label: "上下文" },
        { label: "副本", num: true },
        { label: "所在机器" },
        { label: "状态" },
      ],
      rows
    ));
  }

  const params = document.getElementById("params-table");
  clear(params);
  params.appendChild(table(
    [{ label: "参数" }, { label: "支持" }, { label: "说明" }],
    data.parameters.map((p) => [
      el("code", null, p.name),
      el("span", "support-" + p.support, SUPPORT_LABEL[p.support] || p.support),
      p.note || "",
    ])
  ));
}

/* ---- usage tab ------------------------------------------------------- */

function renderUsage(data) {
  const stats = document.getElementById("usage-stats");
  const attention = document.getElementById("usage-attention");
  const host = document.getElementById("usage-table");
  clear(stats);
  clear(attention);
  clear(host);

  if (data.error) {
    host.appendChild(emptyState("拿不到用量数据：" + data.error));
    return;
  }

  stats.appendChild(statTile("已签发 key", data.keys.length, "个"));
  stats.appendChild(statTile("累计消耗", data.total_spend.toFixed(2), ""));
  const over = data.needs_attention || [];
  stats.appendChild(statTile("接近配额", over.length, "个",
    over.length ? "已用超过 80%" : "无"));

  // The one thing worth checking daily, called out rather than left for the
  // reader to spot in a table.
  if (over.length) {
    const card = el("div", "card callout attention");
    card.appendChild(el("strong", null, "需要关注："));
    card.appendChild(document.createTextNode(
      " " + over.map((k) => `${k.alias}（${k.percent_used}%）`).join("、") +
      "。用量暴涨通常是调用方写了死循环，先联系他们再考虑扩额。"));
    attention.appendChild(card);
  }

  if (!data.keys.length) {
    host.appendChild(emptyState("还没有签发任何 key。"));
    return;
  }

  const rows = data.keys.map((k) => {
    // Budget shows only the ceiling — the spend already has its own column,
    // and repeating it made both harder to scan.
    const budget = k.max_budget
      ? `${k.max_budget}${k.budget_duration ? " / " + k.budget_duration : ""}`
      : "无上限";
    let usage = el("span", null, "—");
    if (k.percent_used !== null && k.percent_used !== undefined) {
      const wrap = el("div");
      const meter = el("div", "meter");
      const fill = el("span");
      fill.style.width = Math.min(100, k.percent_used) + "%";
      if (k.percent_used >= 100) fill.style.background = "var(--critical)";
      else if (k.percent_used >= 80) fill.style.background = "var(--warning)";
      meter.appendChild(fill);
      wrap.appendChild(meter);
      wrap.appendChild(el("div", "instance-meta", k.percent_used + "%"));
      usage = wrap;
    }
    const models = el("span");
    (k.models.length ? k.models : ["(全部)"]).forEach((m) => {
      models.appendChild(el("span", "pill", m));
    });
    return [k.alias, models, k.spend.toFixed(2), budget, usage];
  });

  host.appendChild(table(
    [
      { label: "Key" },
      { label: "可用模型" },
      { label: "消耗", num: true },
      { label: "配额" },
      { label: "已用" },
    ],
    rows
  ));
}

/* ---- polling and chrome ---------------------------------------------- */

const RENDERERS = {
  cluster: { url: "/api/cluster", render: renderCluster },
  models: { url: "/api/models", render: renderModels },
  usage: { url: "/api/usage", render: renderUsage },
};

/* Things this console deliberately does not do, and where they live instead.
 * Rendered as links rather than reimplemented. */
const LINK_LABELS = {
  grafana: { label: "Grafana", hint: "历史趋势与告警" },
  gateway_admin: { label: "网关管理", hint: "签发 key、完整账目" },
};

async function renderLinks() {
  const host = document.getElementById("ext-links");
  clear(host);
  let links = {};
  try {
    links = (await getJSON("/api/links")).links || {};
  } catch (err) {
    return;
  }
  Object.keys(LINK_LABELS).forEach((key) => {
    const url = links[key];
    if (!url) return;
    const meta = LINK_LABELS[key];
    const a = el("a", "ext-link", meta.label);
    a.href = url;
    a.target = "_blank";
    a.rel = "noopener";
    a.title = meta.hint;
    host.appendChild(a);
  });
}

async function refresh() {
  const spec = RENDERERS[state.tab];
  try {
    const data = await getJSON(spec.url);
    spec.render(data);
    state.failures = 0;
    setConnection("live", "更新于 " + ago(data.generated_at || Date.now() / 1000));
  } catch (err) {
    state.failures += 1;
    setConnection(state.failures > 2 ? "broken" : "stale",
      "无法连接控制台服务（" + err.message + "）");
  }
}

function setConnection(tone, text) {
  const dot = document.getElementById("conn-dot");
  const label = document.getElementById("conn-text");
  dot.className = "dot " + tone;
  label.textContent = text;
}

function selectTab(name) {
  state.tab = name;
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.setAttribute("aria-selected", String(tab.dataset.tab === name));
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.hidden = panel.id !== "panel-" + name;
  });
  refresh();
}

function initTheme() {
  const stored = localStorage.getItem("aqueduct-theme");
  if (stored) document.documentElement.setAttribute("data-theme", stored);

  document.getElementById("theme-toggle").addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme");
    const isDark = current === "dark" ||
      (current !== "light" &&
       window.matchMedia("(prefers-color-scheme: dark)").matches);
    const next = isDark ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("aqueduct-theme", next);
  });
}

function init() {
  initTheme();
  renderLinks();
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => selectTab(tab.dataset.tab));
  });
  selectTab("cluster");
  state.timer = setInterval(refresh, REFRESH_MS);

  // Don't poll a tab nobody is looking at.
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      clearInterval(state.timer);
    } else {
      refresh();
      state.timer = setInterval(refresh, REFRESH_MS);
    }
  });
}

document.addEventListener("DOMContentLoaded", init);
