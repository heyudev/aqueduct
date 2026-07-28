# 部署与运维手册

面向要动手部署或值班排障的人。架构背景见 [design/hld.md](design/hld.md)，硬件与模型选型见 [hardware.md](hardware.md)。

## 两类机器

引入集群编排后，命令分两处执行，先分清楚：

- **控制机**：跑 `aqctl`（`make plan` / `apply` / `status`）和网关。可以是任意一台普通机器，甚至你的开发机。它通过 SSH 操作 GPU 机器。
- **GPU 机器**：只跑 vLLM 容器。上面**不需要装本项目的任何东西**，只要有 docker、nvidia-container-toolkit 和 SSH。

下面每一步都标了在哪执行。

## 首次部署

顺序是固定的：体检 → 下模型 → 单机验证 → 纳入集群 → 起网关 → 冒烟 → 压测。前一步不通过不要走下一步，否则出问题很难定位是哪层。

### 1. 机器体检（在 GPU 机器上）

新机器到货后第一件事：

```bash
git clone <repo> aqueduct && cd aqueduct
make env          # 生成 .env
vi .env           # 填密钥、MODEL_DIR
make preflight
```

必须全部 PASS 才继续。常见 FAIL 及处理：

| 报错 | 原因 | 处理 |
|---|---|---|
| `nvidia-smi not found` | 驱动没装 | 装 NVIDIA 驱动，重启 |
| `cannot access GPUs from a container` | 缺 nvidia-container-toolkit | 装完执行 `systemctl restart docker` |
| `no NVLink between GPUs` | 拿到的是 PCIe 版而非 SXM | 找供应商核对型号，这会严重影响 TP=8 性能 |
| `uncorrected ECC errors` | 显卡硬件故障 | 报修换卡，不要带病上线 |

### 2. 下载模型权重（在 GPU 机器上）

几百 GB，走内网镜像源会快很多。

```bash
./scripts/download-model.sh Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
```

下完确认落盘路径和 `models.yaml` 里该模型的 `path` 一致，且在 `cluster.yaml` 的 `model_dir` 之下。

### 3. 单机验证（在 GPU 机器上，只做一次）

在把机器交给集群管理之前，先确认它自己能跑起来：

```bash
make up-model MODEL=primary
make logs-model          # 另开一个窗口看加载过程
make wait-model          # 阻塞直到 /health 返回
```

日志里出现 `Application startup complete` 就说明这台机器没问题。**验证完把它停掉**：

```bash
make down-model
```

这一步只是排除硬件和环境问题。留着不停会占住 GPU，导致后面 `make apply` 起不来。

### 4. 纳入集群（在控制机上）

把这台机器写进 `cluster.yaml`，填真实 IP：

```yaml
hosts:
  - name: gpu-01
    address: 10.0.0.11
    gpu_count: 8
    deployments:
      - model: primary
        gpus: [0, 1, 2, 3, 4, 5, 6, 7]
        port: 8001
```

然后：

```bash
make validate     # 校验声明，不连机器
make plan         # 看将要做什么
make apply        # 执行
```

`apply` 会起容器、等健康检查、生成网关配置。加载几百 GB 权重要几十分钟，超时时间取 `models.yaml` 里该模型的 `health_timeout`。

### 5. 启动网关（在控制机上）

```bash
make up-gateway
make logs-gateway
```

网关配置已由 `apply` 生成，`gateway/config.yaml` 里的 `api_base` 指向各 GPU 机器的内网地址。**不要手工编辑这个文件**——它会在下次 apply 时被覆盖。要改策略改 `gateway/config.base.yaml`，要改拓扑改 `cluster.yaml`。

### 6. 冒烟

```bash
make smoke
```

五项全过说明端到端链路通了。任一项失败，先看 `make logs-gateway`，再看具体实例：`make logs HOST=gpu-01 MODEL=primary`。

### 7. 压测定容量

```bash
make bench
```

跑完把关键数字记在本文档末尾的「实测基线」一节。没有基线，以后任何调优都是凭感觉。

## 签发 API key

给业务方开通：

```bash
TEAM=search BUDGET=200 ./scripts/mint-key.sh
```

key 只显示一次，网关只存哈希。丢了就重新签发，没法找回。通过安全渠道交付，不要贴在群里。

查用量：

```bash
make usage
```

## 日常操作

日常变更都是同一个动作：**改声明 → `make plan` 看清楚 → `make apply`**。改完记得提交，`cluster.yaml` 的 git 历史就是部署历史。

**看现在什么在跑**

```bash
make status
```

每台机器每张卡的占用、实例状态、在途请求数、KV cache 使用率。配置漂移会标成 `STALE CONFIG`，声明里已删除但还在跑的会标成 `UNDECLARED`。

要在浏览器里看同样的信息，外加实时负载曲线和给业务方的接入说明：

```bash
make console
```

只读，改不了任何东西——变更一律走 `cluster.yaml` 加 `apply`。默认只监听本机，要给同事看就 `make console BIND=0.0.0.0`，前提是有防火墙。

**换模型**：改 `models.yaml` 里该模型的 `path`，`make apply`。业务方无感知——他们调的是 `aqueduct-chat` 这个逻辑名。**单副本时这会中断服务**，`plan` 会明确警告，权重重新加载要几十分钟。

**调参数**：改 `models.yaml` 里的 `max_model_len`、`kv_cache_dtype` 等，`make apply`。同样会重建实例。

**加机器**：新机器走完首次部署的第 1–3 步，然后在 `cluster.yaml` 加一个 `hosts` 条目，`make apply`。网关配置自动包含新后端，多副本自动负载均衡。

**下线机器**：从 `cluster.yaml` 删掉该 `hosts` 条目，`make apply`。它上面的容器会被停掉并移除，网关配置里的对应后端也会消失。

**改网关策略**（超时、重试、日志）：改 `gateway/config.base.yaml`，然后 `make gen-gateway && make restart-gateway`。不要改 `gateway/config.yaml`，那是生成产物。

**多副本时怎么不中断**：逐台改。把一台机器的模型换掉、apply、等它健康，再动下一台。期间网关把流量导向仍在服务的副本。单副本没有这个选项——只能接受中断，或者先加机器。

## 排障

**首 token 特别慢（TTFT 几十秒）**

H20 预填充弱，长上下文本来就慢。先确认 prefix caching 是否在生效——看 vLLM 指标里的 `gpu_prefix_cache_hit_rate`。命中率低说明业务方每次请求的前缀都不同，推动他们把系统提示词固定下来。再确认 `models.yaml` 里的 `max_num_batched_tokens` 不要太大（H20 上 4096 起步，见 [hardware.md](hardware.md)）。

**并发一上来就报错或排队**

KV cache 不够。三个方向，按代价从小到大：调小 `max_model_len`、把 `kv_cache_dtype` 设成 `fp8`、换更小的模型（都在 `models.yaml` 里，改完 `make apply`）。`gpu_memory_utilization` 也可以往上调一点点，但超过 0.95 容易在压测中途 OOM。

**启动时卡住不动，没有报错**

多半是共享内存不足。TP 的 8 个 worker 进程通过 `/dev/shm` 交换张量，docker 默认只给 64MB。确认 `cluster.yaml` 里的 `shm_size` 是几十 GB，且宿主机 `/dev/shm` 本身够大（`make preflight` 会查）。

**网关能通但报 model not found**

两边的名字都由 `models.yaml` 的 `served_name` 生成，正常不会不一致。真报这个错，检查是不是手工改过 `gateway/config.yaml`（会被 apply 覆盖），或者 apply 之后忘了 `make restart-gateway`。

**显存泄漏 / 长时间运行后 OOM**

先记录 `nvidia-smi` 输出再重启，不要直接重启掩盖问题。vLLM 的显存是启动时一次性预分配的，运行中增长通常意味着遇到了 bug，值得报给上游。

## 值班检查项

每天扫一眼：

- Grafana 上的 P99 延迟和错误率有没有异常抬升
- `make usage` 里有没有某个 key 用量突然暴涨（多半是业务方写了死循环）
- 磁盘剩余空间。容器日志已配轮转（单容器上限 300MB），涨满通常是模型权重堆积
- 昨晚的数据库备份有没有成功落盘（`ls -lt backups/ | head`）

**数据库备份**是这里唯一丢了就回不来的东西——所有 key 和用量历史都在里面。`make backup` 手工跑一次，上线后挂 cron：

```
0 3 * * * cd /opt/aqueduct && ./scripts/backup-db.sh >> /var/log/aqueduct-backup.log 2>&1
```

跟数据库放同一台机器上的备份不算备份，要复制到别处。恢复演练至少做一次——没验证过能恢复的备份等于没有。

## 实测基线

> 压测跑完后填这里。没有实测数字之前，不要向业务方承诺任何容量指标。

| 日期 | 模型 | 输入长度 | 输出长度 | 并发 | TTFT P50/P99 | TPOT | 吞吐 (tok/s) |
|---|---|---|---|---|---|---|---|
| 待测 | | | | | | | |
