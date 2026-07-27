# 部署与运维手册

面向要动手部署或值班排障的人。架构背景见 [design/hld.md](design/hld.md)，硬件与模型选型见 [hardware.md](hardware.md)。

## 首次部署

顺序是固定的：体检 → 下模型 → 起推理 → 起网关 → 冒烟 → 压测。前一步不通过不要走下一步，否则出问题很难定位是哪层。

### 1. 机器体检

新机器到货后第一件事，在 GPU 机器上：

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

### 2. 下载模型权重

几百 GB，走内网镜像源会快很多。

```bash
./scripts/download-model.sh Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
```

下完确认 `inference/vllm/models/primary.env` 里的 `MODEL_PATH` 指向的是实际落盘目录。

### 3. 启动推理

```bash
make up-model MODEL=primary
make logs-model          # 另开一个窗口看加载过程
make wait-model          # 阻塞直到 /health 返回
```

首次加载几百 GB 权重需要几分钟到几十分钟，取决于磁盘速度。日志里出现 `Application startup complete` 才算起来了。

### 4. 启动网关

网关可以和推理在同一台机器，也可以单独一台。

```bash
make up-gateway
make logs-gateway
```

注意 `.env` 里的 `VLLM_PRIMARY_URL` 必须是**网关容器能访问到的地址**。同机部署时写宿主机的内网 IP，不要写 `127.0.0.1`——那在容器里指向容器自己。

### 5. 冒烟

```bash
make smoke
```

五项全过说明端到端链路通了。任一项失败，先看 `make logs-gateway` 再看 `make logs-model`。

### 6. 压测定容量

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

**换模型**：改 `inference/vllm/models/primary.env` 里的 `MODEL_PATH`，然后 `make down-model && make up-model`。业务方无感知——他们调的是 `aqueduct-chat` 这个逻辑名。

**改网关配置**：改 `gateway/config.yaml` 后 `make restart-gateway`。这只重启网关进程，数据库和已签发的 key 不受影响。

**加第二台 GPU 机器**：在新机器上重复步骤 1–3，然后在网关的 `.env` 里填 `VLLM_PRIMARY_URL_2`，取消 `gateway/config.yaml` 里第二个 `aqueduct-chat` 条目的注释，`make restart-gateway`。网关会自动在两个后端间分流。

## 排障

**首 token 特别慢（TTFT 几十秒）**

H20 预填充弱，长上下文本来就慢。先确认 prefix caching 是否在生效——看 vLLM 指标里的 `gpu_prefix_cache_hit_rate`。命中率低说明业务方每次请求的前缀都不同，推动他们把系统提示词固定下来。再确认 `MAX_NUM_BATCHED_TOKENS` 不要太大（H20 上 4096 起步，见 [hardware.md](hardware.md)）。

**并发一上来就报错或排队**

KV cache 不够。三个方向，按代价从小到大：调小 `MAX_MODEL_LEN`、把 `KV_CACHE_DTYPE` 设成 `fp8`、换更小的模型。`GPU_MEM_UTIL` 也可以往上调一点点，但超过 0.95 容易在压测中途 OOM。

**启动时卡住不动，没有报错**

多半是共享内存不足。TP 的 8 个 worker 进程通过 `/dev/shm` 交换张量，docker 默认只给 64MB。确认 `.env` 里 `SHM_SIZE` 是几十 GB，且宿主机 `/dev/shm` 本身够大。

**网关能通但报 model not found**

`gateway/config.yaml` 里 `openai/` 后面的名字必须和 vLLM 的 `--served-model-name` 完全一致。两边都改过之后容易忘记同步其中一边。

**显存泄漏 / 长时间运行后 OOM**

先记录 `nvidia-smi` 输出再重启，不要直接重启掩盖问题。vLLM 的显存是启动时一次性预分配的，运行中增长通常意味着遇到了 bug，值得报给上游。

## 值班检查项

每天扫一眼：

- Grafana 上的 P99 延迟和错误率有没有异常抬升
- `make usage` 里有没有某个 key 用量突然暴涨（多半是业务方写了死循环）
- 磁盘剩余空间，模型权重加日志容易涨满

## 实测基线

> 压测跑完后填这里。没有实测数字之前，不要向业务方承诺任何容量指标。

| 日期 | 模型 | 输入长度 | 输出长度 | 并发 | TTFT P50/P99 | TPOT | 吞吐 (tok/s) |
|---|---|---|---|---|---|---|---|
| 待测 | | | | | | | |
