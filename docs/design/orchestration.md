# 集群编排设计

> 目标：让「加一台机器」和「换一个模型」都变成改一行声明，而不是 SSH 上去手工操作。
>
> 状态：已实现，见 [`tools/aqctl/`](../../tools/aqctl/)。纯逻辑部分有单元测试（`make test`），
> **凡是涉及真实机器的路径都未验证**——机器尚未到货。整体架构见 [hld.md](hld.md)。

## 1. 结论

用一份 `cluster.yaml` 声明整个集群的期望状态——哪些机器、每台的哪几张卡跑什么模型——再用一条 `aqctl apply` 把实际状态收敛过去。网关配置从同一份声明生成。

三个刻意的选择：

- **卡位是声明的，不是调度的。** 哪几张卡跑哪个模型写死在配置里，不做自动分配。
- **不引入 Kubernetes，不写常驻控制面。** `aqctl` 是一个纯 CLI 工具，跑完就退出。
- **一张卡只属于一个实例。** 不做单卡上的多模型共享。

下面逐条说明理由。

### 为什么卡位是声明的

自动调度（"给我 4 张空闲卡"）看起来更方便，实际上带来三个问题：同样的配置在不同时刻可能落到不同的卡上，排障时无法复现；两个部署请求并发时需要处理资源竞态；出问题时没人说得清"为什么这个模型在这几张卡上"。

声明式的代价是加模型时要自己填卡号，收益是**集群状态完全可预测、可 review、可回滚**——`cluster.yaml` 进 git，每次变更都有记录。对一个只有几台机器的内部服务，这个交换是划算的。

参考实现 `gpu-scheduler` 走的是自动调度路线（从数据库查空闲卡、优先挑连续的），那套设计适合"实验性模型频繁上下线"的场景。Aqueduct 的模型矩阵相对稳定，不需要这个灵活度。

### 为什么不用 Kubernetes

k8s 能原生解决机器伸缩和自愈，但前提是有人维护 k8s 本身。为几台 GPU 机器搭一套集群、配 device plugin、学一套新的排障路径，成本高于它解决的问题。**当机器数长到十台以上，或者需要真正的自动故障恢复时，这个结论应该重新评估。**

### 为什么一张卡只属于一个实例

两个 vLLM 实例共享一张卡时，`gpu-memory-utilization` 要手工分配比例，任何一边的配置变动都可能让另一边 OOM，而且故障会互相传染。收益（多塞一个小模型）远小于代价。

需要多个小模型时，给每个分配独立的卡；卡不够就等下一台机器。

## 2. 声明什么

两份文件，职责分开：**模型注册表**描述"一个模型该怎么跑"，**集群声明**描述"哪台机器跑哪些模型"。

### 2.1 模型注册表 `models.yaml`

一个模型一条记录，加新模型就是加一条，不碰代码。

```yaml
models:
  primary:
    # 权重在 GPU 机器上的路径（各机器一致）
    path: /data/models/Qwen3-235B-A22B-Instruct-2507-FP8
    # 网关看到的名字，也是 vLLM 的 --served-model-name
    served_name: aqueduct-chat
    # 需要几张卡，同时也是 --tensor-parallel-size
    gpus: 8
    # 权重占用，用于校验显存是否够（GB）
    weight_gb: 235
    max_model_len: 65536
    kv_cache_dtype: fp8
    max_num_batched_tokens: 4096
    max_num_seqs: 64
    gpu_memory_utilization: 0.92
    extra_args: ["--enable-expert-parallel"]
    # 权重大的模型加载慢，健康检查要给够时间（秒）
    health_timeout: 1800

  fast:
    path: /data/models/Qwen3-30B-A3B-Instruct-FP8
    served_name: aqueduct-fast
    gpus: 2
    weight_gb: 30
    max_model_len: 32768
    kv_cache_dtype: fp8
    max_num_batched_tokens: 8192
    max_num_seqs: 128
    gpu_memory_utilization: 0.90
    health_timeout: 600

  embed:
    path: /data/models/Qwen3-Embedding-8B
    served_name: aqueduct-embed
    gpus: 1
    weight_gb: 8
    task: embedding          # 走 pooling runner，不需要 max_model_len
    gpu_memory_utilization: 0.85
    health_timeout: 300
```

`weight_gb` 和 `health_timeout` 这两个字段是从 `gpu-scheduler` 的实践里借来的：前者让工具能在部署前算出显存够不够，后者是因为大模型加载可能要半小时，统一的健康检查超时会误判成失败。

### 2.2 集群声明 `cluster.yaml`

```yaml
# 所有机器共用的默认值，单机可覆盖
defaults:
  ssh_user: root
  model_dir: /data/models
  image: vllm/vllm-openai:v0.11.0     # 锁定版本，不用 latest

hosts:
  - name: gpu-01
    address: 10.0.0.11
    gpu_count: 8
    deployments:
      - model: primary                 # 引用 models.yaml 里的 key
        gpus: [0, 1, 2, 3, 4, 5, 6, 7]
        port: 8001

  - name: gpu-02
    address: 10.0.0.12
    gpu_count: 8
    deployments:
      - model: primary
        gpus: [0, 1, 2, 3]
        port: 8001
      - model: fast
        gpus: [4, 5]
        port: 8002
      - model: embed
        gpus: [6]
        port: 8003
      # 卡 7 留空，作为备用
```

加一台机器 = 加一个 `hosts` 条目。换模型 = 改一个 `model` 字段。调参数 = 改 `models.yaml`。三件事都不需要登录任何机器。

## 3. 校验规则

`aqctl validate` 在任何变更前跑，把能静态发现的错误挡在部署之前。这一层的价值在于：这些错误如果漏到部署阶段，表现是模型起不来或者半夜 OOM，排查成本高得多。

| 规则 | 为什么 |
|---|---|
| 同一台机器上卡号不重叠 | 两个实例抢同一张卡，后启动的直接 OOM |
| 卡号在 `[0, gpu_count)` 范围内 | 写错卡号，容器起不来但错误信息不直观 |
| 同一台机器上端口不重复 | 端口冲突，后启动的退出 |
| `len(gpus) == models[m].gpus` | 声明的卡数和模型要求的 TP 数必须一致 |
| TP 数是 1 / 2 / 4 / 8 | TP 数要能整除模型的注意力头数，其他值通常起不来 |
| `weight_gb < len(gpus) × 96 × gpu_memory_utilization` | 权重放不下就没必要启动。留出的余量还要够 KV cache |
| 每台机器至少留一个 KV cache 下限 | 权重占满、KV cache 只剩几十 GB 的配置能起来但没有可用并发 |
| `model` 在 `models.yaml` 中存在 | 拼写错误 |
| 同一个 `served_name` 的实例参数一致 | 同名模型的多个副本如果 `max_model_len` 不同，调用方会看到随机的行为 |

最后一条容易被忽略：网关按 `served_name` 做负载均衡，两个副本如果上下文长度不同，同样的请求打到不同副本会得到不同结果。

## 4. 收敛怎么做

### 4.1 状态从哪来

**期望状态**来自 `cluster.yaml` + `models.yaml`。

**实际状态**通过 `DOCKER_HOST=ssh://user@host` 直接查远端 docker——控制机的 docker CLI 通过 SSH 操作远端守护进程，**GPU 机器上不需要安装任何 Aqueduct 的东西**，只要有 docker 和 SSH。这是选这个方案而不是自研 agent 的主要原因：没有需要自己运维的常驻进程。

容器上打标签，状态识别全靠它，不解析命令行：

```
aqueduct.managed=true          # 只管自己起的容器，不碰机器上的其他容器
aqueduct.model=primary
aqueduct.gpus=0,1,2,3
aqueduct.spec-hash=<指纹>
```

`spec-hash` 是该实例全部启动参数的哈希。**参数变了指纹就变**，这是判断"要不要重建"的唯一依据，比逐项比对命令行可靠。

### 4.2 diff 与执行

```
对每台机器：
  实际 = 查 label aqueduct.managed=true 的容器
  期望 = cluster.yaml 里该机器的 deployments

  期望有、实际无            → 创建
  实际有、期望无            → 删除
  两边都有、spec-hash 不同  → 重建（先删后建）
  两边都有、spec-hash 相同  → 不动
```

执行顺序上**先删后建**，因为删除才能释放 GPU；同一台机器上的操作串行，不同机器之间并行。

创建后不等待，统一在最后轮询各实例的 `/health`，超时时间取模型注册表里的 `health_timeout`。几百 GB 的权重加载要几十分钟，逐个同步等待没有意义。

### 4.3 命令

```bash
aqctl validate        # 只校验声明，不连接任何机器
aqctl plan            # 显示将要做什么，不执行 —— 变更前必看
aqctl apply           # 执行收敛
aqctl status          # 每台机器每张卡当前在跑什么
aqctl logs <host> <model>
aqctl gen-gateway     # 从声明生成网关配置
```

`plan` 和 `apply` 分开是有意的。`plan` 的输出应该让人一眼看出"这次变更会重启哪些实例"——重启主力模型意味着几十分钟的服务中断，这种事必须在执行前看见。

### 4.4 一个约束：重建即中断

当前设计下，改动一个正在运行的实例的参数会导致它被销毁重建，期间该实例不可用。多副本时网关会把流量切到其他副本，单副本时就是服务中断。

滚动更新（先起新的再停旧的）在单机上做不到——显存不够同时放两份权重。要做到不中断，只能靠多副本，逐台 apply。这是 `plan` 必须先看的原因。

## 5. 网关配置怎么生成

`aqctl gen-gateway` 遍历所有机器的所有实例，按 `served_name` 聚合成 LiteLLM 的 `model_list`：

```yaml
# 自动生成，不要手工编辑
model_list:
  - model_name: aqueduct-chat
    litellm_params:
      model: openai/aqueduct-chat
      api_base: http://10.0.0.11:8001/v1
      api_key: os.environ/VLLM_API_KEY
  - model_name: aqueduct-chat          # 同名 = 自动负载均衡
    litellm_params:
      model: openai/aqueduct-chat
      api_base: http://10.0.0.12:8001/v1
      api_key: os.environ/VLLM_API_KEY
  - model_name: aqueduct-fast
    litellm_params:
      model: openai/aqueduct-fast
      api_base: http://10.0.0.12:8002/v1
      api_key: os.environ/VLLM_API_KEY
```

生成的部分和手工维护的部分（`general_settings`、`litellm_settings`、`router_settings`）分开存放，生成的文件带"自动生成"头注释，避免有人手工改了又被覆盖。

`apply` 完成后自动重新生成并重启网关。网关重启只影响新连接，是秒级的。

## 6. 控制台

`aqctl serve` 起一个**只读**的网页控制台（`make console`，默认 `127.0.0.1:8080`）。

**只读是刻意的。** 页面上没有任何能改变集群的按钮——所有变更都走 `cluster.yaml` 和 `apply`。一个能重启实例、改卡位的控制台会悄悄变成第二个真相来源，集群状态就不再能从 git 复现了。

### 三个页面，各自的理由

| 页面 | 回答什么 | 为什么不是用现成的 |
|---|---|---|
| 集群 | 每台机器每张卡在跑什么，健康吗，负载多少 | 没有现成方案。这是 `aqctl status` 的网页版 |
| 模型接入 | 业务方能调哪些模型、怎么调、支持哪些参数 | 现在只能翻文档，接入成本高 |
| 用量 | 谁用得异常 | 网关自带管理界面功能更全，这里只做「一眼发现跑飞的 key」，管理动作跳转过去 |

**指标没有独立页面**，实时负载和最近 30 分钟的曲线直接画在集群页每个实例那一行。做成独立的监控页就是山寨 Grafana；而 Grafana 做不到的，是把负载曲线放在它所属的卡位旁边——排障时看的正是这个。历史趋势和告警仍然指向 Grafana，控制台右上角有入口。

### 数据怎么来

两个后台采集循环，周期不同，因为代价差两个数量级：

| 循环 | 采什么 | 周期 | 代价 |
|---|---|---|---|
| 慢 | 各机器的容器清单（SSH）、网关用量 | 30s | 每台机器秒级 |
| 快 | 各实例的 `/health` 和 `/metrics`（HTTP） | 10s | 毫秒级 |

**页面请求永远不碰机器**，只读最近一次采集的快照。否则一台宕机的机器会让整个页面卡在 SSH 握手上。

指标滑窗只在内存里，服务重启即清空——这是刻意的，控制台不该变成一个时序数据库。

### 一个实现上的坑

容器名是 `aqueduct-<模型>-<端口>`，**跨机器不唯一**：两台机器跑同一模型的副本，容器名完全相同。所以任何按实例存的数据（健康、指标、滑窗）都必须用 `<机器名>/<容器名>` 做 key，否则一个副本会覆盖另一个。多副本正是这套架构的核心场景，这个坑踩下去很难发现。

### 外部链接一律显式配置，不推断

控制台右上角指向 Grafana 和网关管理界面的链接，**必须在 `.env` 里显式配置，没配就不显示**。不从网关地址拼、不按默认端口猜。

理由是猜出来的地址往往能打开，只是打开的不是你以为的东西——比如按 Grafana 默认端口猜出的 `:3000`，在一台机器上可能跑着完全无关的服务。页面上没有任何东西能告诉读者这个链接是猜的。缺一个链接远比给一个自信的错链接好。

### 未声明的容器会占卡

如果机器上有我们管理的容器却不在声明里（上次 apply 之前留下的、或者手工起的），它占用的卡会被标成「非声明」而不是「空闲」。把这些卡显示成空闲会两头误导：以为有容量可用，而下次 apply 试图占用它们时会失败。

## 7. 与现有配置的关系

引入 `aqctl` 后，仓库里会有两套启动 vLLM 的方式，必须说清楚各自的定位，否则会造成混乱：

| 方式 | 定位 |
|---|---|
| `aqctl apply` | **日常使用**。多机、多实例、声明式 |
| `make up-model` + `inference/vllm/docker-compose.yml` | **仅用于单机首次验证和调试**。新机器到货后先用它确认能跑通，再纳入集群管理 |

`inference/vllm/models/*.env` 里的参数会迁移到 `models.yaml`，同一份数据不维护两处。迁移完成后 `.env` 文件保留只是为了 compose 调试路径能用。

## 8. 不做什么

- **不做自动扩缩容。** 负载高了就改声明加机器，没有基于指标的自动决策。
- **不做故障自愈。** 实例挂了 `aqctl status` 会显示出来，但不会自动拉起。容器本身的 `restart: unless-stopped` 能处理进程崩溃，处理不了机器宕机。
- **不做灰度发布。** 换模型是替换，不是渐进切流。需要灰度时用不同的 `served_name` 并存，由网关侧控制，见 [lld.md](lld.md) 的模型上线流程。
- **不管理机器本身。** 装驱动、装 docker、挂载磁盘不在范围内，那是 `scripts/preflight.sh` 检查、由人处理的事。
- **控制台不做任何变更操作。** 没有重启按钮，没有改卡位的表单。理由见上一节。

## 9. 实现

工具在 [`tools/aqctl/`](../../tools/aqctl/)，Python 3，依赖只有 PyYAML——docker 操作全部通过 `docker -H ssh://...` 子进程调用，不引入 SDK。

模块划分的原则是**把决策和 IO 分开**：

| 模块 | 职责 | 有 IO |
|---|---|---|
| `config.py` | 解析与校验声明 | 否 |
| `spec.py` | 声明 → 容器规格 + 指纹 | 否 |
| `planner.py` | 期望 vs 实际 → 动作列表 | 否 |
| `gateway.py` | 声明 → LiteLLM model_list | 否 |
| `remote.py` | 远端 docker 与健康检查 | **是** |
| `cli.py` | 参数解析与命令编排 | — |

决策全在纯函数里，因此不需要 GPU 机器就能测；`remote.py` 刻意保持很薄，因为它是唯一测不了的部分。

### 已验证与未验证

`make test` 覆盖校验规则、diff 算法、容器参数构造、指纹计算、网关生成，共 64 个用例。此外在本机验证过 `validate`、`gen-gateway` 的完整链路，以及机器不可达时的降级行为。

**未验证**：任何真正操作机器的路径——创建容器、删除容器、健康检查轮询、`status` 的指标抓取。这些要等机器到货后按 [../runbook.md](../runbook.md) 逐步验证。

### 前置条件

- 控制机：Python 3 + PyYAML、docker CLI、能免密 SSH 到所有 GPU 机器
- GPU 机器：docker + nvidia-container-toolkit（`make preflight` 覆盖），**不需要装本项目的任何东西**
