# Aqueduct

公司内部大模型推理服务。对外暴露一套 OpenAI 兼容 API，内部由 vLLM 承载开源模型，跑在 H20 GPU 服务器上。

任何业务系统只要会调 OpenAI SDK，改个 `base_url` 和 `api_key` 就能接入。

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://aqueduct.internal:4000/v1",
    api_key="sk-aq-...",          # 由 Aqueduct 签发的内部 key
)
resp = client.chat.completions.create(
    model="aqueduct-chat",         # 逻辑模型名，背后指向哪个开源模型由网关决定
    messages=[{"role": "user", "content": "你好"}],
)
```

## 现状

硬件已定：**H20 96GB，单机 8 卡共 768GB 显存**，主力模型定在 235B 级 MoE + FP8。机器尚未到货。

仓库里是「到货即可部署」的配置与脚本。**所有内容都未在真实 H20 上运行过**，文档中标注「待实测」的性能数字都是估算，第一次上机需要按 [docs/runbook.md](docs/runbook.md) 逐步验证并回填。

## 组成

| 层 | 组件 | 职责 |
|---|---|---|
| 接入 | LiteLLM Proxy | OpenAI 兼容入口、API key 鉴权、按 key 配额与用量统计、多模型路由、故障转移 |
| 推理 | vLLM | 单机 8 卡张量并行加载模型，提供 OpenAI 兼容后端 |
| 观测 | Prometheus + Grafana | 吞吐、延迟分位、KV cache 命中率、显存占用、按 key 用量 |
| 存储 | PostgreSQL | LiteLLM 的 key、预算、调用日志 |

一次请求的路径：业务系统 → LiteLLM（验 key、查配额、选实例）→ vLLM（推理）→ 原路返回，用量写入 PostgreSQL。

## 目录

```
cluster.yaml           集群声明：哪台机器、哪几张卡、跑什么模型
models.yaml            模型注册表：每个模型怎么跑
tools/aqctl/           把实际状态收敛到声明的 CLI，以及只读控制台
docs/
  design/hld.md            概要设计：架构、选型理由、非功能设计、演进路线
  design/lld.md            详细设计：接口契约、配置项、容量计算、故障处理、测试
  design/orchestration.md  集群编排设计：声明式的机器伸缩与模型部署
  hardware.md              H20 硬件特性与模型选型矩阵（采购决策依据）
  runbook.md               部署步骤与日常运维
gateway/               LiteLLM 网关：compose 与配置
inference/vllm/        vLLM 推理服务：compose 与各模型启动参数
monitoring/            Prometheus 抓取配置与 Grafana 面板
scripts/               上机体检、模型下载、冒烟测试、压测、签发 key
```

## 快速开始

机器到货后，先在 GPU 机器上体检：

```bash
make preflight
```

通过后，在控制机上编辑 `cluster.yaml` 填真实地址，然后：

```bash
make validate && make plan && make apply
```

`apply` 会起容器、等健康检查、生成网关配置。再 `make up-gateway`，最后 `make smoke` 验证端到端。完整步骤见 [docs/runbook.md](docs/runbook.md)。

日常的加机器、换模型也是同一个动作：改 `cluster.yaml` 或 `models.yaml`，`make plan` 看清楚，`make apply`。

想看集群现在什么样，`make console` 起一个只读的网页控制台（默认 `127.0.0.1:8080`）：每台机器每张卡在跑什么、实例是否健康、实时负载，以及给业务方看的模型接入说明。

## 先读哪份文档

| 你要做的事 | 读这份 |
|---|---|
| 决定买什么机器、跑什么模型 | [docs/hardware.md](docs/hardware.md) |
| 理解整体架构和技术选型的取舍 | [docs/design/hld.md](docs/design/hld.md) |
| 接入这个 API，或要改配置、定容量 | [docs/design/lld.md](docs/design/lld.md) |
| 加机器、上下线模型、看控制台 | [docs/design/orchestration.md](docs/design/orchestration.md) |
| 动手部署或值班排障 | [docs/runbook.md](docs/runbook.md) |

概要设计面向所有人，一页看懂这是什么；详细设计面向研发、测试和运维，细到能照着实施。
