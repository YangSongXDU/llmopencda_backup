# LLM Agent Sensor Tool Scheduling for Autonomous Driving

本仓库在 OpenCDA/CARLA 仿真环境中实现了一个面向自动驾驶的资源感知
LLM Agent 多传感器工具调度原型。研究关注的问题是：不同道路场景需要的感知
信息并不相同，若始终处理全部模态，会在简单场景中产生不必要的计算开销。
本项目让 Agent 根据当前环境观察、历史工具结果和调用预算，动态选择需要执行
的感知工具，并将新的工具结果用于后续风险评估和驾驶行为决策。

## 方法概览

```text
环境观察
  -> LLM Agent 评估风险与信息需求
  -> 生成结构化工具调用计划
  -> 按预算执行感知工具
  -> 更新多源感知信息
  -> SafetyShield 检查
  -> 超车状态机
  -> OpenCDA 规划与控制
  -> CSV 实验记录与性能分析
```

当前实现包含：

- `camera_tool`：提供轻量级前方 ROI 视觉线索。
- `lidar_tool`：提取前方点云距离信息。
- `radar_tool`：提取距离、相对速度和 TTC 等信息。
- `fusion_tool`：在需要时对已获得的工具结果进行结果级融合。
- LLM Agent：结合道路状态、历史结果和预算选择后续工具。
- SafetyShield：对风险评估和高层行为建议进行安全约束。
- 超车状态机：执行变道、通过、返回原车道和终止等阶段。
- 实验记录器：记录工具选择、LLM 调用、资源占用和超车过程。

`front_vehicle_debug_tool` 和 `lane_check_tool` 使用 CARLA actor/map 信息，
用于原型闭环验证。严格自车感知实验应在配置中关闭这两个工具。

## 环境准备

推荐使用项目现有版本组合：

- Ubuntu
- Python 3.7.10
- CARLA 0.9.11
- 与 CARLA 版本匹配的 Python API

创建 Python 环境并安装项目：

```bash
conda env create -f environment.yml
conda activate opencda
pip install -e .
```

资源监控中的 CPU/RSS 采样可选依赖 `psutil`；GPU 采样优先使用
`pynvml`，不可用时会尝试调用 `nvidia-smi`：

```bash
pip install psutil nvidia-ml-py3
```

如果 CARLA Python API 尚未加入环境，请按本机 CARLA 安装路径配置：

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/CARLA_0.9.11/PythonAPI/carla"
export PYTHONPATH="${PYTHONPATH}:/path/to/CARLA_0.9.11/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg"
```

更完整的基础环境说明可参考
[OpenCDA installation guide](https://opencda-documentation.readthedocs.io/en/latest/md_files/installation.html)。

## 配置 LLM

两个实验的 Agent 配置分别位于：

- `opencda/scenario_testing/config_yaml/agent_single_llm_tool_demo.yaml`
- `opencda/scenario_testing/config_yaml/agent_single_llm_tool_demo_2.yaml`

使用默认 DeepSeek 配置时，在启动实验的同一终端中设置密钥：

```bash
export DEEPSEEK_API_KEY="YOUR_API_KEY"
```

连接其他 OpenAI-compatible 服务时，修改 YAML 中以下字段：

```yaml
llm_provider: openai
llm_model: YOUR_MODEL_NAME
llm_base_url: http://YOUR_LLM_HOST:PORT/v1
api_key_env: YOUR_API_KEY_ENV
```

然后通过环境变量提供密钥：

```bash
export YOUR_API_KEY_ENV="YOUR_API_KEY"
```

客户端接受基础 URL 或完整的 `/chat/completions` URL。API 密钥只从环境变量
读取，不应写入 YAML 或提交到仓库。`robust_demo` 模式允许 API 不可用时使用
本地策略回退；论文实验应结合 CSV 中的 `llm_call_success`、
`llm_fallback_used` 和 `llm_backend` 区分真实 LLM 调用与回退结果。

## 启动实验

先在一个终端启动 CARLA：

```bash
cd /path/to/CARLA_0.9.11
./CarlaUE4.sh -quality-level=Low
```

在项目根目录的另一个终端运行场景。

双车道基础超车场景：

```bash
conda activate opencda
python opencda.py -t agent_single_llm_tool_demo -v 0.9.11
```

Town06 多车道、多车辆和重复超车场景：

```bash
conda activate opencda
python opencda.py -t agent_single_llm_tool_demo_2 -v 0.9.11
```

Demo 2 启动时会执行一次 LLM preflight，并打印模型、可用状态、HTTP 状态码、
延迟和回退状态。场景结束后会自动生成统计摘要和图表。

## 实验模式

通过 YAML 中的 `llm_sensor_agent.tool_selection` 切换工具调度策略：

| 实验模式 | 关键配置 | 用途 |
|---|---|---|
| LLM Agent | `strategy: llm` | Agent 按场景信息和预算动态选择工具 |
| Rule-based | `strategy: rule` | 使用确定性规则进行工具调度 |
| Always-all | `strategy: always_all` | 每轮执行全部可选工具，作为高开销对照 |
| No Fusion | 从 `selectable_tools` 移除 `fusion_tool`，并设置其 `enabled: false` | 检验结果级融合的作用 |
| Strict Self-perception | 设置 `prototype_oracle_tools.enabled: false` | 排除 CARLA actor/map 辅助信息 |

工具预算、单步最多工具数和缓存时间由以下配置控制：

```yaml
tool_selection:
  max_tools_per_step: 3
  cost_budget_per_step: 6.5
  allow_cached_results: true
  tool_cache:
    camera_tool:
      ttl_steps: 1
    radar_tool:
      ttl_steps: 2
    lidar_tool:
      ttl_steps: 5
    fusion_tool:
      ttl_steps: 5
```

对比实验应复制独立 YAML 文件，并保持道路场景、随机种子、车辆初始状态和
终止条件一致。

## 输出与分析

场景运行后，原始 CSV 默认写入：

```text
opencda_output/agent_single_llm_tool_demo/agent_single_llm_tool_demo.csv
opencda_output/agent_single_llm_tool_demo_2/agent_single_llm_tool_demo_2.csv
```

手动分析 Demo 2：

```bash
python scripts/analyze_agent_tool_selection_csv.py \
  --csv opencda_output/agent_single_llm_tool_demo_2/agent_single_llm_tool_demo_2.csv \
  --out-json results/agent_single_llm_tool_demo_2/summary.json \
  --out-md results/agent_single_llm_tool_demo_2/summary.md

python scripts/plot_agent_tool_selection_results.py \
  --csv opencda_output/agent_single_llm_tool_demo_2/agent_single_llm_tool_demo_2.csv \
  --out-dir results/agent_single_llm_tool_demo_2
```

当前分析覆盖：

- LLM 调用成功率、后端分布、回退率、重试和请求延迟；
- 风险等级、信息不确定性和资源预算等级分布；
- 各工具请求率、执行率、缓存率、融合调用率和代理成本；
- Agent 周期耗时、工具执行耗时、CPU、内存和可选 GPU 指标；
- 自车轨迹、前向距离估计、超车状态和重复超车完成情况。

`opencda_output/` 和 `results/` 是本地生成目录，不纳入代码版本控制。

## 代码入口

```text
opencda/customize/core/llm_agent/       Agent、提示构建、响应解析与安全约束
opencda/customize/core/tools/           感知工具及结果级融合
opencda/customize/core/plan/            Agent 决策与超车行为状态机
opencda/customize/core/common/          实验记录与资源监控
opencda/scenario_testing/               Demo 1 与 Demo 2 场景
scripts/                                CSV 统计与图表生成
```

## 基础项目与许可

本项目基于 [OpenCDA](https://github.com/ucla-mobility/OpenCDA) 进行研究开发。
使用前请阅读仓库中的 `LICENSE`。OpenCDA 的论文引用信息和完整文档见其
[官方项目页面](https://opencda-documentation.readthedocs.io/)。
