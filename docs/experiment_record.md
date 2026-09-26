# Qwen3-4B-Base 后训练 SFT 实验记录

记录日期：2026-09-25 至 2026-09-26（公开版；历史检查快照保留，下方最终结果为准）  
依据：课程 PDF、实际执行命令、用户提供的终端日志，以及通过已登录 Jupyter 直接检查的运行结果。本文不将计划配置或试评测分数视为完整实验结论。

## 1. 当前完成情况

| 项目 | 状态 |
| --- | --- |
| Python、GPU 与 LlamaFactory 环境准备 | 已完成并验证 |
| Qwen3-4B-Base 下载、加载和 GPU 生成测试 | 已完成 |
| TULU-3 下载与全分片读取检查 | 已完成 |
| 格式清理、3k/10k/30k 随机子集生成 | 已完成 |
| 10 个优化步的 LoRA SFT 试运行 | 已完成 |
| 3k 数据、3 epoch 的正式 LoRA SFT | 已完成 |
| lm-evaluation-harness 安装 | 已完成，依赖检查通过 |
| SFT 模型 GSM8K 五题试评测 | 已完成；修正生成上限后复测成功 |
| Base 模型五题对照评测 | 已完成，宽松提取与严格匹配均为 0/5 |
| 完整 GSM8K、MMLU、IFEval 评测 | Base 与三组 SFT 共 12 项均已完成 |
| 10k × 3 epoch 与 30k × 1 epoch 训练 | 已完成，模型和评测结果已保存 |
| 高质量 10k 数据筛选挑战项目 | 尚未开始 |

## 2. 课程要求与本次实现

课程文件：`【实验课】后训练SFT实验-2026.9.20.pdf`。

基础实验使用 Qwen3-4B-Base、LlamaFactory 和 TULU-3 SFT 数据子集，使用 3,000 条数据训练 3 个 epoch，并通过 lm-evaluation-harness 在 MMLU、IFEval、GSM8K 上评测。进阶实验比较 10,000 条数据训练 3 个 epoch 与 30,000 条数据训练 1 个 epoch；挑战项目研究高质量数据筛选。

本次采用 LoRA 参数高效微调。课程 PDF 未明确要求全参数微调；LoRA 是结合单张 24GB 显卡所选的实现方式，不应在报告中写成全参数微调。

## 3. 计算环境

| 项目 | 实际记录 |
| --- | --- |
| 运行位置 | 租用的 Linux GPU 实例，通过网页 Jupyter Terminal 操作 |
| GPU | NVIDIA GeForce RTX 4090 |
| GPU 总显存 | 24,564 MiB |
| NVIDIA 驱动 | 550.54.14，初始检查记录 |
| nvidia-smi 显示的 CUDA 版本 | 12.4，表示驱动支持能力，不代表另装了 CUDA Toolkit |
| Conda 环境 | `sft` |
| 实验 Python | 3.12；实际路径 `python`，补丁版本未单独记录 |
| PyTorch | `2.6.0+cu124` |
| NumPy | `1.26.4` |
| Transformers | `4.52.4` |
| LlamaFactory | `0.9.3` |
| datasets | `3.6.0` |
| peft | `0.15.2` |
| accelerate | `1.7.0` |
| tokenizers | `0.21.1`，评测安装前版本约束文件的终端输出，复核后更正 |
| huggingface-hub | `0.36.2` |
| lm_eval | `0.4.8`，安装包含 IFEval 附加依赖 |
| 工作目录 | `/workspace` |
| 磁盘 | 容量约 50GB；2026-09-25 评测准备时显示已用约 11GB、可用约 40GB |

`torch.cuda.is_available()` 返回 True，GPU 矩阵乘法测试成功。安装评测工具后，`python -m pip check` 返回 `No broken requirements found.`。

初始 Jupyter 默认 Python 为 3.10.15，路径为 `/base/mambaforge/bin/python`，当时未安装 PyTorch。之后创建独立 `sft` 环境，并注册 `Python (SFT)` 内核。

## 4. 环境配置中遇到的问题

### 4.1 mamba 启动失败

错误为无法从 `conda.cli.main` 导入 `generate_parser`。改用 Conda 的 classic 求解器完成环境创建：

```bash
CONDA_NO_PLUGINS=true /base/mambaforge/bin/conda create -n sft python=3.12 pip -y --solver classic
source /base/mambaforge/etc/profile.d/conda.sh
conda activate sft
```

### 4.2 平台 PyTorch 约束冲突

平台通过 `PIP_CONSTRAINT` 引入 `torch==2.7.0a0+7c8ec84dab.nv25.3` 的约束，与指定的 2.6.0 冲突。对安装命令临时移除此变量后成功：

```bash
env -u PIP_CONSTRAINT python -m pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

未通过修改平台全局约束文件解决问题。root 用户警告未造成安装失败。

### 4.3 下载网络问题

服务器直连 `huggingface.co` 超时。模型改由魔搭的 `Qwen/Qwen3-4B-Base` 仓库下载；TULU-3 和试评测数据通过第三方 `hf-mirror.com` 镜像访问。评测工具最初从默认 PyPI 源下载较慢，停止该下载进程后，改用清华 PyPI 镜像安装成功。

### 4.4 远程操作方式

曾尝试本地 SSH 密钥接入，但平台 SSHPiper 网关只提供密码认证，密钥登录未成功。之后在 Codex 内置浏览器中由用户登录 Jupyter，成功直接操作网页终端。本文不包含密码、token 或私钥。

## 5. 基础模型

模型目录：`/workspace/models/Qwen3-4B-Base`。

检查结果：

- 配置类型为 `qwen3`。
- 3 个 safetensors 权重分片存在且非空。
- 配置与分词器可离线读取。
- 使用 BF16、SDPA 在 GPU 上成功加载，参数量显示为 4.02B。
- 加载后 PyTorch 已分配显存为 7.49 GiB；这不是训练峰值显存。
- 输入 `The capital of France is`，生成 `The capital of France is Paris. This is a true statement.`。

模型通过魔搭下载，实际下载的仓库提交哈希未在本记录中取得。不能将此前失败的 Hugging Face 下载计划当作模型版本已锁定的证据。

## 6. TULU-3 数据处理

原始数据集：`allenai/tulu-3-sft-mixture`。原始文件目录：

```text
/workspace/datasets/tulu-3-sft-mixture/data
```

所有 Parquet 文件均进行了逐批完整读取，并核对读取条数与各文件元数据一致：

| 分片 | 大小（MiB，日志值） | 条数 |
| --- | ---: | ---: |
| train-00000-of-00006.parquet | 344.3 | 156,558 |
| train-00001-of-00006.parquet | 454.9 | 156,557 |
| train-00002-of-00006.parquet | 140.1 | 156,557 |
| train-00003-of-00006.parquet | 154.6 | 156,557 |
| train-00004-of-00006.parquet | 142.6 | 156,557 |
| train-00005-of-00006.parquet | 110.9 | 156,557 |
| 合计 | — | **939,343** |

最初错误地根据数据卡说明文字断言应有 939,344 条，导致检查报错。随后核对官方页面，实际 `Number of rows` 为 939,343，与本地读取结果一致。因此不是缺失分片，无需补样本或重下数据。

### 6.1 格式过滤

初次直接抽样因样本 937 缺少有效 assistant 回答而停止，尚未写出子集。随后改为先检查全部样本，再从合格记录中随机抽样。

过滤条件：消息为非空列表、内容为字符串；允许开头一条 system 消息；之后必须由非空 user/assistant 消息成对交替构成，且结束于 assistant。没有按答案正确性、难度或模型评分筛选。

| 排除原因 | 条数 |
| --- | ---: |
| empty_user | 40 |
| incomplete_dialogue | 15 |
| empty_assistant | 716 |
| 合计 | **771** |
| 合格记录 | **938,572** |

排除比例约为 0.082%。每条记录按脚本首先发现的原因计数，各原因并非独立统计。

### 6.2 随机抽样

使用 Python `random.Random(42)`，通过蓄水池抽样从所有合格记录中无放回均匀抽取 30,000 条，再打乱抽中记录，取前 3,000、10,000、30,000 条保存为嵌套子集。各子集内部没有重复抽取同一原始行；这不等于对原始文本进行了内容去重。

输出目录：`/workspace/sft-experiment/data_validated`。

```text
tulu3_random_3000.json
tulu3_random_10000.json
tulu3_random_30000.json
sampling_manifest.json
dataset_info.json
```

保留原始 messages、id 和 source 等字段。`sampling_manifest.json` 记录随机种子、过滤统计、原始行索引及下载版本信息；实际哈希值需从服务器文件读取，本文未抄录。

## 7. 训练配置

正式配置路径：`/workspace/sft-experiment/configs/sft_3k_3ep.yaml`。

| 参数 | 值 |
| --- | --- |
| stage / do_train | sft / true |
| finetuning_type | lora |
| lora_rank / lora_alpha / lora_dropout | 16 / 32 / 0.05 |
| lora_target | all |
| dataset | tulu3_random_3000 |
| 数据映射 | sharegpt，messages → messages，role/content 字段 |
| template | qwen |
| cutoff_len | 2048 |
| train_on_prompt | false，只对回答部分计算损失 |
| mask_history | false，多轮历史回答也参与训练 |
| packing | false |
| 单卡 micro batch size | 1 |
| gradient_accumulation_steps | 16 |
| learning_rate | 0.0001 |
| num_train_epochs | 3.0 |
| lr_scheduler_type / warmup_ratio | cosine / 0.03 |
| 精度 | BF16 |
| gradient_checkpointing | true |
| flash_attn | sdpa |
| optim | adamw_torch |
| seed / data_seed | 42 / 42 |
| preprocessing_num_workers / dataloader_num_workers | 2 / 0 |
| logging_steps | 5 |
| save_strategy / save_total_limit | epoch / 2 |
| report_to | none |
| 验证集 | 本轮未配置 |

`qwen` 模板使用 ChatML 格式，对没有系统提示的记录默认加入 Qwen 系统提示。它不是 `qwen3` reasoning 模板。后续比较应明确记录并保持提示模板策略一致。

长对话会受到 2048 token 的长度限制。尚未统计实际截断比例、有效监督 token 数以及训练峰值显存；不能声称所有完整对话均参与了训练。

## 8. 训练结果

### 8.1 十步试运行

试运行使用最多 64 条样本，10 个优化步，梯度累积 16，输出单独保存。日志结果：

| 指标 | 记录值 |
| --- | ---: |
| epoch | 2.5 |
| train_loss | 0.556 |
| train_runtime | 00:00:55.45 |
| train_samples_per_second | 2.885 |
| train_steps_per_second | 0.18 |

目录：`/workspace/sft-experiment/outputs/smoke_10steps`。成功保存 adapter 配置、约 127M 的权重、分词器和 loss 图片。

### 8.2 正式 3k × 3 epoch 训练

从原始 Base 模型开始训练，未接着试运行 adapter 训练。使用 nohup 在后台启动，本地模型和数据读取时启用离线模式。

完成时间：日志显示为 2026-09-24 12:01:05（服务器日志时间）。

| 指标 | 记录值 |
| --- | ---: |
| epoch | **3.0** |
| train_loss | **0.6523** |
| train_runtime | **00:50:08.23** |
| train_samples_per_second | 2.992 |
| train_steps_per_second | 0.187 |
| total_flos | 123182164 GF（原日志显示） |

`train_loss` 是训练过程汇总指标，不是验证集 loss，也不是最后一步 loss。试运行和正式训练的数据范围、训练调度不同，不宜直接将两者的平均 loss 作效果比较。

正式输出目录：`/workspace/sft-experiment/outputs/sft_3k_3ep`。

已确认的文件和信息：

- `adapter_config.json`：871 字节。
- `adapter_model.safetensors`：132,187,888 字节（终端 ls 显示约 127M）。
- adapter 配置的基础模型路径正确，LoRA rank 为 16。
- 分词器配置、特殊 token 配置和 `chat_template.jinja` 已保存。
- `training_loss.png` 已生成，但本文未对图片曲线形态作视觉分析。

完整日志：`/workspace/sft-experiment/logs/sft_3k_3ep.log`。

关于警告：未配置验证集，因此 `No metric eval_loss/eval_accuracy to plot` 不表示训练失败。模型卡部分结果缺少字段的提示也不等于权重保存失败。终端显示后台任务 `Done`；随后 Ctrl+C 退出的是 tail 日志查看。

## 9. 评测工具安装与试评测

2026-09-25 通过已登录 Jupyter 的网页终端直接完成。

安装前保存 `environment_before_eval.txt`，并将 torch、transformers、numpy、datasets、peft、accelerate、tokenizers、huggingface-hub 的已装版本写入 `constraints.txt`。安装 `lm_eval[ifeval]==0.4.8` 后依赖检查通过。

### 9.1 已执行的评测设置

| 设置 | 值 |
| --- | --- |
| 评测模型 | 原始 Base + 正式 sft_3k_3ep LoRA adapter |
| tokenizer | 正式训练输出目录中的 tokenizer |
| 任务 | gsm8k |
| num_fewshot | 5 个解题示例 |
| limit | 5 道测试题，仅用于验证流程 |
| apply_chat_template | 开启 |
| batch_size / device | 1 / cuda:0 |
| dtype | bfloat16 |
| max_length | 4096 |
| do_sample | false |
| log_samples | 开启 |

正式全量评测的生成预算和提示策略尚未最终确定，不能将上述试运行配置称为完整课程评测方案。

### 9.2 首次运行发现的生成长度覆盖

首次只指定 `max_gen_toks=512`，但本地模型 `generation_config.json` 含有 `max_new_tokens=2048`。Transformers 日志明确提示后者优先，实际生成上限因此为 2048 token。

首次五题生成阶段约 8 分 6 秒。结果保留在：

```text
/workspace/sft-experiment/evaluation/smoke_sft_gsm8k
```

之后同时显式设置 `max_gen_toks=512,max_new_tokens=512,do_sample=False`，未修改模型权重或评分规则；复测写入新目录，保留初次记录。

### 9.3 修正后复测结果

复测日志确认 `max_new_tokens=512` 生效，5/5 题生成完成，生成阶段约 2 分 3 秒。退出状态文件内容为 **0**，结果和逐题 JSONL 保存成功。

| GSM8K 指标 | 结果 | 日志标准误 |
| --- | ---: | ---: |
| exact_match / flexible-extract | 0.6000（3/5） | 0.2449 |
| exact_match / strict-match | 0.0000（0/5） | 0.0000 |

首次与修正后两次试评测显示相同的上述分数。五题样本很小，0/5 对应的日志标准误 0 不代表真实能力已被精确估计。

逐题检查发现：回答未按严格提取所需的 `#### 数字` 格式输出；部分回答在作答后持续重复文本。宽松提取可从回答中提取数字，因此与严格格式分数不同。首题还出现了错误计算：把 16 − 3 − 4 写成 13，给出 26 美元；该题参考结果是 18 美元。因此不仅有格式问题，也有实际答题错误。

现有证据不足以判断重复文本的根因，也不足以判断 SFT 相对 Base 的变化。不能把“宽松提取 60%”写作全量 GSM8K 准确率或微调带来的提升。

日志出现 `not a git repository` 提示，但之后成功保存结果且退出码为 0，不能仅凭该提示判定评测失败。

## 10. 服务器文件索引

### 2026-09-25 对照评测进展补记

补跑 Base 的同五题试评测：使用与 SFT 相同的正式训练 tokenizer、聊天模板、5-shot、512 token 上限，仅不加载 LoRA。Base 的 flexible-extract 与 strict-match 均为 0/5；对应 SFT 为 3/5 与 0/5。该对照仅描述这五题在统一聊天模板下的表现，不能外推全量能力。

随后启动正式全量队列，脚本为 `evaluation/run_formal_comparison.py`，输出根目录为 `evaluation/formal_chat_v1`。顺序为 GSM8K Base、GSM8K SFT、MMLU Base、MMLU SFT、IFEval Base、IFEval SFT。使用 batch size 4、max_length 4096、BF16、统一聊天模板、随机种子串 `0,1234,1234,1234`。GSM8K 为 5-shot、512 token 生成上限；MMLU 为 5-shot、选择题似然评分；IFEval 为 0-shot、1280 token 上限。没有设置 `--limit`。

队列使用独立输出目录、逐项日志、`status.json` 和安装后的 `environment.txt`，遇到非零退出码即停止。已观察首项 GSM8K Base 正常推进，任务总量为 1,319 题，初次检查进度为 125/1,319，GPU 利用率 100%、显存占用 10,577 MiB。此进度是当时快照，不是最终结果。SFT 阶段可能因生成更长而明显更慢。

IFEval 的 NLTK 资源下载也已启动；本补记时尚未确认资源下载成功，不能记作已验证。正式任务的后续数据加载和评分仍可能出现需要处理的问题。

以下均为远程服务器路径，而非本地 Windows 文件路径。

| 用途 | 路径 |
| --- | --- |
| LlamaFactory 源码 | `/workspace/LlamaFactory` |
| Base 模型 | `/workspace/models/Qwen3-4B-Base` |
| 原始 TULU-3 | `/workspace/datasets/tulu-3-sft-mixture` |
| 过滤抽样后的数据与清单 | `/workspace/sft-experiment/data_validated` |
| 训练配置 | `/workspace/sft-experiment/configs` |
| 训练日志 | `/workspace/sft-experiment/logs` |
| 正式 LoRA 输出 | `/workspace/sft-experiment/outputs/sft_3k_3ep` |
| 评测工作目录 | `/workspace/sft-experiment/evaluation` |
| 原下载源安装日志 | `evaluation/install.log`（相对实验根目录；下载被停止） |
| 镜像安装日志 | `evaluation/install_mirror.log` |
| 首次试评测脚本 / 日志 | `evaluation/run_smoke.sh` / `evaluation/smoke.log` |
| 修正后试评测脚本 / 日志 | `evaluation/run_smoke_512.sh` / `evaluation/smoke_512.log` |
| 修正后退出状态 | `evaluation/smoke_512.exit` |
| 修正后结果与逐题输出 | `evaluation/smoke_sft_gsm8k_512/` |

## 11. 待完成与报告边界

1. 保存安装评测工具后的完整环境版本，补录模型、数据和评测文件的实际版本或哈希。
2. 在相同评测设置下比较 Base 与 SFT；明确聊天模板、few-shot、生成上限、停止条件和提取规则。正式评测前检查是否有上下文截断。
3. 完成全量 GSM8K、MMLU、IFEval。截至下方最新检查，前五项已成功，最后一项 IFEval SFT 仍在运行；NLTK punkt_tab 和 words 已确认下载成功。
4. 分析 loss 曲线、重复输出及错误样本；目前只有日志和有限样本观察，未完成系统诊断。
5. 完成 10k × 3 epoch 和 30k × 1 epoch 对照训练。两组应分别从同一个 Base 开始，保持其他训练配置一致；样本呈现次数相同不保证 token 数相同，应统计长度限制后的实际训练 token 和监督 token。
6. 另行开展挑战任务的数据质量筛选。本次空消息与格式过滤不能代替该研究任务。

**阶段结论：基础 SFT 训练已完成，全量对照评测已有部分结果；完整评测与进阶实验结论仍待完成。统一聊天模板下的分数不能直接解释为无提示格式影响的能力提升。**

### 最新检查：全量评测与进阶训练队列（2026-09-25）

从正式结果 JSON 与状态文件核对，六项任务中五项退出码为 0；IFEval SFT 仍在运行。检查时进度为 341/541，约 63%，日志估计剩余约 53 分钟。此为检查快照，不是完成时间承诺。

| 指标（全量，统一聊天模板） | Base | SFT 3k × 3 epoch |
| --- | ---: | ---: |
| GSM8K strict-match，1,319 题 | 0.6823% | 40.7885% |
| GSM8K flexible-extract，1,319 题 | 0.7582% | 71.9484% |
| MMLU acc | 24.6689% | 71.9342% |
| IFEval prompt strict，541 题 | 38.0776% | 待完成 |
| IFEval instruction strict | 50.4796% | 待完成 |
| IFEval prompt loose | 42.8835% | 待完成 |
| IFEval instruction loose | 54.4365% | 待完成 |

Base 与 SFT 使用相同聊天模板。Base 的低分需要结合原始输出和提示格式适配情况进一步核查；当前结果只支持该评测协议下的比较。GSM8K 严格与宽松提取指标分开报告。

已生成并检查 `configs/sft_10k_3ep.yaml` 和 `configs/sft_30k_1ep.yaml`；对应数据文件分别为 10,000 和 30,000 条。两组均从原始 Base 初始化 LoRA，沿用 3k 训练配置，仅改变数据集、epoch 和独立输出目录。

已启动 `run_advanced_training.py`，后台 PID 为 26323。状态文件 `logs/advanced_training_status.json` 已验证为 `waiting_for_formal_evaluation`，`runs` 为空，表示尚未开始进阶训练。它在六项评测全部成功后依次运行 10k × 3 epoch、30k × 1 epoch；评测失败或训练非零退出时停止，检测到已有输出时停止，文件锁避免重复启动。驱动日志为 `logs/advanced_driver.log`，每组训练另存同名日志。最终评测尚未排入此训练脚本。

按 3k × 3 epoch 的 50 分钟粗略线性估算，两组训练合计约 5–6 小时，另加当前评测剩余时间。实际时长受样本长度等影响；这不是已测得时长。两组均呈现约 30,000 条次，但实际有效训练 token 和监督 token 尚未统计，不能声称计算预算完全相等。

### 2026-09-26 检查与下一阶段

正式 Base/3k SFT 六项评测已全部完成。3k SFT 的 IFEval 最终结果为 prompt strict 36.4140%、instruction strict 50.4796%、prompt loose 39.0018%、instruction loose 53.4772%。与 Base 相比，严格指令级分数相同，其他三个指标较低；不能概括为所有任务均提升。此前表中的“待完成”为历史快照，以本段为准。

两组进阶训练的退出码均为 0，最终 adapter 文件均存在，状态为 `training_complete_evaluation_pending`：

| 训练 | epoch | train_loss | train_runtime（秒） |
| --- | ---: | ---: | ---: |
| 10k × 3 | 3.0 | 0.653388 | 10251.8881 |
| 30k × 1 | 1.0 | 0.675747 | 9965.4558 |

训练 loss 来自不同样本分布，不能单凭它判断哪组模型更好。

已启动 `evaluation/run_advanced_eval.py`，后台 PID 29358。输出目录为 `evaluation/advanced_chat_v1`，总日志为 `evaluation/advanced_eval_driver.log`。队列按 GSM8K、MMLU、IFEval 顺序分别评测两组模型，共六项。命令复用前一轮对应 SFT 评测，只替换 LoRA 路径和输出目录；继续使用同一 tokenizer、聊天模板、种子、batch size、few-shot 与生成上限。新目录防止覆盖，文件锁防止重复运行，非零退出码会停止队列。

检查时首项 `gsm8k_sft_10k_3ep` 已启动，CUDA 模型三个权重分片加载完成；尚无进阶评测分数，不能记录为评测已完成。

### 全量进阶评测与提取规则审计（2026-09-26）

六项进阶评测均完成，退出码全部为 0。

| 指标（百分比） | 10k × 3 | 30k × 1 |
| --- | ---: | ---: |
| GSM8K strict | 54.132 | 76.118 |
| GSM8K flexible | 64.215 | 62.699 |
| MMLU acc | 72.753 | 72.262 |
| IFEval prompt strict | 42.144 | 42.699 |
| IFEval instruction strict | 54.436 | 56.235 |
| IFEval prompt loose | 46.396 | 47.689 |
| IFEval instruction loose | 58.513 | 60.192 |

核查本机 lm_eval 的 GSM8K YAML：strict-match 提取 `####` 后的数字，flexible-extract 用 `group_select: -1` 提取最后一个匹配数字。因此二者不构成包含关系，宽松规则分数不保证更高。

按 doc_id 配对全部 1,319 题的两条评分记录，得到：

| 模型 | 两者正确 | 仅严格正确 | 仅宽松正确 | 两者错误 |
| --- | ---: | ---: | ---: | ---: |
| Base | 9 | 0 | 1 | 1309 |
| 3k × 3 | 527 | 11 | 422 | 359 |
| 10k × 3 | 616 | 98 | 231 | 374 |
| 30k × 1 | 793 | 211 | 34 | 281 |

30k 组实例中，参考答案为 230，严格提取为 230，宽松提取为 80。原始回答在给出 `#### 230` 后反复继续叙述计算过程，尾部最后出现 80。该实例证明重复续写会使最后数字规则取错；尚未逐一人工归因全部不一致样本，不能断言所有差异都源于重复续写。

结论：本次单次实验中，10k × 3 的 MMLU 高约 0.49 个百分点，30k × 1 的 IFEval 四项指标均较高；GSM8K 排序依赖提取规则，必须并列报告。尚未进行多随机种子或显著性检验，也未对有效监督 token 预算做严格匹配，不应宣称某策略普遍更优。

服务器 `reports/metrics.json`、`reports/results_summary.md` 和 `reports/gsm8k_extraction_audit.json` 已生成；审计 JSON 包含配对统计及每组至多 12 个差异实例。归档脚本为 `evaluation/audit_and_package.py`，打包最终三组 adapter、配置、抽样数据、日志及评测证据，不包含 Base 权重与中间 checkpoint。服务器内打包不等于异地备份。

## 12. 参考资料

- 本地课程 PDF：`【实验课】后训练SFT实验-2026.9.20.pdf`。
- [Qwen3-4B-Base 官方模型页](https://huggingface.co/Qwen/Qwen3-4B-Base)
- [魔搭 Qwen3-4B-Base 下载入口](https://modelscope.cn/models/Qwen/Qwen3-4B-Base)
- [TULU-3 SFT 官方数据集](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)
- [LlamaFactory v0.9.3](https://github.com/hiyouga/LLaMA-Factory/tree/v0.9.3)
- [LlamaFactory 数据格式说明](https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.3/data/README.md)
- [lm-evaluation-harness v0.4.8](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.8)
- [本次 GSM8K 任务定义](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.8/lm_eval/tasks/gsm8k/gsm8k.yaml)

