# Qwen3-4B-Base LoRA SFT 实验

在单张 RTX 4090 24GB 上，使用 LlamaFactory 0.9.3 和 TULU-3 数据完成三组 LoRA SFT，并以 lm-evaluation-harness 0.4.8 评测 Base 与微调模型。实验日期：2026-09-24 至 2026-09-26。

## 实际结果

分数单位为 %。同一聊天模板、同一 tokenizer、同一评测参数；这些是本实验协议下的结果，不是官方模型榜单成绩。

| 指标 | Base | 3k × 3 | 10k × 3 | 30k × 1 |
|---|---:|---:|---:|---:|
| GSM8K strict | 0.68 | 40.79 | 54.13 | 76.12 |
| GSM8K flexible | 0.76 | 71.95 | 64.22 | 62.70 |
| MMLU acc | 24.67 | 71.93 | 72.75 | 72.26 |
| IFEval prompt strict | 38.08 | 36.41 | 42.14 | 42.70 |
| IFEval instruction strict | 50.48 | 50.48 | 54.44 | 56.24 |
| IFEval prompt loose | 42.88 | 39.00 | 46.40 | 47.69 |
| IFEval instruction loose | 54.44 | 53.48 | 58.51 | 60.19 |

**结果解读：**10k × 3 的 MMLU 略高，30k × 1 的 IFEval 四项指标较高。GSM8K 的模型排序依赖答案提取规则，不能只选一个指标宣称全面提升。实验仅运行单个训练随机种子，未作显著性检验；相同样本呈现次数不代表相同 token 预算。Base 使用聊天模板可能影响其表现。

## GSM8K 提取规则审计

strict 提取 `####` 后的数字，flexible 提取最后一个匹配数字，因此后者分数不保证更高。30k 模型的 1,319 题中，793 题两者正确，211 题仅 strict 正确，34 题仅 flexible 正确，281 题两者错误。检查实例发现正确答案后重复续写会干扰最后数字的提取。没有对全部差异题做人工归因。

完整指标见 [metrics.json](reports/metrics.json)，配对统计见 [extraction_counts.json](reports/extraction_counts.json)，过程与局限见 [实验记录](docs/experiment_record.md)。

## 目录

- `configs/`：实际训练配置，仅将机器路径改为仓库相对路径。
- `data/`：原始抽样索引清单与 LlamaFactory 数据注册表，不含训练文本。
- `scripts/`：整理后的数据重建、训练和评测入口。
- `historical/`：运行时脚本的路径匿名化副本，供审计，不作为通用入口。
- `training/`：三组实际训练指标、训练状态与 loss 图。
- `reports/`：原始精度的汇总指标和答案提取审计计数。

## 复现

建议 Linux + Python 3.12，GPU 显存 24GB。先创建独立环境，然后：

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
git clone --branch v0.9.3 --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
python -m pip install -c requirements-constraints.txt -e './LLaMA-Factory'
python -m pip install -c requirements-constraints.txt 'lm_eval[ifeval]==0.4.8' pyarrow pyyaml
python -m pip check
python -m nltk.downloader punkt_tab words
```

通过模型官方渠道准备 `models/Qwen3-4B-Base/`。根据 TULU-3 数据集的许可和访问要求，从清单中指定 revision 下载六个训练 parquet，放入 `datasets/tulu-3-sft-mixture/`（允许子目录）。

```bash
python scripts/restore_subsets.py --source datasets/tulu-3-sft-mixture
python scripts/train.py configs/sft_3k_3ep.yaml
python scripts/train.py configs/sft_10k_3ep.yaml
python scripts/train.py configs/sft_30k_1ep.yaml
python scripts/evaluate.py --base models/Qwen3-4B-Base --tokenizer outputs/sft_3k_3ep --output evaluation/reproduction
```

评测串行运行 12 项，无 `--limit`；GSM8K / MMLU 为 5-shot，IFEval 为 0-shot，batch size 4、BF16、max_length 4096，生成上限分别为 512 / 1280。`--tokenizer` 指向 3k 训练保存的 tokenizer，以复用同一模板。新入口拒绝覆盖已有输出。

整理后的入口仅完成静态语法及数据一致性检查，未重新执行昂贵的完整训练。数据重建依据保存的原始行索引；必须使用相同数据 revision 与分片排序，不能保证未来上游模型或评测数据变化后的结果逐位一致。

## 训练设置与数据

LoRA rank 16、alpha 32、dropout 0.05、target all；cutoff 2048；batch 1、梯度累积 16；学习率 1e-4，cosine，warmup 0.03；BF16、SDPA、梯度检查点；seed 42。完整参数以 YAML 为准。

原始数据实读 939,343 条；格式与空消息过滤后 938,572 条，排除 771 条。随机抽样 30k 后打乱，3k / 10k / 30k 是嵌套子集。此操作不是质量排序，也未完成高质量 10k 筛选挑战任务。

训练耗时：3k × 3 约 50 分钟，10k × 3 约 2 小时 51 分钟，30k × 1 约 2 小时 46 分钟。

## 发布范围

不包含模型权重、原始/抽样训练文本、完整逐题回答、账号信息、课程 PDF/PPT 或 490MB 服务器归档。模型与数据应按各自上游许可获取；本仓库不重新授予它们的权利。尚未为本项目单独指定开源许可证。

- [Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base)
- [TULU-3 SFT mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)
- [LlamaFactory v0.9.3](https://github.com/hiyouga/LLaMA-Factory/tree/v0.9.3)
- [lm-evaluation-harness v0.4.8](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.8)
