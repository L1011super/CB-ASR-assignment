# Contextual Biasing ASR for Tianjin Metro Station Recognition

本项目实现天津地铁自助售票机场景下的固定句式语音站名识别。输入是一段语音，例如“我要去营口道站”，系统输出标准 JSON 站名：

```json
{"station": "营口道"}
```

如果语音中的目的地不是天津地铁标准站名，则输出：

```json
{"station": null}
```

项目包含站名表处理、Kokoro TTS 数据生成、Whisper baseline、Speech-to-Qwen 两阶段训练、hotword contextual biasing、LoRA 微调、最终模型推理和实验结果可视化。

## 1. 项目结构

```text
contextual-biasing-asr/
├── baselines/                     # Whisper raw / exact / fuzzy / pinyin / lexicon baseline
├── configs/                       # TTS、baseline、stage1、stage2 训练配置
├── data/
│   ├── stations/                  # 天津地铁标准站名表
│   ├── manifests/                 # train/valid/test JSONL manifest
│   ├── raw/                       # 原始 TTS 音频，本地保留，不进 Git
│   ├── processed/                 # 处理后 16kHz mono 音频，本地保留，不进 Git
│   └── external/                  # 外部数据，本地保留，不进 Git
├── outputs/
│   ├── baseline/                  # baseline 预测和统计结果
│   ├── preds/                     # stage2 模型测试集预测结果
│   ├── report_figures/            # 报告图表和指标表
│   ├── stage1_align_attention_q32/ # 第一阶段 alignment checkpoint
│   └── stage2_bias_*              # 第二阶段 hotword/LoRA checkpoint
├── pretrained/                    # Whisper、Qwen、Kokoro 等大模型本体，本地保留，不进 Git
├── real_people_test/              # 真人 wav 单文件推理入口
│   ├── inpt/                      # 放入 xxx.wav
│   └── real_people_test.py
├── scripts/                       # 数据生成、检查、训练、推理、可视化脚本
├── src/
│   ├── data/                      # 数据集、collator、prompt、station lexicon
│   ├── models/                    # Whisper encoder、Speech Adapter、Audio-Qwen 模型
│   ├── training/                  # stage1/stage2 训练逻辑
│   └── tts/                       # Kokoro TTS 和音频后处理
├── EXPERIMENT_REPORT_SUMMARY.md   # 实验报告总结草稿
└── README.md
```

## 2. 环境配置

推荐使用 Windows + conda，当前项目使用的环境名为 `CB-ASR`。

```powershell
conda create -n CB-ASR python=3.11 -y
conda activate CB-ASR
```

安装主要依赖：

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate peft librosa soundfile tqdm pyyaml rapidfuzz pypinyin pandas matplotlib
pip install kokoro jieba
```

如果本机 CUDA/PyTorch 版本不同，请按显卡驱动选择对应的 PyTorch 安装命令。训练和最终推理默认使用：

```text
device: cuda:0
fp16: true
```

## 3. 大模型下载与目录约定

项目默认不在线下载模型，所有模型均从本地目录加载。请确保以下目录存在：

```text
pretrained/whisper-small
pretrained/Qwen2.5-1.5B-Instruct
pretrained/Kokoro-82M-v1.1-zh
```

示例下载命令如下。

### 3.1 Whisper-small

```powershell
huggingface-cli download openai/whisper-small --local-dir pretrained/whisper-small
```

### 3.2 Qwen2.5-1.5B-Instruct

```powershell
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir pretrained/Qwen2.5-1.5B-Instruct
```

### 3.3 Kokoro 中文 TTS

```powershell
huggingface-cli download --resume-download hexgrad/Kokoro-82M-v1.1-zh --local-dir pretrained/Kokoro-82M-v1.1-zh
```

注意：`pretrained/` 是大模型本体目录，不进入 Git 跟踪。训练得到的轻量 adapter、LoRA checkpoint 位于 `outputs/`，可以进入 Git 跟踪。

## 4. 数据准备

标准站名表：

```text
data/stations/tianjin_metro_stations.csv
```

字段包括：

```text
station_id,name,line_ids,pinyin,pinyin_compact,confusions,is_rare
```

当前站名表统计：

```text
标准站名：241
生僻/易错站名：68
带 confusions 的站名：220
多线路站点：32
```

## 5. 生成 Kokoro TTS 数据集

配置文件：

```text
configs/kokoro_tts.yaml
```

小规模测试：

```powershell
python scripts/generate_kokoro_tts_dataset.py --config configs/kokoro_tts.yaml --limit_stations 2 --overwrite
python scripts/check_tts_dataset.py --config configs/kokoro_tts.yaml
```

完整生成：

```powershell
python scripts/generate_kokoro_tts_dataset.py --config configs/kokoro_tts.yaml --overwrite
python scripts/check_tts_dataset.py --config configs/kokoro_tts.yaml
```

生成的数据规模：

| split | 总数 | 站名样本 | NULL 样本 |
| --- | ---: | ---: | ---: |
| train | 2660 | 2410 | 250 |
| valid | 532 | 482 | 50 |
| test | 532 | 482 | 50 |
| 合计 | 3724 | 3374 | 350 |

音频输出目录：

```text
data/raw/kokoro_tts/
data/processed/audio/
```

manifest 输出：

```text
data/manifests/train_tts.jsonl
data/manifests/valid_tts.jsonl
data/manifests/test_tts.jsonl
```

## 6. 运行 Whisper Baseline

纯 Whisper 原始转写：

```powershell
python baselines/run_whisper_raw.py --config configs/whisper_raw.yaml
```

运行站名匹配 baseline：

```powershell
python baselines/run_whisper_station_baselines.py `
  --manifest data/manifests/test_tts.jsonl `
  --whisper_output outputs/baseline/whisper_raw_test_predictions.jsonl `
  --station_csv data/stations/tianjin_metro_stations.csv `
  --output_dir outputs/baseline/whisper_station_baselines
```

已得到的 baseline 结果：

| 方法 | Overall Acc | Station Acc | NULL Acc |
| --- | ---: | ---: | ---: |
| Whisper + exact match | 23.87% | 17.22% | 88.00% |
| Whisper + edit fuzzy | 31.58% | 24.69% | 98.00% |
| Whisper + pinyin fuzzy | 93.80% | 93.78% | 94.00% |
| Whisper + lexicon normalization | 93.80% | 93.78% | 94.00% |

## 7. 第一阶段训练：Speech-to-Qwen Alignment

第一阶段目标是让 Whisper Encoder 的语音特征通过 Speech Adapter / Projection 接入 Qwen。该阶段不加入 hotword list，不开启 LoRA，冻结 Whisper 和 Qwen，只训练 Adapter 与 Projection。

配置：

```text
configs/stage1_align_attention.yaml
```

测试 forward：

```powershell
python scripts/test_stage1_forward.py --config configs/stage1_align_attention.yaml --limit 2
```

训练：

```powershell
python scripts/train_stage1_align.py --config configs/stage1_align_attention.yaml
```

checkpoint：

```text
outputs/stage1_align_attention_q32/best.pt
```

## 8. 第二阶段训练：Hotword Contextual Biasing + LoRA

第二阶段加载 stage1 checkpoint，在 prompt 中加入天津地铁候选站名表，并开启 LoRA，使模型输出标准 JSON。

普通 hotword 训练：

```powershell
python scripts/test_stage2_forward.py --config configs/stage2_bias_attention.yaml --limit 2
python scripts/train_stage2_bias.py --config configs/stage2_bias_attention.yaml
```

最终 hard-refine 训练配置：

```text
configs/stage2_bias_attention_hard_refine.yaml
```

训练命令：

```powershell
python scripts/train_stage2_bias.py `
  --config configs/stage2_bias_attention_hard_refine.yaml `
  --resume_from outputs/stage2_bias_hotwords_attention_q32_nullboost8/best
```

最终 checkpoint：

```text
outputs/stage2_bias_hotwords_attention_q32_hard_refine/best
```

最终测试推理：

```powershell
python scripts/infer_manifest.py `
  --manifest data/manifests/test_tts.jsonl `
  --ckpt outputs/stage2_bias_hotwords_attention_q32_hard_refine/best `
  --output outputs/preds/stage2_hotwords_attention_q32_hard_refine_test_best.jsonl `
  --with_hotwords true
```

最终测试结果：

| 指标 | 正确数 / 总数 | Accuracy |
| --- | ---: | ---: |
| Overall | 524 / 532 | 98.50% |
| Station | 479 / 482 | 99.38% |
| NULL | 45 / 50 | 90.00% |
| Rare station | 133 / 136 | 97.79% |
| Confusion station | 437 / 440 | 99.32% |

## 9. 真人 wav 单文件推理

将真人录音放入：

```text
real_people_test/inpt/
```

例如：

```text
real_people_test/inpt/lsdtest.wav
```

运行：

```powershell
python real_people_test/real_people_test.py lsdtest.wav
```

如果需要查看模型原始 JSON 输出：

```powershell
python real_people_test/real_people_test.py lsdtest.wav --print_raw
```

也可以在 Python 中调用：

```python
from real_people_test.real_people_test import infer_wav

station = infer_wav("lsdtest.wav")
print(station)
```

## 10. 实验结果可视化

生成报告用柱状图和指标表：

```powershell
python scripts/plot_experiment_results.py
```

输出：

```text
outputs/report_figures/accuracy_bar_overall.png
outputs/report_figures/accuracy_bar_overall.pdf
outputs/report_figures/experiment_metrics_table.csv
outputs/report_figures/experiment_metrics_table.md
outputs/report_figures/experiment_metrics_summary.json
```

## 11. 调试与测试方式

### 11.1 站名表测试

```powershell
python scripts/test_station_lexicon.py
```

重点测试：

```text
营口到 -> 营口道
迎口道 -> 营口道
渌水道/澧水道不过度归一化
```

### 11.2 TTS 数据检查

```powershell
python scripts/check_tts_dataset.py --config configs/kokoro_tts.yaml
```

输出：

```text
outputs/data_checks/kokoro_tts_check_report.json
outputs/data_checks/kokoro_tts_bad_cases.csv
```

检查内容包括：manifest 是否存在、音频是否 16kHz mono、时长是否合理、是否全静音、每站数量是否正确、NULL 数量是否正确、target_station 是否合法。

### 11.3 Stage1 forward 测试

```powershell
python scripts/test_stage1_forward.py --config configs/stage1_align_attention.yaml --limit 2
```

### 11.4 Stage2 forward 测试

```powershell
python scripts/test_stage2_forward.py --config configs/stage2_bias_attention_hard_refine.yaml --limit 2
```

### 11.5 常见问题

1. **模型加载失败**  
   检查 `pretrained/whisper-small` 和 `pretrained/Qwen2.5-1.5B-Instruct` 是否完整。项目默认 `local_files_only=True`，不会自动联网下载。

2. **CUDA 不可用**  
   检查 `torch.cuda.is_available()`，或将配置中的 `device` 改为 `cpu`。CPU 可以跑通小测试，但 Qwen 推理会很慢。

3. **TTS 站名读音错误**  
   项目已将所有站名加入 jieba 词典，并对“团泊”做 TTS 输入侧旁路修正。

4. **NULL 被误判为站名**  
   这是最终模型的主要剩余问题。可以继续增加非法目的地 hard negatives，或加入单独的 NULL 判别策略。

5. **Git index.lock**  
   如果 Git 操作中断后出现 `.git/index.lock`，确认没有其他 git 进程后删除该锁文件再提交。

## 12. Git 跟踪约定

不跟踪：

```text
data/raw/
data/processed/
data/external/
pretrained/
ckpts/kokoro-v1.1/
```

可以跟踪：

```text
configs/
src/
scripts/
baselines/
data/stations/
data/manifests/
outputs/ 中的训练日志、预测结果和轻量 checkpoint
```

## 13. 当前最优结果文件

```text
outputs/stage2_bias_hotwords_attention_q32_hard_refine/best
outputs/preds/stage2_hotwords_attention_q32_hard_refine_test_best.jsonl
outputs/preds/stage2_hotwords_attention_q32_hard_refine_test_best_metrics.json
outputs/preds/stage2_hotwords_attention_q32_hard_refine_test_best_errors.csv
```

报告总结草稿：

```text
EXPERIMENT_REPORT_SUMMARY.md
```
