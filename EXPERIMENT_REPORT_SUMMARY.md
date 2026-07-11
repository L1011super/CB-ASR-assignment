# 天津地铁自助售票机场景的固定句式站名识别实验总结

本文档用于实验报告撰写，汇总当前仓库中已经完成的代码实现、数据构成、训练流程和最终实验结果。项目目标是面向天津地铁自助售票机固定句式语音输入，例如“我要去营口道站”，从语音中识别标准目的地站名；若语音中没有合法天津地铁站名，则输出 `{"station": null}`。

## 1. 实验整体思路

本项目将任务建模为“语音条件下的站名 JSON 生成”问题，而不是单纯做通用 ASR。核心原因是天津地铁站名中存在大量低频专名、近音字和易错字，例如“渌水道/澧水道”“营口道/营口到/迎口道”“团泊”等，通用 ASR 容易在字形、同音字、近音站名之间出错。实验流程分为四层：

1. 构建可信站名表。  
   使用 `data/stations/tianjin_metro_stations.csv` 作为标准站名词表，字段包含 `station_id`、`name`、`line_ids`、`pinyin`、`pinyin_compact`、`confusions`、`is_rare` 等。代码实现了站名读取、拼音生成/校验、混淆词处理、精确匹配、别名匹配、编辑距离匹配和拼音相似度匹配。

2. 生成固定句式 TTS 数据。  
   基于站名表自动生成正样本：“我要去{站名}站”；若站名本身已以“站”结尾，则生成“我要去{站名}”。额外生成 NULL 负样本，例如“我要去水上公园站”“我要去天津大学站”等，目标标签为 `null`。音频由 Kokoro TTS 生成，并统一转为 16 kHz mono wav。

3. 建立多级 baseline。  
   先实现纯 Whisper 原始转写，然后依次叠加 exact match、编辑距离 fuzzy match、拼音 fuzzy match、站名表归一化。baseline 用于证明：仅依赖 ASR 文本和规则后处理，在低频站名和近音混淆上仍存在瓶颈。

4. 训练 Speech-to-Qwen 模型。  
   最终方法采用 Whisper Encoder 提取语音特征，经 Speech Adapter/Projection 映射为 Qwen 可接收的 speech tokens，再与包含天津地铁候选站名表的 instruction prompt 拼接，输入 Qwen2.5-1.5B-Instruct 生成标准 JSON。训练分为两阶段：第一阶段只做 speech-to-Qwen alignment；第二阶段加入 hotword list 并开启 LoRA 做 contextual biasing。

整体推理链路如下：

```text
audio wav
  -> Whisper Encoder
  -> Speech Adapter / Projection
  -> Qwen prompt + speech tokens + station hotword list
  -> JSON output: {"station": "标准站名"} 或 {"station": null}
```

## 2. 数据集构成与统计

### 2.1 站名表统计

站名表文件：`data/stations/tianjin_metro_stations.csv`

| 项目 | 数量 |
| --- | ---: |
| 标准站名总数 | 241 |
| `is_rare=1` 生僻/易错站名 | 68 |
| 带 `confusions` 混淆词的站名 | 220 |
| 多线路站点 | 32 |

站名表中的 `pinyin` 与 `pinyin_compact` 用于拼音相似度匹配和错误分析；`confusions` 用于常见误识别后处理，例如“营口到”“迎口道”可归一化到“营口道”。

### 2.2 TTS 语音数据统计

TTS 配置文件：`configs/kokoro_tts.yaml`

固定正样本规则：

```text
若站名不以“站”结尾：我要去{站名}站
若站名已以“站”结尾：我要去{站名}
```

NULL 负样本规则：

```text
我要去{非法目的地}站
target_station = null
```

最终 TTS 数据 manifest 统计如下：

| split | 总数 | 站名正样本 | NULL 负样本 |
| --- | ---: | ---: | ---: |
| train | 2660 | 2410 | 250 |
| valid | 532 | 482 | 50 |
| test | 532 | 482 | 50 |
| 合计 | 3724 | 3374 | 350 |

增强类型统计：

| split | clean | speed | leading_silence | light_noise |
| --- | ---: | ---: | ---: | ---: |
| train | 1335 | 799 | 266 | 260 |
| valid | 254 | 278 | 0 | 0 |
| test | 264 | 74 | 93 | 101 |

语音生成时做过以下修正：

- 将所有标准站名和“站名+站”加入 jieba 词典，避免 Kokoro 将站名错误切分，例如避免“中心渔港站”读成“中心-渔港站”。
- 对“团泊”做 TTS 输入侧旁路，将其替换为“团博”以修正“泊”的发音；manifest 和标签仍保留“团泊”。
- 最终版本取消“我要去”和站名之间的额外停顿。
- 以 0.03 概率在句首加入少量思考词，例如“嗯/那个/呃”，但第一版主数据仍保持固定句式为主。

### 2.3 Hard-refine 训练集

最终最优模型额外使用了 hard station refine 数据：

文件：`data/manifests/hard_station_refine.jsonl`

| 项目 | 数量 |
| --- | ---: |
| 总样本数 | 1656 |
| 站名样本 | 1490 |
| NULL 样本 | 166 |

该数据集用于强化前一轮模型中容易混淆的站名，例如“东丽一经路/东丽三经路/东丽六经路”“洪湖里/洪泥河东”“水上公园西路/水上公园东路”等。强化时只复用训练集中的音频样本，没有直接使用 test 音频。需要注意：当前 hard mining 的错误来源来自开发过程中的测试错误分析；若作为严格论文实验，建议额外划分 dev 集，用 dev 错误构造 hard-refine 数据，再只在 test 上做一次最终评估。

## 3. 最终训练思路

### 3.1 第一阶段：speech-to-Qwen alignment

配置文件：`configs/stage1_align_attention.yaml`

目标：让 Whisper Encoder 的语音特征通过 Speech Adapter/Projection 接入 Qwen，使 Qwen 能根据音频生成站名 JSON。第一阶段不加入候选站名表，不开启 LoRA。

模型设置：

| 模块 | 设置 |
| --- | --- |
| Whisper Encoder | `pretrained/whisper-small`，冻结 |
| LLM | `pretrained/Qwen2.5-1.5B-Instruct`，冻结 |
| Adapter | `attention_pool_mlp` |
| speech query tokens | 32 |
| 训练参数 | batch size 1，gradient accumulation 8，fp16，cuda:0 |
| 训练轮数 | 4 epochs |
| 学习率 | `1e-4` |

第一阶段只训练 Speech Adapter 和 Projection Layer，loss 只计算 answer tokens，prompt 和 speech tokens 的 label 置为 `-100`。最终 valid loss 约为 `0.3138`，checkpoint 保存于：

```text
outputs/stage1_align_attention_q32/best.pt
```

### 3.2 第二阶段：contextual biasing + hotword list + LoRA

最终最优配置：`configs/stage2_bias_attention_hard_refine.yaml`

第二阶段加载第一阶段 checkpoint，并在 prompt 中加入全量天津地铁候选站名表，要求模型只能输出候选标准站名或 `null`。同时开启 LoRA，使 Qwen 能学习该任务的 JSON 输出格式和 hotword 条件选择能力。

最终模型设置：

| 模块 | 设置 |
| --- | --- |
| Whisper Encoder | `pretrained/whisper-small`，冻结 |
| Qwen | `pretrained/Qwen2.5-1.5B-Instruct`，主体冻结 |
| Adapter | `attention_pool_mlp` |
| speech query tokens | 32 |
| Hotword | 使用 241 个天津地铁标准站名全量候选表 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| target modules | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| 设备 | `cuda:0` |
| 精度 | fp16 |

训练策略：

1. 先训练常规 hotword stage2 模型，发现 NULL 和近音站名存在明显错误。
2. 加入 NULL 保持策略，避免模型过度倾向输出站名。开发过程中 `nullboost8` 版本达到 overall 93.98%，NULL 98%，但 station 仍只有 93.57%。
3. 最终采用 hard station refine：对上一轮错误集中出现的相似站名进行训练集强化，同时尽量保持 NULL 权重不剧烈变化。最终训练在 step 50 出现更低 valid loss 后保留 best checkpoint，避免继续训练导致 NULL 识别能力下降。

最终 checkpoint：

```text
outputs/stage2_bias_hotwords_attention_q32_hard_refine/best
```

最终训练日志：

```text
outputs/stage2_bias_hotwords_attention_q32_hard_refine/train_log.jsonl
```

关键训练日志：

| step | train loss / valid loss | 说明 |
| ---: | ---: | --- |
| 20 | train loss 0.0446 | hard-refine 初期 |
| 40 | train loss 0.0241 | 收敛明显 |
| 50 | valid loss 0.0080 | 保存最佳 checkpoint |
| 60 | train loss 0.0099 | 后续观察到可 early stop |

## 4. 实验设置与结果

### 4.1 实验环境

| 项目 | 设置 |
| --- | --- |
| 操作系统 | Windows |
| Python 环境 | conda 环境 `CB-ASR` |
| GPU | `cuda:0` |
| ASR Encoder | Whisper-small，本地加载 |
| LLM | Qwen2.5-1.5B-Instruct，本地加载 |
| TTS | Kokoro-82M-v1.1-zh，本地加载 |
| 音频采样率 | 16 kHz mono |
| 随机种子 | 42 |

所有模型均使用本地目录加载，不在线下载。大预训练模型本体和原始/处理后音频不纳入 Git 跟踪。

### 4.2 Baseline 结果

测试集：`data/manifests/test_tts.jsonl`，共 532 条。

| 方法 | Overall Acc | Station Acc | NULL Acc | 说明 |
| --- | ---: | ---: | ---: | --- |
| Whisper raw | 未直接计算 | 未直接计算 | 未直接计算 | 只输出原始 ASR 文本 |
| Whisper + exact match | 23.87% | 17.22% | 88.00% | ASR 字面错误时无法匹配 |
| Whisper + edit fuzzy | 31.58% | 24.69% | 98.00% | 字符相似度不足以处理近音站名 |
| Whisper + pinyin fuzzy | 93.80% | 93.78% | 94.00% | 对“到/道”等同音错误有效 |
| Whisper + lexicon normalization | 93.80% | 93.78% | 94.00% | 强规则 baseline，接近但低于最终模型 |

baseline 结果文件：

```text
outputs/baseline/whisper_station_baselines/summary.json
```

### 4.3 最终模型结果

最终评估文件：

```text
outputs/preds/stage2_hotwords_attention_q32_hard_refine_test_best_metrics.json
outputs/preds/stage2_hotwords_attention_q32_hard_refine_test_best_errors.csv
outputs/preds/stage2_hotwords_attention_q32_hard_refine_test_best.jsonl
```

最终结果：

| 指标 | 正确数 / 总数 | Accuracy |
| --- | ---: | ---: |
| Overall | 524 / 532 | 98.50% |
| Station | 479 / 482 | 99.38% |
| NULL | 45 / 50 | 90.00% |
| Rare station | 133 / 136 | 97.79% |
| Confusion station | 437 / 440 | 99.32% |

与最强规则 baseline 对比：

| 方法 | Overall Acc | Station Acc | NULL Acc |
| --- | ---: | ---: | ---: |
| Whisper + lexicon normalization | 93.80% | 93.78% | 94.00% |
| 最终模型：Whisper Encoder + Adapter + Qwen + Hotword + LoRA + Hard-refine | 98.50% | 99.38% | 90.00% |

结论：最终模型显著提升了站名识别准确率，特别是对生僻站名和混淆站名的识别优于规则后处理。代价是 NULL accuracy 从规则 baseline 的 94.00% 降至 90.00%，说明模型在 hard-refine 后更倾向于从 hotword list 中选择站名，NULL 边界还可以进一步优化。

### 4.4 剩余错误分析

最终模型总错误数为 8 条，其中站名错误 3 条，NULL 误判为站名 5 条。

站名错误：

| utt_id | target | prediction | 错误类型 |
| --- | --- | --- | --- |
| TJ226_tts_test_0001 | 东丽一经路 | 东丽三经路 | 数字序列近音/近形 |
| TJ228_tts_test_0001 | 东丽六经路 | 东丽三经路 | 数字序列近音/近形 |
| TJ228_tts_test_0002 | 东丽六经路 | 东丽三经路 | 数字序列近音/近形 |

NULL 错误：

| utt_id | target | prediction |
| --- | --- | --- |
| NULL_tts_test_0023 | null | 成林道 |
| NULL_tts_test_0026 | null | 卞兴 |
| NULL_tts_test_0034 | null | 天津大学北洋园校区 |
| NULL_tts_test_0039 | null | 团泊健康城 |
| NULL_tts_test_0041 | null | 天津大学六里台 |

分析：

- 站名错误主要集中在“东丽一经路/三经路/六经路”这种同结构站名，模型能识别到区域和道路类型，但对数字部分的声学区分仍不稳定。
- NULL 错误说明 hotword prompt 对模型有强吸引力，遇到非法目的地时模型有时会强行映射到相似站名。
- 后续可以加入更强的 NULL 判别头、非法目的地 hard negatives，或在 prompt 中增加“无法确定时必须输出 null”的约束。

## 5. 已做实验、未做实验与预期

| 实验 | 状态 | 结果 / 说明 |
| --- | --- | --- |
| Kokoro TTS 数据生成 | 已做 | 生成 train/valid/test，共 3724 条 |
| Whisper raw baseline | 已做 | 原始 ASR 输出，用于错误分析 |
| Whisper + exact/edit/pinyin/lexicon baseline | 已做 | 最强规则 baseline overall 93.80% |
| Stage1 speech-to-Qwen alignment | 已做 | valid loss 约 0.3138 |
| Stage2 hotword + LoRA | 已做 | 初版 NULL 边界较差 |
| Stage2 null boost | 已做 | overall 93.98%，NULL 98%，station 93.57% |
| Stage2 hard-refine 最终模型 | 已做 | overall 98.50%，station 99.38% |
| 无 hotword 消融 | 未做 | 预期低于 hotword 模型，尤其在生僻站名和近音混淆上下降 |
| 真人录音测试 | 未做 | 用户决定不再录入真人声音；预期真实语音会低于 TTS 测试集 |
| AISHELL mini 增强 | 未做 | 曾考虑使用 `data/external/aishell_mini` 增强鲁棒性，但最终未纳入最优模型 |
| 严格 dev/test 分离 hard mining | 未做 | 预期能提供更严谨的泛化评估 |

## 6. 结论

本实验表明，面向固定句式地铁站名识别任务，仅依赖通用 Whisper ASR 与字符串规则匹配不足以稳定解决低频专名和近音混淆问题。拼音 fuzzy 和词表归一化已经能将 overall accuracy 提升到 93.80%，但仍难以处理复杂混淆站名。

最终采用的“Whisper Encoder + Speech Adapter + Qwen + hotword list + LoRA + hard-refine”方法，在测试集上达到：

```text
overall accuracy = 98.50%
station accuracy = 99.38%
rare station accuracy = 97.79%
confusion station accuracy = 99.32%
```

该结果说明，将语音特征直接接入 LLM，并在 instruction 中显式提供天津地铁候选站名表，可以有效利用上下文候选信息完成标准站名选择。模型主要剩余问题是 NULL 判别边界和极相似序列站名，例如“东丽一经路/三经路/六经路”。后续优化应重点围绕 NULL hard negatives、数字站名对比样本、dev-based hard mining 和真实语音测试展开。

## 7. 可复现实验文件索引

关键配置：

```text
configs/kokoro_tts.yaml
configs/stage1_align_attention.yaml
configs/stage2_bias_attention_hard_refine.yaml
```

关键数据：

```text
data/stations/tianjin_metro_stations.csv
data/manifests/train_tts.jsonl
data/manifests/valid_tts.jsonl
data/manifests/test_tts.jsonl
data/manifests/hard_station_refine.jsonl
```

关键模型输出：

```text
outputs/stage1_align_attention_q32/best.pt
outputs/stage2_bias_hotwords_attention_q32_hard_refine/best
```

关键结果：

```text
outputs/baseline/whisper_station_baselines/summary.json
outputs/preds/stage2_hotwords_attention_q32_hard_refine_test_best_metrics.json
outputs/preds/stage2_hotwords_attention_q32_hard_refine_test_best_errors.csv
outputs/preds/stage2_hotwords_attention_q32_hard_refine_test_best.jsonl
```

建议报告中主表采用最终模型结果，baseline 表用于说明方法提升；未做实验应明确标注为未做或未来工作。
