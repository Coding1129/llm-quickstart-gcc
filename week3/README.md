# 探究模型量化与QLoRA微调参数对训练效果的影响

## 一、模型量化
通过两种主流量化方法（GPTQ、AWQ）对OPT-6.7B模型进行量化，验证不同量化技术在模型压缩中的应用效果，具体作业配置如下：

### 1 GPTQ量化OPT-6.7B模型
- **任务描述**：使用GPTQ算法对OPT-6.7B模型进行量化，实现模型体积压缩的同时保障基础性能。
- **课程参考代码**：[AutoGPTQ_opt-2.7b.ipynb](https://github.com/DjangoPeng/LLM-quickstart/blob/main/quantization/AutoGPTQ_opt-2.7b.ipynb)
- **作业实现代码**：[作业1-1：AutoGPTQ_opt-6.7b.ipynb](https://github.com/Coding1129/llm-quickstart-gcc/blob/main/week3/%E4%BD%9C%E4%B8%9A1%EF%BC%9AAutoGPTQ_opt-6.7b.ipynb)

### 2 AWQ量化OPT-6.7B模型
- **任务描述**：使用AWQ（Activation-aware Weight Quantization）算法对OPT-6.7B模型进行量化，优化量化过程中激活值与权重的匹配度。
- **课程参考代码**：[AWQ-opt-125m.ipynb](https://github.com/DjangoPeng/LLM-quickstart/blob/main/quantization/AWQ-opt-125m.ipynb)
- **作业实现代码**：[作业1-2：AWQ_opt-6.7b.ipynb](https://github.com/Coding1129/llm-quickstart-gcc/blob/main/week3/%E4%BD%9C%E4%B8%9A2%EF%BC%9AAWQ_opt-6.7b.ipynb)


## 二、QLoRA微调
基于PEFT库的QLoRA技术，对量化后的ChatGLM3-6B模型进行微调，通过控制**训练步数、epoch数量、参数配置**三个变量，探究不同微调策略对模型训练效果与生成质量的影响，具体作业流程与配置如下：

### 1 基础微调任务
- **任务描述**：以“服饰描述”为目标任务，通过调整核心训练参数，对比不同配置下的模型收敛情况，所有任务均基于量化后的ChatGLM3-6B模型，使用PEFT库QLoRA方法实现微调。
- **课程参考代码**：[peft_chatglm_inference.ipynb](https://github.com/DjangoPeng/LLM-quickstart/blob/main/peft/peft_chatglm_inference.ipynb)
- **作业实现代码**：如下
| 作业编号 | 任务描述                                                                 | 作业实现代码                                                                 |
|----------|--------------------------------------------------------------------------|------------------------------------------------------------------------------|
| **作业2-1** | 固定训练步数（max_steps=100），完成基础QLoRA微调                          | [作业2-1：peft_qlora_chatglm_example.ipynb](https://github.com/Coding1129/llm-quickstart-gcc/blob/main/week3/%E4%BD%9C%E4%B8%9A2-1%EF%BC%9Apeft_qlora_chatglm_example.ipynb) |
| **作业2-2** | 固定训练轮次（num_train_epochs=1，完整遍历1次数据集），使用linear学习率调度 | [作业2-2：peft_qlora_chatglm_10k_linear.ipynb](https://github.com/Coding1129/llm-quickstart-gcc/blob/main/week3/%E4%BD%9C%E4%B8%9A2-2%EF%BC%9Apeft_qlora_chatglm_10k_linear.ipynb) |
| **作业2-3** | 基于作业2-2优化参数（适配RTX 4090显卡、降低train loss），使用cosine学习率调度 | [作业2-3：peft_qlora_chatglm_10k_cosine.ipynb](https://github.com/Coding1129/llm-quickstart-gcc/blob/main/week3/%E4%BD%9C%E4%B8%9A2-3%EF%BC%9Apeft_qlora_chatglm_10k_cosine.ipynb) |

### 2 微调模型效果对比
- **任务描述**：对比作业2-2（linear调度）与作业2-3（cosine调度）微调模型在“服饰描述”任务上的输出表现，验证参数优化对生成质量的提升作用。
- **课程参考代码**：[peft_chatglm_inference.ipynb](https://github.com/DjangoPeng/LLM-quickstart/blob/main/peft/peft_chatglm_inference.ipynb)
- **作业实现代码**：[作业2-4：peft_chatglm_inference_compare.ipynb](https://github.com/Coding1129/llm-quickstart-gcc/blob/main/week3/%E4%BD%9C%E4%B8%9A2-4%EF%BC%9Apeft_chatglm_inference_compare.ipynb)


## 三、核心参数对比与训练环境
### 1 作业2-2与作业2-3关键参数差异
作业2-3通过调整参数适配RTX 4090显卡，同时优化训练稳定性与收敛效果，具体参数对比如下：

| 参数                          | 作业2-2（linear调度） | 作业2-3（cosine调度） | 关键差异分析                                                                 |
|-------------------------------|-----------------------|-----------------------|------------------------------------------------------------------------------|
| `max_input_length/max_output_length`              | 512/1536              | 1024/2048             | 作业2-3提升长文本处理能力，适配更复杂的服饰描述场景                           |
| `per_device_train_batch_size` | 16                    | 24                    | 作业2-3增大批量大小，更充分利用RTX 4090显存，降低梯度噪声                     |
| `lr_scheduler_type`           | linear                | cosine                | 作业2-3采用余弦衰减，调度更平滑，利于训练后期精细调整参数，提升收敛效果       |
| `warmup_ratio`                | 0.1                   | 0.05                  | 作业2-3减少预热阶段占比，更快进入正式训练，适配cosine调度节奏                 |
                  |
### 2 训练环境配置
通过`nvidia-smi`查看训练时显卡状态：
```bash
(peft) cc@js:~/.virtualenvs$ nvidia-smi
Thu Aug 21 16:01:07 2025       

cc@js:/$ nvidia-smi
Thu Aug 21 16:30:48 2025       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.230.02             Driver Version: 535.230.02   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4090        Off | 00000000:01:00.0  On |                  Off |
| 81%   69C    P2             379W / 450W |  16932MiB / 24564MiB |     99%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce RTX 4090        Off | 00000000:05:00.0 Off |                  Off |
| 71%   61C    P2             362W / 450W |  21531MiB / 24564MiB |    100%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
```


## 四、训练效果与生成质量对比分析
### 训练过程指标对比（作业2-2 vs 作业2-3）
#### 1. 训练损失（Train Loss）
- **作业2-2**：损失从3.64逐步降至3.03，下降幅度平缓，后期下降趋势放缓，说明收敛效率较低。
- **作业2-3**：损失从3.48快速降至2.99，下降速度更快，且训练全程保持稳定下降，收敛更充分。

#### 2. 训练效率与计算量
- **训练时间**：两者总时长差异较小（7810秒 vs 8092秒），无明显效率损耗。
- **总浮点运算数（total_flos）**：作业2-3（7.19e17）高于作业2-2（6.92e17），更大的batch size带来更稳定的梯度估计，“学习强度”更高。

#### 3. 参数配置合理性验证
- **学习率调度**：cosine调度器（作业2-3）后期学习率平滑下降，避免参数震荡，与“损失更低”的结果直接呼应。
- **Batch Size**：24的批量大小减少梯度噪声，使作业2-3的损失曲线更平滑，训练稳定性显著提升。

### 生成质量对比（基于“连衣裙描述”任务）
#### 1. 模型输出示例
| 模型版本                | 输出内容                                                                 |
|-------------------------|--------------------------------------------------------------------------|
| 作业2-2（linear微调）   | 这一款连衣裙撞色圆领的设计，修饰脖颈精致优雅。简约的版式，穿着大方优雅，特别有文艺范。精美的印花装饰，精致美丽特别出彩。加上压褶的设计，修饰身材显瘦显高。 |
| 作业2-3（cosine微调）   | 连衣裙采用撞色印花图案设计，搭配上精致圆领设计，穿着尽显文艺气息。而其修身版型设计，结合压褶裙摆设计，穿着尽显俏皮可爱气息。简约大方，更显灵动飘逸。 |

#### 2. 生成质量多维度评分
| 评价维度                | 作业2-2（linear微调）                          | 作业2-3（cosine微调）                          |
|-------------------------|------------------------------------------------|------------------------------------------------|
| **内容相关性**          | 覆盖核心元素（撞色、圆领、印花、压褶），但维度集中 | 拆解细节（撞色印花、修身版型、压褶裙摆），维度更丰富 |
| **描述丰富度**          | 用词平实，重复“优雅”类词汇，表达单调            | 新增“灵动飘逸”“俏皮可爱”，风格化词汇更立体      |
| **语言流畅度**          | 句式规整，无语法错误，但节奏平缓                | 长短句结合，节奏生动，避免单调感                |
| **风格一致性**          | 聚焦“文艺优雅”，风格统一但单一                  | 融合“文艺+俏皮+灵动”，风格更贴合服饰定位        |
