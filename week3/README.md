## 📝 作业1：

### 作业1-1：
- 任务描述：使用 GPTQ 量化 OPT-6.7B 模型。课程代码（https://github.com/DjangoPeng/LLM-quickstart/blob/main/quantization/AutoGPTQ_opt-2.7b.ipynb）  
- 相关链接：[作业1-1](https://github.com/Coding1129/llm-quickstart-gcc/blob/main/week3/%E4%BD%9C%E4%B8%9A1%EF%BC%9AAutoGPTQ_opt-6.7b.ipynb)

### 作业1-2：
- 任务描述：使用 AWQ 量化 Facebook OPT-6.7B 模型。课程代码（https://github.com/DjangoPeng/LLM-quickstart/blob/main/quantization/AWQ-opt-125m.ipynb）  
- 相关链接：[作业1-2](https://github.com/Coding1129/llm-quickstart-gcc/blob/main/week3/%E4%BD%9C%E4%B8%9A2%EF%BC%9AAWQ_opt-6.7b.ipynb)

---

## 📝 作业2：

课程代码（https://github.com/DjangoPeng/LLM-quickstart/blob/main/peft/peft_qlora_chatglm.ipynb）

### 作业2-1：
- 任务描述：使用PEFT库QLoRA微调量化后的ChatGLM3-6B模型，其中训练的max_steps=100。
- 相关链接：[作业2-1：peft_qlora_chatglm_example.ipynb]()

### 作业2-2：
- 任务描述：使用PEFT库QLoRA微调量化后的ChatGLM3-6B模型，其中训练1个完整的数据集遍历（epoch）
- 相关链接：[作业2-2：peft_qlora_chatglm_10k_linear.ipynb]()

### 作业2-3：
- 任务描述：使用PEFT库QLoRA微调量化后的ChatGLM3-6B模型，其中训练1个完整的数据集，并针对作业2-2进行参数调整，更适配显卡，训练效果更好，达到了更低的train loss
- 相关链接：[作业2-3：peft_qlora_chatglm_10k_cosine.ipynb]()

课程代码（https://github.com/DjangoPeng/LLM-quickstart/blob/main/peft/peft_chatglm_inference.ipynb）

### 作业2-4：
- 任务描述：比较作业2-2和作业2-3训练的两个模型在任务上的表现，作业2-3的模型明显更优
- 相关链接：[作业2-4：peft_chatglm_inference_compare.ipynb]()

## 补充

### 作业2-2和作业2-3训练参数对比

| 参数 | 本例配置 | 示例配置 | 关键差异 |
|------|------------|------------|----------|
| `max_xxx_length` | `1024/2048` | `512/1536 ` | 增强长文本处理能力 |
| `per_device_train_batch_size` | 24 | 16 | 更大的批量大小，更有效利用显存 |
| `num_train_epochs`/`max_steps` | `num_train_epochs=1` | `max_steps=100` | epoch控制 vs 步数控制 |
| `lr_scheduler_type` | `cosine` | `linear` | 余弦 vs 线性衰减，余弦更有利于模型训练后期精细调整参数 |
| `warmup_ratio` | 0.05 | 0.1 | 减少预热阶段，避免占用更多训练步数 |
| `logging_steps` | 200 | 10 | 降低日志更新频次 |
| `save_steps` | 无 | 20 | 检查点 |
| `bf16`/`fp16` | `bf16=True` | `fp16=True` | 精度格式不同，bf=16更适用于4090上训练 |

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

### 作业2-2和作业2-3的训练分析

#### 1. 训练损失
- 作业2-2：损失从 3.64 逐步降至 3.03，下降幅度较平缓，最后几步下降放缓
- 作业2-3：3.48 快速降至 2.99，下降速度更快，且后期仍保持下降趋势


#### 2. 训练效率对比
- 训练速度：最终总训练时间差异不大（7810 秒 vs 8092 秒）。
- 计算量：作业2-3的total_flos（总浮点运算数）更高（7.19e17 vs 6.92e17），说明其在相同 epoch 内对数据的 "学习强度" 更高（更大的 batch size 可能带来更稳定的梯度估计）。


#### 3. 参数设置的合理性
- 学习率调度器：cosine 调度器（作业2-3）相比 linear 调度器（作业2-2），后期学习率下降更平滑，更有利于模型精细调整参数，通常能获得更好的收敛效果（这与训练 2 损失更低一致）。
- 预热比例：作业2-3的预热比例更低（5%），在相同步数下能更快进入正式训练阶段，结合 cosine 调度器，可能更适配其训练节奏。
- batch size：更大的 batch size（24）可能减少梯度噪声，使模型更新更稳定（作业2-3的损失下降更平滑也印证了这一点）。



### 作业2-2和作业2-3的微调效果分析
作业2-4：peft_chatglm_inference_compare.ipynb 两个模型的输出如下：  

**ChatGLM3-6B liner微调后: 
这一款连衣裙撞色圆领的设计，修饰脖颈精致优雅。简约的版式，穿着大方优雅，特别有文艺范。精美的印花装饰，精致美丽特别出彩。加上压褶的设计，修饰身材显瘦显高。
ChatGLM3-6B cosine微调后: 
连衣裙采用撞色印花图案设计，搭配上精致圆领设计，穿着尽显文艺气息。而其修身版型设计，结合压褶裙摆设计，穿着尽显俏皮可爱气息。简约大方，更显灵动飘逸。**

从生成内容的相关性、丰富度、语言流畅度和风格适配性来看，两个微调配置的输出各有特点，但作业2-3的结果整体表现更优，具体如下：  
#### 1. 内容相关性与核心信息覆盖：  
两者均围绕“连衣裙设计特点”展开，覆盖撞色、圆领、印花、压褶等核心元素。作业2-2聚焦核心卖点但维度较集中；作业2-3额外拆解“撞色印花”“修身版型”“压褶裙摆”，细节更细致。  
#### 2. 描述丰富度与表达细腻度：    
作业2-2用词平实且重复“优雅”类词汇，表达稍单调；作业2-3新增“灵动飘逸”“俏皮可爱”等风格化词汇，通过递进表达让描述更立体，逻辑关联更强。  
#### 3. 语言流畅度与风格一致性：  
两者均无语法错误，但作业2-2句式规整、节奏平缓；作业2-3长短句结合，节奏更生动，风格融合“文艺”“俏皮”“灵动”，避免单调感。  
#### 4. 与训练效果的关联性：  
作业2-3训练损失更低、收敛更充分，对“服饰描述逻辑”“风格化词汇搭配”学习更充分，输出更细腻自然；作业2-2因训练损失下降较平缓，细节丰富度和语言灵动性稍逊。  

#### 总结
作业2-3的输出在细节拆解、词汇丰富度、语言流畅度上均优于作业2-2，更能体现连衣裙设计的层次感和风格特点，与训练阶段“损失更低、收敛更优”的结果呼应，微调效果更贴合实际需求。 

### 结论：作业2-3更优
作业2-3 凭借cosine 调度器 + 更大batch size的组合，更有利于模型收敛，最终损失更低、损失下降更快，微调后的整体训练效果更优。