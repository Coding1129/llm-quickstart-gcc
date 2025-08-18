
## 📝 作业1：

### 作业1-1：
- 任务描述：使用完整的YelpReviewFull数据集训练，观察acc的变化  
- 相关链接：[作业1-1](https://github.com/Coding1129/llm-quickstart-gcc/blob/main/week2/%E4%BD%9C%E4%B8%9A1-1%EF%BC%9Afine-tune-quickstart.ipynb)


### 作业1-2：
- 任务描述：加载本地保存的模型，训练评估再训练得到更高的F1 score  
- 相关链接：[作业1-2](https://github.com/Coding1129/llm-quickstart-gcc/blob/main/week2/%E4%BD%9C%E4%B8%9A1-2%EF%BC%9A%20fine_tune_QA.ipynb)


---

## 📝 作业2：

### 作业2-1：
- 任务描述：LoRA 在OpenAI Whisper-large-v2模型上实现语音识别(ASR)任务的微调训练  
  （注：因调节训练参数时显存不足需中断内核，故将数据导入与模型训练分开处理）

  - 数据处理：[数据处理代码](https://github.com/Coding1129/llm-quickstart-gcc/blob/main/week2/save_dataset.ipynb)  
  - 1. 针对中文全数据集的微调：[中文全量微调](https://github.com/Coding1129/llm-quickstart-gcc/blob/main/week2/%E4%BD%9C%E4%B8%9A2-1%EF%BC%9A%20peft_lora_whisper_large_v2_fi_alldata_finetune.ipynb)  
  - 2. 针对芬兰语的全数据集微调：[芬兰语全量微调](https://github.com/Coding1129/llm-quickstart-gcc/blob/main/week2/%E4%BD%9C%E4%B8%9A2-1%EF%BC%9A%20peft_lora_whisper_large_v2_fi_alldata_finetune.ipynb)


### 作业2-2：
- 任务描述：测试集合的评估  

  - 1. 针对中文微调模型的评估：[中文模型评估](https://github.com/Coding1129/llm-quickstart-gcc/blob/main/week2/%E4%BD%9C%E4%B8%9A2-2%EF%BC%9A%20peft_lora_whisper_large_v2_zhch_alldataset_evl.ipynb)  
  - 2. 针对波兰语微调模型的评估（增大了评估数据）：[波兰语模型评估](https://github.com/Coding1129/llm-quickstart-gcc/blob/main/week2/%E4%BD%9C%E4%B8%9A2-2%EF%BC%9A%20peft_lora_whisper_large_v2_fi_alldataset_evl.ipynb)

