# 探究不同的微调数据和轮次对模型效果的影响

1. 首先利用`data/zhouyi_dataset_20240118_152413.csv`微调ChatGLM3-6B模型
   - 作业1-1：`qlora_chatglm3_timestamp.ipynb`

2. 其次利用`generate_noise_data.py`脚本生成不同数据错误的csv微调数据：
   - `zhouyi_format_error_30%.csv`
   - `zhouyi_label_error_30%.csv`
   - `zhouyi_repeat_30%.csv`

3. 分别利用上述数据进行微调：
   - 作业1-2：`qlora_chatglm3_timestamp_format_error.ipynb`
   - 作业1-3：`qlora_chatglm3_timestamp_label_error.ipynb`
   - 作业1-4：`qlora_chatglm3_timestamp_repeat.ipynb`

4. 检验过拟合对模型效果的影响：
   - 设置`epoch=50`进行微调
   - 作业1-5：`qlora_chatglm3_timestamp_epoch50.ipynb`

5. 比较上述微调模型在任务上的表现：
   - 作业2：`chatglm_inference.ipynb`


