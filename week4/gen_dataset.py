import os
import csv
import datetime
import requests
import json
from dotenv import load_dotenv

# -------------------------- 1. 环境配置与基础函数 --------------------------
# 加载环境变量（存储豆包API密钥）
load_dotenv()


class DoubaoConfig:
    """豆包API配置类"""
    API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    DEFAULT_MODEL = "doubao-1-5-thinking-pro-250415"

    # API请求头（含认证信息）
    @classmethod
    def get_headers(cls):
        api_key = os.getenv("DOUBA_API_KEY")
        if not api_key:
            raise ValueError("请在.env文件中配置DOUBA_API_KEY（格式：DOUBA_API_KEY='sk-xxx'）")
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }


# -------------------------- 2. 原始数据加载模块 --------------------------
def load_raw_data(raw_data_path: str = "raw_data.txt") -> list:
    """
    加载周易卦象原始数据（raw_data.txt），按空行分割为单个样本
    :param raw_data_path: 原始数据文件路径
    :return: 清洗后的卦象样本列表
    """
    # 若原始数据文件不存在，创建示例文件
    if not os.path.exists(raw_data_path):
        with open(raw_data_path, "w", encoding="utf-8") as f:
            f.write("""蒙卦原文
蒙。亨。匪我求童蒙，童蒙求我。初筮告，再三渎，渎则不告。利贞。
象曰：山下出泉，蒙。君子以果行育德。
白话文解释：蒙卦象征通泰，需通过教育启蒙化解蒙昧，君子应果断行动培养德行。

屯卦原文
屯。元，亨，利，贞。勿用，有攸往，利建侯。
象曰：云，雷，屯；君子以经纶。
白话文解释：屯卦代表万物初生的艰难，需刚毅果敢应对，君子应效法云雷治理事务。""")
        print(f"已创建示例原始数据文件：{raw_data_path}，请补充完整卦象内容")

    # 读取并分割数据
    with open(raw_data_path, "r", encoding="utf-8") as f:
        content = f.read()
        # 按连续空行分割不同卦象
        data_samples = [sample.strip() for sample in content.split("\n\n") if sample.strip()]
    print(f"成功加载 {len(data_samples)} 个卦象原始样本")
    return data_samples


# -------------------------- 3. 豆包结构化数据生成模块 --------------------------
def gen_zhouyi_structured_data(raw_content: str) -> str:
    """
    调用豆包API，将原始卦象文本转换为content（卦名）+ summary（结构化解读）格式
    :param raw_content: 单个卦象的原始文本
    :return: 豆包生成的结构化文本
    """
    # 提示词设计（明确周易专家身份+格式要求+示例引导）
    system_prompt = """
    你是中国古典哲学专家，精通周易卦象解读。请按以下要求处理输入的卦象内容：
    1. 提取卦名（如“蒙卦”“屯卦”），用content字段存储；
    2. 整合原始内容，生成summary字段，需包含：
       - 卦象构成（上下卦及象征，如“下坎上艮，坎为水、艮为山”）；
       - 核心哲学（如“启蒙教育的重要性”“初生艰难的应对”）；
       - 传统解读（《象辞》《序卦》等引用，或邵雍、傅佩荣等学者观点）；
       - 运势/事业/婚恋指引（简要提炼实用价值）；
    3. 输出格式严格遵循（无多余内容，不换行破坏格式）：
    content:"{卦名}"
    summary:"{结构化解读内容}"
    """

    # 构建API请求体
    request_body = {
        "model": DoubaoConfig.DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"原始卦象内容：{raw_content}"}
        ],
        "temperature": 0.6,  # 控制生成严谨性（0.6平衡灵活与准确）
        "max_tokens": 1500,
        "stream": False
    }

    # 发送API请求
    try:
        response = requests.post(
            url=DoubaoConfig.API_URL,
            headers=DoubaoConfig.get_headers(),
            data=json.dumps(request_body),
            timeout=30
        )
        response.raise_for_status()  # 非200状态码抛出异常
        result = response.json()["choices"][0]["message"]["content"].strip()
        return result
    except Exception as e:
        print(f"豆包API调用失败：{str(e)}")
        return ""


# -------------------------- 4. 数据解析模块 --------------------------
def parse_structured_data(generated_text: str) -> tuple:
    """
    解析豆包生成的结构化文本，提取content（卦名）和summary（解读）
    :param generated_text: 豆包返回的结构化文本
    :return: (content, summary) 或 (None, None)（解析失败）
    """
    try:
        # 提取content（格式：content:"xxx"）
        content_start = generated_text.find('content:"') + len('content:"')
        content_end = generated_text.find('"\nsummary:')
        content = generated_text[content_start:content_end].strip().replace('"', "")

        # 提取summary（格式：summary:"xxx"）
        summary_start = generated_text.find('summary:"') + len('summary:"')
        summary_end = generated_text.rfind('"')
        summary = generated_text[summary_start:summary_end].strip()

        if not content or not summary:
            raise ValueError("提取的content或summary为空")
        return content, summary
    except Exception as e:
        print(f"数据解析失败：{str(e)}，生成文本：{generated_text[:100]}...")
        return None, None


# -------------------------- 5. 数据增强模块（多样化提问） --------------------------
def generate_question_pairs(content: str, summary: str) -> list:
    """
    基于卦名生成20种多样化提问，与summary组成（question, answer）对，提升数据集丰富度
    :param content: 卦名（如“蒙卦”）
    :param summary: 结构化解读（answer）
    :return: 20组提问-回答对
    """
    question_templates = [
        f"{content}是什么？请详细解释。",
        f"周易中的{content}有什么象征意义？",
        f"{content}的卦象构成是怎样的？包含哪些哲学思想？",
        f"如何理解{content}的核心含义？",
        f"{content}在《象辞》中的解读是什么？",
        f"{content}对事业发展有哪些指引？",
        f"占得{content}意味着什么？运势如何？",
        f"{content}与启蒙教育有什么关联？（若无关则替换为对应主题）",
        f"请说明{content}的上下卦及各自象征。",
        f"{content}的传统解读中，邵雍或傅佩荣有哪些观点？",
        f"{content}的卦辞“{summary[:10]}...”（截取开头）该如何理解？",
        f"为什么说{content}是周易中的重要卦象？",
        f"{content}对婚恋决策有什么建议？",
        f"{content}代表的自然意象（如天、地、水）有哪些？",
        f"{content}与其他卦象（如屯卦、需卦）有什么区别？",
        f"学习{content}能获得哪些人生启示？",
        f"{content}中的“贞”“亨”等卦辞该如何解读？",
        f"现代生活中，{content}的智慧该如何应用？",
        f"{content}为什么象征{summary}？",
        f"请总结{content}的核心思想和实用价值。"
    ]
    # 生成提问-回答对（确保每个提问对应同一summary）
    return [(template.format(content=content), summary) for template in question_templates]


# -------------------------- 6. 数据集持久化模块 --------------------------
def save_dataset(data_pairs: list) -> str:
    """
    将提问-回答对保存为CSV文件（含时间戳，避免覆盖）
    :param data_pairs: （question, answer）对列表
    :return: 保存的CSV文件路径
    """
    # 创建data目录（若不存在）
    if not os.path.exists("data"):
        os.makedirs("data")

    # 生成带时间戳的文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"data/zhouyi_dataset_{timestamp}.csv"

    # 写入CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer"])  # 表头（适配微调格式：输入-输出）
        writer.writerows(data_pairs)

    print(f"\n数据集已保存至：{csv_path}，共{len(data_pairs)}条训练数据")
    return csv_path


# -------------------------- 7. 主流程控制模块 --------------------------
def main():
    print("=" * 60)
    print("         豆包周易卦象训练数据集自动化生成工具")
    print("=" * 60)

    # 1. 加载原始数据
    raw_samples = load_raw_data()
    if not raw_samples:
        print("无有效原始数据，退出程序")
        return

    # 2. 批量处理每个卦象样本
    all_question_answer_pairs = []
    for idx, raw_sample in enumerate(raw_samples, 1):
        print(f"\n正在处理第 {idx}/{len(raw_samples)} 个卦象...")

        # 3. 豆包生成结构化数据
        generated_text = gen_zhouyi_structured_data(raw_sample)
        if not generated_text:
            print(f"第 {idx} 个卦象生成失败，跳过")
            continue

        # 4. 解析结构化数据
        content, summary = parse_structured_data(generated_text)
        if not content or not summary:
            print(f"第 {idx} 个卦象解析失败，跳过")
            continue
        print(f"✅ 成功解析：卦名 = {content}")

        # 5. 生成多样化提问对（数据增强）
        question_pairs = generate_question_pairs(content, summary)
        all_question_answer_pairs.extend(question_pairs)

    # 6. 保存数据集
    if all_question_answer_pairs:
        save_dataset(all_question_answer_pairs)
        print("\n🎉 所有有效卦象处理完成！")
    else:
        print("\n❌ 未生成有效训练数据，请检查API配置或原始数据")


# -------------------------- 8. 运行入口 --------------------------
if __name__ == "__main__":
    # 检查.env文件是否存在
    if not os.path.exists(".env"):
        with open(".env", "w", encoding="utf-8") as f:
            f.write('DOUBA_API_KEY=""  # 替换为你的豆包API密钥，获取地址：https://www.doubao.com/\n')
        print("已自动创建.env文件，请填写你的豆包API密钥后重新运行！")
        exit(1)

    # 启动主流程
    main()


"""
运行输出：gen_dataset.py 
============================================================
         豆包周易卦象训练数据集自动化生成工具
============================================================
成功加载 8 个卦象原始样本

正在处理第 1/8 个卦象...
✅ 成功解析：卦名 = 蒙卦

正在处理第 2/8 个卦象...
✅ 成功解析：卦名 = 屯卦

正在处理第 3/8 个卦象...
✅ 成功解析：卦名 = 需卦

正在处理第 4/8 个卦象...
✅ 成功解析：卦名 = 讼卦

正在处理第 5/8 个卦象...
✅ 成功解析：卦名 = 师卦

正在处理第 6/8 个卦象...
✅ 成功解析：卦名 = 比卦

正在处理第 7/8 个卦象...
✅ 成功解析：卦名 = 坤卦

正在处理第 8/8 个卦象...
✅ 成功解析：卦名 = 乾卦

数据集已保存至：data/zhouyi_dataset_20250825_185104.csv，共160条训练数据

🎉 所有有效卦象处理完成！

Process finished with exit code 0

"""