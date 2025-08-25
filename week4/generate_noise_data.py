import pandas as pd
import random
import argparse
import re
from typing import List, Tuple

# 预设无关语料库
IRRELEVANT_SENTENCES = [
    "现代计算机由CPU、内存和硬盘组成，运行速度越来越快。",
    "历史上的唐朝始于618年，是中国最强盛的朝代之一。",
    "光合作用是植物利用阳光将二氧化碳转化为能量的过程。",
    "数学中的勾股定理指直角三角形两直角边平方和等于斜边平方。",
    "围棋起源于中国，是一种策略性两人棋类游戏，已有数千年历史。"
]

# 常见错别字映射表（周易领域相关）
TYPO_MAP = {
    "乾": "幹", "坤": "堃", "爻": "爻",  # 形近字
    "刚健": "刚建", "柔顺": "柔順", "象征": "象徵",
    "天行健": "天行建", "自强不息": "自强不熄"
}

def load_dataset(input_path: str) -> pd.DataFrame:
    """加载原始数据集（CSV格式，含'问题'和'回答'列）"""
    df = pd.read_csv(input_path)
    # 简单清洗：过滤空值
    df = df.dropna(subset=["content", "summary"]).reset_index(drop=True)
    df = df.rename(
    columns={
        "content": "问题",
        "summary": "回答"
    }
)
    print(f"加载原始数据完成，共 {len(df)} 条样本")
    return df

def inject_label_error(df: pd.DataFrame, noise_ratio: float) -> Tuple[pd.DataFrame, List[int]]:
    """注入标签错误：随机替换回答为其他样本的回答"""
    n_samples = len(df)
    n_noise = int(n_samples * noise_ratio)
    # 随机选择需要污染的样本索引
    noise_indices = random.sample(range(n_samples), n_noise)
    # 随机选择用于替换的回答（排除自身）
    all_answers = df["回答"].tolist()
    for idx in noise_indices:
        # 确保替换的回答不是原回答
        new_answer = random.choice([a for a in all_answers if a != df.loc[idx, "回答"]])
        df.loc[idx, "回答"] = new_answer
    return df, noise_indices

def inject_content_repeat(df: pd.DataFrame, noise_ratio: float) -> Tuple[pd.DataFrame, List[int]]:
    """注入内容重复：在回答中重复部分短句"""
    n_samples = len(df)
    n_noise = int(n_samples * noise_ratio)
    noise_indices = random.sample(range(n_samples), n_noise)
    for idx in noise_indices:
        answer = df.loc[idx, "回答"]
        # 按标点分割短句（简单分割逻辑）
        sentences = re.split(r"[，。,;；]", answer)
        sentences = [s for s in sentences if s.strip()]  # 过滤空句
        if len(sentences) < 2:
            continue  # 句子太少不重复
        # 随机选1-2个短句重复插入
        repeat_sentences = random.sample(sentences, k=min(2, len(sentences)))
        # 在随机位置插入重复内容
        insert_pos = random.randint(0, len(sentences))
        new_sentences = sentences[:insert_pos] + repeat_sentences + sentences[insert_pos:]
        df.loc[idx, "回答"] = "，".join(new_sentences)
    return df, noise_indices

def inject_irrelevant_info(df: pd.DataFrame, noise_ratio: float) -> Tuple[pd.DataFrame, List[int]]:
    """注入无关信息：在回答中插入无关句子"""
    n_samples = len(df)
    n_noise = int(n_samples * noise_ratio)
    noise_indices = random.sample(range(n_samples), n_noise)
    for idx in noise_indices:
        answer = df.loc[idx, "回答"]
        # 随机选1句无关内容
        irrelevant = random.choice(IRRELEVANT_SENTENCES)
        # 在随机位置插入（开头/中间/结尾）
        insert_pos = random.choice(["start", "middle", "end"])
        if insert_pos == "start":
            new_answer = f"{irrelevant} {answer}"
        elif insert_pos == "middle":
            # 简单分割为两部分插入
            split_pos = len(answer) // 2
            new_answer = f"{answer[:split_pos]} {irrelevant} {answer[split_pos:]}"
        else:
            new_answer = f"{answer} {irrelevant}"
        df.loc[idx, "回答"] = new_answer
    return df, noise_indices

def inject_format_error(df: pd.DataFrame, noise_ratio: float) -> Tuple[pd.DataFrame, List[int]]:
    """注入格式混乱：添加多余标点、替换错别字"""
    n_samples = len(df)
    n_noise = int(n_samples * noise_ratio)
    noise_indices = random.sample(range(n_samples), n_noise)
    for idx in noise_indices:
        answer = df.loc[idx, "回答"]
        # 1. 随机添加多余标点
        punctuation = ["，", "。", "、", "；", "，，"]
        # 每5个字符随机插入一个标点
        new_answer = []
        for i, char in enumerate(answer):
            new_answer.append(char)
            if i % 5 == 0 and random.random() < 0.3:  # 30%概率插入
                new_answer.append(random.choice(punctuation))
        answer = "".join(new_answer)
        # 2. 替换部分错别字
        for orig, typo in TYPO_MAP.items():
            if random.random() < 0.2:  # 20%概率替换该词
                answer = answer.replace(orig, typo)
        df.loc[idx, "回答"] = answer
    return df, noise_indices

def inject_qna_mismatch(df: pd.DataFrame, noise_ratio: float) -> Tuple[pd.DataFrame, List[int]]:
    """注入问题-回答不匹配：替换问题为其他样本的问题"""
    n_samples = len(df)
    n_noise = int(n_samples * noise_ratio)
    noise_indices = random.sample(range(n_samples), n_noise)
    all_questions = df["问题"].tolist()
    for idx in noise_indices:
        # 确保替换的问题不是原问题
        new_question = random.choice([q for q in all_questions if q != df.loc[idx, "问题"]])
        df.loc[idx, "问题"] = new_question
    return df, noise_indices

def verify_noise_ratio(noise_indices: List[int], total_samples: int, expected_ratio: float) -> None:
    """验证实际噪声比例是否符合预期"""
    actual_ratio = len(noise_indices) / total_samples
    print(f"噪声注入完成：预期比例 {expected_ratio:.2f}，实际比例 {actual_ratio:.2f}（共 {len(noise_indices)} 条噪声样本）")

def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    """保存噪声数据集"""
    df.to_csv(output_path, index=False)
    print(f"噪声数据集已保存至：{output_path}")

def main():
    # 命令行参数配置
    parser = argparse.ArgumentParser(description="生成带噪声的周易问答数据集")
    parser.add_argument("--input", type=str, required=True, help="原始数据集CSV路径")
    parser.add_argument("--output", type=str, required=True, help="噪声数据集输出路径")
    parser.add_argument("--noise-type", type=str, required=True,
                        choices=["label_error", "content_repeat", "irrelevant_info", "format_error", "qna_mismatch"],
                        help="噪声类型")
    parser.add_argument("--noise-ratio", type=float, required=True, help="噪声比例（0-1之间）", default=0.1)
    parser.add_argument("--seed", type=int, default=42, help="随机种子（保证可复现）")
    args = parser.parse_args()

    # 固定随机种子
    random.seed(args.seed)

    # 加载数据
    df = load_dataset(args.input)

    # 根据噪声类型注入噪声
    noise_functions = {
        "label_error": inject_label_error,
        "content_repeat": inject_content_repeat,
        "irrelevant_info": inject_irrelevant_info,
        "format_error": inject_format_error,
        "qna_mismatch": inject_qna_mismatch
    }
    noise_func = noise_functions[args.noise_type]
    df_noise, noise_indices = noise_func(df.copy(), args.noise_ratio)

    # 验证与保存
    verify_noise_ratio(noise_indices, len(df), args.noise_ratio)
    save_dataset(df_noise, args.output)

if __name__ == "__main__":
    main()






