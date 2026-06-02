import os
import json
import pandas as pd


def build_repe_cm_dataset(file_path):
    """
    读取指定路径的本地 Commonsense Morality 数据，重构为符合 RepE 论文规范的对比对。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到目标文件，请检查路径是否正确: {file_path}")

    print(f"正在解析本地 CSV: {file_path}")
    # 读取并剥离无效空行
    df = pd.read_csv(file_path)[['input', 'label']].dropna()

    # 按标签分流 (0: 符合常识道德 / 1: 不道德)
    df_moral = df[df['label'] == 0].reset_index(drop=True)
    df_immoral = df[df['label'] == 1].reset_index(drop=True)

    # 严格套用 RepE 论文附录 D.1.9 的官方 Morality 概念模板
    moral_template = (
        "USER: Consider the acceptability of the action in the following scenario:\n"
        "Scenario: {}\n\n"
        "ASSISTANT:"
    )

    contrastive_pairs = []
    # 1:1 构建无监督差值矩阵所需的强对比对 (Contrastive Pairs)
    min_len = min(len(df_moral), len(df_immoral))

    for i in range(min_len):
        high_prompt = moral_template.format(df_moral.loc[i, 'input'])
        low_prompt = moral_template.format(df_immoral.loc[i, 'input'])

        contrastive_pairs.append({
            "high": high_prompt,
            "low": low_prompt
        })

    print(f"数据集重构成功，共对齐 {len(contrastive_pairs)} 组样本对。")
    return contrastive_pairs


if __name__ == "__main__":
    # 使用 r"" 声明原始字符串，完美兼容 Windows 物理路径
    CSV_FILE_PATH = r"E:\Ajou作业\AI Reserch\REPE复现\data\ethics\commonsense\cm_train.csv"

    try:
        # 执行管道清洗
        templated_data = build_repe_cm_dataset(CSV_FILE_PATH)

        # 自动获取 CSV 所在目录，将 JSON 生成在同级目录下
        output_dir = os.path.dirname(CSV_FILE_PATH)
        json_output_path = os.path.join(output_dir, "morality_templated_pairs.json")

        # 写入本地存储
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(templated_data, f, ensure_ascii=False, indent=2)

        print(f"工程闭环！规范化 JSON 已保存至: {json_output_path}")

    except Exception as e:
        print(f"【工程错误】: {e}")