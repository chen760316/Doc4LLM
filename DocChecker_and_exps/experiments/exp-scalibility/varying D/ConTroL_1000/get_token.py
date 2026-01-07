import json
import tiktoken


def consolidate_samples_strict(input_json_path, output_json_path, target_samples=10, target_tokens=1000):
    """
    严格控制每个样本长度为 1000 tokens，多余文本会流转到下一个样本。
    """
    # 初始化分词器 (GPT-4 常用编码)
    enc = tiktoken.get_encoding("cl100k_base")

    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 将所有文本先合并成一个大的 token 池
    all_tokens = []
    for entry in data:
        all_tokens.extend(enc.encode(entry.get('text', '')))

    consolidated_data = []

    # 按照 target_tokens 的步长进行硬切分
    for i in range(target_samples):
        start_idx = i * target_tokens
        end_idx = start_idx + target_tokens

        # 提取指定范围内的 tokens
        sample_tokens = all_tokens[start_idx:end_idx]

        if not sample_tokens:
            break

        # 将 tokens 还原回文本
        consolidated_data.append({
            "label": "c",
            "text": enc.decode(sample_tokens),
            "actual_token_count": len(sample_tokens)
        })

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(consolidated_data, f, ensure_ascii=False, indent=2)

    print(f"处理完成：共生成 {len(consolidated_data)} 个样本，每段长度限制为 {target_tokens} tokens。")


# 使用前请安装: pip install tiktoken
consolidate_samples_strict('sampled_data.json', 'sampled_data.json')