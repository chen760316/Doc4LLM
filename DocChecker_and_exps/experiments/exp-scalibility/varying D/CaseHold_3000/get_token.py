import json
import tiktoken


def consolidate_samples_strict_v2(input_json_path, output_json_path, target_samples=10, target_tokens=3000):
    """
    适配嵌套字典格式 (pos/neg)，严格控制每个样本长度为 1000 tokens。
    """
    # 1. 初始化分词器
    enc = tiktoken.get_encoding("cl100k_base")

    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 2. 提取所有文本进入 Token 池
    # 遍历 pos 和 neg 两个大类
    all_tokens = []

    # 定义处理顺序，先 pos 后 neg（或者根据需求调整）
    for category in ['pos', 'neg']:
        if category in data:
            # 遍历类别下的每一个 ID (如 "19107", "13694")
            for item_id in data[category]:
                item = data[category][item_id]
                text_content = item.get('text', '')

                # 如果需要区分文本边界，可以加一个换行符作为间隔
                if text_content:
                    # 将当前条目的文本转为 tokens 并加入大池子
                    all_tokens.extend(enc.encode(text_content + "\n\n"))

    # 3. 按照 target_tokens 的步长进行硬切分
    consolidated_data = []
    for i in range(target_samples):
        start_idx = i * target_tokens
        end_idx = start_idx + target_tokens

        sample_tokens = all_tokens[start_idx:end_idx]

        if not sample_tokens:
            print(f"警告：数据不足，仅生成了 {len(consolidated_data)} 个样本。")
            break

        # 将 tokens 还原回文本
        consolidated_data.append({
            "sample_index": i + 1,
            "text": enc.decode(sample_tokens),
            "actual_token_count": len(sample_tokens)
        })

    # 4. 保存结果
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(consolidated_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 处理完成：共从 pos/neg 数据中生成 {len(consolidated_data)} 个样本。")
    print(f"📏 每个样本目标长度: {target_tokens} tokens。")


# 调用
consolidate_samples_strict_v2('sampled_data.json', 'sampled_data.json')