import json
import random
import os
from collections import defaultdict
from typing import Dict, List, Any

# --- 配置 ---
# ⚠️ 请根据您的实际文件路径修改
INPUT_FILE_PATH = "ContraDoc.json"
# 输出文件名
OUTPUT_FILE_PATH = "sampled_data.json"
# 每种 scope 类型的采样数量
SAMPLE_COUNT_PER_SCOPE = 30


def sample_contradoc_by_scope(input_path: str, output_path: str, count: int):
    """
    从 ContraDOC JSON 文件的 'pos' 部分，按 'scope' 标签采样指定数量的样本。
    如果一个样本有多个 scope 标签，它会被计入所有相应的 scope 组进行采样。
    """
    if not os.path.exists(input_path):
        print(f"❌ 错误：找不到输入文件 '{input_path}'。请检查路径。")
        return

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except json.JSONDecodeError:
        print(f"❌ 错误：JSON 文件 '{input_path}' 解析失败。")
        return
    except Exception as e:
        print(f"❌ 读取文件时发生错误: {e}")
        return

    # 1. 初始化存储结构
    # 键是 scope 类型 (如 'local', 'intra', 'global')，值是符合该 scope 的样本列表
    scope_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # 提取 pos 样本
    pos_samples = full_data.get('pos', {})

    # 2. 按 scope 分组样本
    print(f"🔍 正在处理 {len(pos_samples)} 条正例样本...")
    all_scopes = set()
    for uid, item in pos_samples.items():
        scopes = item.get('scope')

        # 确保 scope 字段存在且是字符串或列表，并处理为列表
        if isinstance(scopes, str):
            scopes = [scopes]  # 单个字符串也视为列表
        elif not isinstance(scopes, list):
            scopes = []  # 其他非列表/字符串格式则忽略

        for scope_tag in scopes:
            scope_tag = scope_tag.lower().strip()  # 标准化标签
            if scope_tag:
                all_scopes.add(scope_tag)
                # 存储样本 ID 和内容
                sample_data = item.copy()
                sample_data['uid'] = uid
                scope_groups[scope_tag].append(sample_data)

    # 3. 从每个 scope 组中进行采样
    # 使用 set 来确保最终输出的样本是唯一的 (因为一个样本可能被多个 scope 采样到)
    sampled_uids = set()
    sampled_data_dict: Dict[str, Any] = {}

    print(f"\n📦 发现以下 scope 类型: {', '.join(sorted(list(all_scopes)))}")
    print(f"📦 开始从每种类型中采样 {count} 条样本...")

    for scope_tag in sorted(list(all_scopes)):
        samples = scope_groups[scope_tag]

        if len(samples) < count:
            print(f"⚠️ 警告：类型 '{scope_tag}' 样本不足 ({len(samples)} < {count})，将全部采样。")
            sampled_list = samples
        else:
            # 使用 random.sample 进行不重复采样
            sampled_list = random.sample(samples, count)

        # 将采样的样本添加到最终结果字典中，并记录 UID，以保证唯一性
        for sample in sampled_list:
            uid = sample.pop('uid')
            if uid not in sampled_uids:
                sampled_uids.add(uid)
                sampled_data_dict[uid] = sample

        print(f"✅ 类型 '{scope_tag}' 采样完成 (共采样 {len(sampled_list)} 条，新增唯一样本 {len(sampled_uids)} 条)。")

    # 4. 封装结果并保存为 JSON 文件

    # 结果只包含采样的 pos 样本，结构与 ContraDOC 保持一致
    final_output = {
        "pos": sampled_data_dict,
        # ⚠️ 注意：未采样的 pos 样本和 neg 样本将被排除
    }

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # 使用 ensure_ascii=False 确保中文等非 ASCII 字符正常显示
            json.dump(final_output, f, indent=4, ensure_ascii=False)
        print(f"\n🎉 成功将 {len(sampled_uids)} 条**唯一**样本保存到文件: {output_path}")
    except Exception as e:
        print(f"❌ 保存文件时发生错误: {e}")


# --- 主执行块 ---
if __name__ == '__main__':
    sample_contradoc_by_scope(INPUT_FILE_PATH, OUTPUT_FILE_PATH, SAMPLE_COUNT_PER_SCOPE)