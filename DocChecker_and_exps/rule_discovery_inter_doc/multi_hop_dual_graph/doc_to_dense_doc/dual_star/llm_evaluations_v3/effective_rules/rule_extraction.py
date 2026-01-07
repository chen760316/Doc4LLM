import os
import re
import json

# Step 1: Parse indice.txt
def parse_indice(indice_file):
    rule_sets = []
    with open(indice_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.lower() == "no rule":
                rule_sets.append([])
            else:
                # Example: "Rule 3,4,5" -> [3,4,5]
                nums = re.findall(r'\d+', line)
                rule_sets.append(list(map(int, nums)))
    return rule_sets


# Step 2: Read and split rules by blank lines
def split_rules_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='ISO-8859-1') as f:
            text = f.read()

    # 按空行分割成规则块
    rule_blocks = [block.strip() for block in re.split(r'\n\s*\n', text.strip()) if block.strip()]
    return rule_blocks


# Step 3: Load the center data from the json files
def load_center_data(center_folder):
    center_data = {}
    for filename in os.listdir(center_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(center_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    json_data = json.load(f)
                    # 假设每个文件中有一个 'center' 字段
                    if json_data and isinstance(json_data, list):
                        center = json_data[0].get('center', '')
                        center_data[filename] = center
                except json.JSONDecodeError:
                    print(f"❌ 无法解析文件 {filename} 为 JSON 格式")
    return center_data


# Step 4: Replace Q[x0, y0] with Q[center, center] in the rule content
def replace_q_with_center(rule_text, center):
    # 用正则查找并替换 Q[x0, y0] -> Q[center, center]
    return re.sub(r'Q\[x0, y0\]', f'Q[{center}, {center}]', rule_text)


# Step 5: Extract and save rules according to indice.txt and replace Q[x0, y0]
def extract_and_save_rules(indice_file, rules_folder, center_folder, output_folder):
    rule_sets = parse_indice(indice_file)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 载入 center 数据
    center_data = load_center_data(center_folder)

    # 按顺序遍历 initial 文件夹中的文件
    txt_files = sorted(
        [f for f in os.listdir(rules_folder) if f.endswith('.txt')],
        key=lambda x: int(re.search(r'(\d+)', x).group(1))  # 按编号排序
    )

    for idx, filename in enumerate(txt_files, start=1):
        file_path = os.path.join(rules_folder, filename)
        rule_blocks = split_rules_in_file(file_path)
        rule_set = rule_sets[idx - 1] if idx - 1 < len(rule_sets) else []

        # 获取当前 JSON 文件的 center 内容
        center = center_data.get(f"filled_data_{idx}.json", "")

        output_filename = f"extracted_rules_file_{idx}.txt"
        output_path = os.path.join(output_folder, output_filename)

        with open(output_path, 'w', encoding='utf-8') as out_f:
            if not rule_set:
                out_f.write("No rule selected.\n")
            else:
                for rule_num in rule_set:
                    if 1 <= rule_num <= len(rule_blocks):
                        # 获取规则并替换 Q[x0, y0] 为 Q[center, center]
                        rule_content = rule_blocks[rule_num - 1]
                        updated_rule_content = replace_q_with_center(rule_content, center)
                        out_f.write(f"Rule {rule_num}:\n{updated_rule_content}\n\n")
                    else:
                        out_f.write(f"Rule {rule_num}: No content available\n\n")

        print(f"✅ {filename} -> 提取 {len(rule_set)} 条规则, 保存为 {output_filename}")


# Main execution
if __name__ == "__main__":
    indice_file = 'indice.txt'
    rules_folder = r"E:/Document cleaning/kg-gen/document_cleaning/doc_to_llm/rule_discovery_inter_doc/multi_hop_dual_graph/doc_to_dense_doc/dual_star/llm_evaluations_v3/initial"
    center_folder = r"../../../dual_star_json_with_attrs"
    output_folder = '../../llm_evaluations_v3/effective_rules'

    extract_and_save_rules(indice_file, rules_folder, center_folder, output_folder)
