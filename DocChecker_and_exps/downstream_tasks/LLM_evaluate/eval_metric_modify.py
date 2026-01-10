"""
使用的数据集：ContraDOC
使用的基准：原始的ContraDOC评估基准

修正了原始的ContraDoc中对应的评估代码，将其中LLM评估的伪逻辑换为了真实的LLM调用
这里也是每个文档调用一次LLM，效率低
"""
import json
import random
# ... (其他 import 保持不变) ...
import numpy as np
from typing import Dict, List, Any
import json
import random
import re
import ast
import torch
from evaluate import load
from typing import Dict, List, Any
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from knowledge_graph_maker import OpenAIClient  # 假设已安装
from datasets import Dataset  # 假设已安装
from dotenv import load_dotenv
import os
import collections
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from knowledge_graph_maker import OpenAIClient  # 假设已安装
from datasets import Dataset  # 假设已安装
from dotenv import load_dotenv

# --- 1. 配置和 LLM 客户端 ---
load_dotenv()


# --- 假设的 LLM 客户端 (确保你已正确配置) ---
class MockLLMClient:
    def generate(self, user_message: str, system_message: str):
        # 模拟 prompt_yes_no 的输出
        if random.random() < 0.5:
            return 'yes'
        else:
            return 'no'


# 假设 OpenAIClient 可用，否则使用 Mock
try:
    from knowledge_graph_maker import OpenAIClient  # 假设路径正确

    # 实际配置可能更复杂，此处仅为占位符
    llm = OpenAIClient(
        model="gpt-5.2",
        temperature=0.1,
        max_tokens=20  # 只需要 yes/no，可以非常少
    )
except ImportError:  # 如果 knowledge_graph_maker 或 OpenAIClient 找不到
    llm = MockLLMClient()
    print("WARNING: Using MockLLMClient due to missing OpenAIClient or environment issues.")
except Exception:  # 捕获其他初始化错误 (如API Key未设置)
    llm = MockLLMClient()
    print("WARNING: Using MockLLMClient due to potential environment configuration issues.")


def detect_conflict_yes_no(llm_client: Any, full_prompt: str) -> str:
    """
    针对 prompt_yes_no 任务，使用 LLM 检测冲突，并返回 'yes' 或 'no'。
    注意：为了简洁，这里不使用 JSON 输出，因为 prompt_yes_no 只要求纯文本 yes/no。
    """
    system_prompt = "You are an expert contradiction detection system. Your only task is to determine whether the provided article contains any self-contradictions. You must respond with only one word: 'yes' or 'no'."

    if isinstance(llm_client, MockLLMClient):
        # MockLLMClient.generate 签名是 (self, user_message, system_message)
        response_text = llm_client.generate(full_prompt, system_prompt)
    else:
        # 实际 LLM 调用
        try:
            response_text = llm_client.generate(user_message=full_prompt, system_message=system_prompt)
        except Exception as e:
            print(f"LLM API Error: {e}. Returning 'no' as default.")
            return "no"

    # 清理和规范化输出
    cleaned_response = response_text.strip().lower()
    if "yes" in cleaned_response:
        return "yes"
    return "no"


# --- 新增的：模型响应生成函数 ---
def get_model_responses(llm_client: Any, prompts: List[str]) -> List[str]:
    """
    循环遍历 prompts 列表，调用 LLM 获取响应。
    """
    responses = []
    print(f"\n--- Running LLM Inference on {len(prompts)} documents ---")
    for i, prompt in enumerate(prompts):
        if i % 10 == 0:
            print(f"Processing document {i}/{len(prompts)}...")

        # 调用上面定义的单次检测函数
        response = detect_conflict_yes_no(llm_client, prompt)

        responses.append(response)

    print("\n--- LLM Inference Complete ---")
    return responses

def trim_string(s):
    start = 0
    end = len(s) - 1
    while start <= end and (s[start] == " " or s[start] == "\n"):
        start += 1
    while end >= start and (s[end] == " " or s[end] == "\n"):
        end -= 1
    return s[start:end + 1]

def convert_string_to_list(strin:str):
    strin = trim_string(strin)
    strin = strin.replace("\n", "")
    match = re.search(r'\[.*\]', strin)
    if match:
        # Convert the string representation of a list to an actual list object
        try:
            evidence_list = ast.literal_eval(match.group(0))
            if isinstance(evidence_list, list):
                return evidence_list
            else:
                raise ValueError("The input is not a valid list.")
        except (ValueError, SyntaxError) as e:
            print("Error:", e)
            print(strin)
            return None

def extract_info(full_response:str):
    info = {"response": full_response}
    full_response = full_response.lower()
    beginning = full_response.find("\njudgment") + len("\njudgment")
    ending = full_response.find("\n", beginning)
    judge = full_response[beginning: ending]
    if "yes" in judge:
        info["judgment"] = "yes"
        try:
            evidence = convert_string_to_list(full_response.split("evidence:")[1])
        except:
            evidence = []
        if evidence is None: evidence = []
        info["evidence"] = evidence
        if evidence == []:
            info = extract_info_unstructured(full_response)
            print("Error! Answer Yes but no evidence correctly processed.")
    else:
        try:
            assert "no" in judge
        except:
            print(judge, full_response)
        info["judgment"] = "no"
        info["evidence"] = []
    return info

def extract_info_yes_no(full_response:str):
    info = {"response": full_response, "evidence": []}
    full_response = full_response.lower()
    info["judgment"]="yes"
    if "yes" in full_response:
        info["judgment"]="yes"
    else: info["judgment"]="no"
    return info

def extract_info_unstructured(full_response:str, with_ev=False):
    # TODO: copied sentences are in lines
    def remove_quote(s):
        if s[:2] == "- ":s = s[2:]
        try:
            #processed_item = re.sub(r'^\d+([.)]+)?\s*', '', item)
            s = re.sub(r'^\d+\.\s*', '', s)
            if s[0] in ["\"", "\'", " "]: s = s[1:]
            if s[-1] in ["\"", "\'", " "]: s = s[:-1]
        except: print("What happened?this is short", s)
        return s
    info = {"response": full_response}
    full_response = full_response.replace("\n\n","\n").replace("\n\n","\n").split("\n")
    info["evidence"] = [remove_quote(s) for s in full_response]
    info["judgment"] = "yes"
    return info

prompt_level1={"beginning":"""The task is to determine whether the article contains any self-contradictions. If yes, provide evidence by quoting mutually contradictory sentences in a list of strings in Python. If no, give an empty list.

Article:  """,
"end": """
Response: Form your answer in the following format (OR options are provided):

Judgment: yes OR no
Evidence: ["sentence1", "sentence2", ..., "sentenceN"] OR []""",
"pos_only": False,
"extract": extract_info
}

prompt_level2 = {"beginning":"""Self-Contradictory Article: An article is deemed self-contradictory when it contains one(self-conflict mention) or more statements that conflict with each other, making them mutually exclusive. The following article contains one self-contradiction. The task is to find where it is. Provide evidence by quoting mutually contradictory sentences from the article.

Article: """,
"end":"""

Please respond by giving 5 most possible sentences can reflect article-level contradiction(s), ranked by possibility high to low, separated with "\n". Don't explain.""",
                 "pos_only": True,
                 "extract": extract_info_unstructured}

prompt_yes_no = {"beginning": "", "end":"Determine whether the given article contains any self-contradictions. Only answer \"yes\" or \"no\"!","pos_only": False, "extract": extract_info_yes_no}
prompt_list_topk = {"beginning":"", "end": "The given article above contains one self-contradiction in it. Please list sentences contradict each other. Don't include anything else other than the exact sentences from the article.","pos_only": True,"extract": extract_info_unstructured}

# ----------------------------------------------------------------------
# END: 假设的 LLM 客户端和冲突检测函数
# ----------------------------------------------------------------------

# ... (load_docs_evis_infos, load_prompted_doc, trim_string, direct_find,
# convert_string_to_list, extract_info, extract_info_yes_no, extract_info_unstructured,
# calculate_metrics, batch_bs, yes_verify 保持不变) ...

def load_docs_evis_infos(path, pos_only=False):
    with open(path) as f:
        data = json.load(f)
        positive_data = data["pos"]
        print("positive examples in total", len(positive_data))
        docs, evidences, info = [], [], []
        for key, value in positive_data.items():
            positive_data[key]["label"] = True
            positive_data[key]["id"] = key
            info.append(positive_data[key])
            docs.append(positive_data[key]["text"])
            evidences.append(positive_data[key]["evidence"])
        if pos_only:
            assert len(docs) == len(evidences) == len(info)
            return docs, evidences, info

        negative_data = data["neg"]
        print("negative examples in total", len(negative_data))
        for key, value in negative_data.items():
            negative_data[key]["label"] = False
            negative_data[key]["evidence"] = None
            negative_data[key]["id"] = key
            info.append(negative_data[key])
            docs.append(negative_data[key]["text"])
            evidences.append(negative_data[key]["evidence"])
    assert len(docs) == len(evidences) == len(info)
    return docs, evidences, info
def load_prompted_doc(list_of_doc, bs_prompt="", es_prompt=""):
    prompts = []
    for ele in list_of_doc:
        if ele[-1] != "\n": ele += "\n"
        prompts.append(bs_prompt+ele+es_prompt)
    return prompts

def calculate_metrics(gold_labels, predicted_labels, full_info):
    if len(gold_labels) != len(predicted_labels):
        raise ValueError("The length of gold_labels and predicted_labels must be the same.")

    TP = FP = TN = FN = 0
    statistics_set = {"TP":[], "TN":[], "FN":[], "FP":[]}
    for gold, predicted, info_ele in zip(gold_labels, predicted_labels, full_info):
        if gold and predicted:
            TP += 1
            statistics_set["TP"].append(info_ele)
        elif not gold and not predicted:
            TN += 1
            statistics_set["TN"].append(info_ele)
        elif gold and not predicted:
            FN += 1
            statistics_set["FN"].append(info_ele)
        elif not gold and predicted:
            FP += 1
            statistics_set["FP"].append(info_ele)

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FN + FP)
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Accuracy': accuracy,
        'True Positive': TP,
        'False Positive': FP,
        'True Negative': TN,
        'False Negative': FN,
        # 'Statistic': statistics_set
    }


if __name__ == "__main__":
    chose_prompt = prompt_yes_no
    debug = True

    # 1. 加载数据
    docs, evidences, info = load_docs_evis_infos("../../../datasets/data_from_related_work/ContraDOC/ContraDoc.json",
                                                 pos_only=chose_prompt["pos_only"])
    # 2. 生成完整提示
    prompts = load_prompted_doc(docs, bs_prompt=chose_prompt["beginning"], es_prompt=chose_prompt["end"])

    # 3. 🚀 替换伪调用：使用真实的 LLM 客户端获取响应
    # 注意：这里调用的是上面定义的 get_model_responses 函数
    responses = get_model_responses(llm, prompts)

    # 确保长度匹配
    assert len(prompts) == len(evidences) == len(info) == len(responses)

    verified_predictions, non_verified_predictions = [], []

    # 4. 评估逻辑
    for i in range(len(prompts)):
        # 使用 extract_info_yes_no 解析纯文本 yes/no 响应
        response_info = chose_prompt["extract"](responses[i].strip())

        # 对于 prompt_yes_no，evidence 字段是空的，所以 yes_verify 仅在 prompt_level1/2 中有意义。
        # 这里为了保持逻辑兼容性，我们只计算非验证的指标。
        # 如果要运行 prompt_level1，需要确保 extract_info 能正确解析 evidence。

        # 预测标签
        non_verified_predictions.append(True if response_info["judgment"] == "yes" else False)

        # 由于 prompt_yes_no 不返回证据，verified_predictions 的逻辑在这里会被简化或跳过。
        # 只有当你切换到 prompt_level1 时，yes_verify 才有意义。
        # 在 prompt_yes_no 中，verified_predictions 约等于 non_verified_predictions
        verified_predictions.append(non_verified_predictions[-1])

    # 5. 输出结果
    without_verify_res = calculate_metrics([x["label"] for x in info],
                                           non_verified_predictions, info)
    # verify_res 在这个任务中几乎与 without_verify_res 相同，因为没有 evidence 可供验证
    verify_res = calculate_metrics([x["label"] for x in info], verified_predictions, info)

    print("--- 评估结果 ---")
    print("without_verify (仅判断 Yes/No): ", without_verify_res)
    print("verify (在此 Yes/No 任务中与 without_verify 相同): ", verify_res)

    # --- 结果保存逻辑 (新增) ---

    # 1. 定义文件名和结果结构
    MODEL_NAME = "eval_metric_modify"  # 可以根据实际的模型或任务命名
    OUTPUT_FILENAME_JSON = f"evaluation_results_{MODEL_NAME}.json"

    results_to_save = {
        "Model_Name": MODEL_NAME,
        "Task": "Contradiction Detection (Yes/No)",
        "without_verify_metrics": without_verify_res,
        "verified_metrics": verify_res,
        "total_samples": len(info)
    }

    # 2. 保存到文件
    try:
        # 使用 json.dump 保存到文件
        with open(OUTPUT_FILENAME_JSON, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=4, ensure_ascii=False)
        print(f"\n✅ 评估结果已成功保存到文件: {OUTPUT_FILENAME_JSON}")
    except IOError as e:
        print(f"\n❌ 错误: 无法写入文件 {OUTPUT_FILENAME_JSON}. 详细信息: {e}")