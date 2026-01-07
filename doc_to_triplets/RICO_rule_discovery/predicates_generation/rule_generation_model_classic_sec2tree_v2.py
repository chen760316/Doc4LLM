"""
Seq2Tree 模型通常是一个神经网络模型：输入一个序列（比如谓词序列 +上下文），输出一个树结构（AST 风格），
树结构可以代表规则的逻辑结构。你可以把规则公式解析为树（例如根节点 “RULE”，子节点 “PREMISE” 子树 + “CONCLUSION” 子树）。

✅ 核心步骤
创建训练数据：将已有规则转换为树结构 +为输入准备谓词序列（或属性描述序列）。
构建 Seq2Tree 模型：通常 encoder + tree decoder。decoder 每一步预测一个 node 类型或子结构。
训练模型：在训练集上训练参数。
推理 / 生成：给定新的谓词序列（或图谓词集合＋上下文提示），生成规则树，再转为文本格式。

说明和后续方向:

规则解析： parse_rule_to_tree() 把规则字符串转为树结构（方便 Seq2Tree 训练目标）
编码器： LSTM 双向编码谓词序列（前提谓词按顺序编码）
解码器： 用递归网络（示范用 LSTMCell 递归生成子节点）
训练流程： Dataset 读取规则，生成输入序列和目标树；DataLoader 负责批处理。
简化演示：
解码器只展示了递归预测树的思路，实际需要设计更完整的解码步骤和 loss 计算（如节点分类交叉熵，词预测交叉熵，树结构匹配等）。
你可按需要拓展解码器，实现beam search、teacher forcing、损失函数。
扩展： 你可以把 PCFG 生成的规则作为训练样本，配合这个 Seq2Tree 模型进一步强化生成规则质量。

是对rule_generation_model_classic_sec2tree.py的改进版本，可以生成有效的RICO规则，前者的生成质量低
需要先借助train_sec2tree.py生成训练文件，用于rule_generation_model_classic_sec2tree.py中的seq2tree模型的训练
"""

import torch
import pickle
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import re
import json
from collections import Counter, defaultdict
from typing import List, Tuple
from itertools import combinations

class RuleNode:
    def __init__(self, node_type, value=None):
        self.type = node_type  # e.g., "RULE", "PREMISE", "CONCLUSION", "PREDICATE"
        self.value = value  # e.g., predicate string
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self):
        if self.type == "PREDICATE":
            return f"{self.type}({self.value})"
        return f"{self.type}({', '.join(map(str, self.children))})"

class Vocab:
    def __init__(self):
        self.token2idx = {"<pad>": 0, "<unk>": 1, "<sep>": 2, "<ctx_sep>": 3, "<conclusion>": 4}
        self.idx2token = {idx: tok for tok, idx in self.token2idx.items()}
        self.next_index = len(self.token2idx)

    def add_token(self, token):
        if token not in self.token2idx:
            self.token2idx[token] = self.next_index
            self.idx2token[self.next_index] = token
            self.next_index += 1

    def build_vocab(self, sentences):
        for sent in sentences:
            tokens = tokenize_predicate(sent)
            for tok in tokens:
                self.add_token(tok)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def tokenize_predicate(pred):
    # 简单 token 分词器
    tokens = re.findall(r'\w+|[^\s\w]', pred)
    return tokens

def rule_tree_to_str(node: RuleNode) -> str:
    if node.type == "PREDICATE":
        return node.value
    elif node.type in ["PREMISE", "BODY"]:
        return " ∧ ".join([rule_tree_to_str(child) for child in node.children])
    elif node.type == "CONCLUSION":
        return rule_tree_to_str(node.children[0]) if node.children else ""
    elif node.type == "RULE":
        premise = rule_tree_to_str(node.children[0])
        conclusion = rule_tree_to_str(node.children[1])
        return f"Q[x0,y0] ({premise} → {conclusion})"
    else:
        return "UNKNOWN"

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, emb_size=128, num_layers=1):
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(input_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, input_seq, input_lengths):
        embedded = self.embedding(input_seq)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (hidden, cell) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # 双向拼接
        return hidden, outputs

class ImprovedTreeDecoder(nn.Module):
    def __init__(self, hidden_size, vocab, node_type_list=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.vocab_embedding = nn.Embedding(len(vocab.token2idx), hidden_size)
        self.node_type_list = node_type_list or ["RULE", "PREMISE", "CONCLUSION", "PREDICATE"]
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_size, len(self.node_type_list)),
            nn.ReLU()
        )
        self.token_classifier = nn.Linear(hidden_size, len(vocab.token2idx))
        self.child_rnn = nn.GRUCell(hidden_size, hidden_size)

    def forward(self, hidden_state, target_node=None):
        hidden_state = F.dropout(hidden_state, p=0.3, training=self.training)
        node_logits = self.node_classifier(hidden_state)

        if target_node:
            node_type_id = self.node_type_list.index(target_node.type)
            node_type_target = torch.tensor([node_type_id], device=hidden_state.device)
            loss_node_type = F.cross_entropy(node_logits, node_type_target)
            node_type = target_node.type
        else:
            node_probs = F.softmax(node_logits, dim=-1)
            node_type_id = torch.multinomial(node_probs, 1).item()
            node_type = self.node_type_list[node_type_id]
            loss_node_type = 0

        node = RuleNode(node_type)
        total_loss = loss_node_type

        if node_type == "PREDICATE":
            if target_node:
                target_tokens = tokenize_predicate(target_node.value)
                token_loss = 0
                current_hidden = hidden_state
                for token in target_tokens:
                    token_logits = self.token_classifier(current_hidden)
                    token_id = self.vocab.token2idx.get(token, self.vocab.token2idx['<unk>'])
                    token_target = torch.tensor([token_id], device=hidden_state.device)
                    token_loss += F.cross_entropy(token_logits, token_target)
                    token_embedding = self.vocab_embedding(token_target)
                    current_hidden = self.child_rnn(token_embedding, current_hidden)
                total_loss += token_loss / max(len(target_tokens), 1)
                node.value = ' '.join(target_tokens)
            else:
                current_hidden = hidden_state
                predicted_tokens = []
                for _ in range(15):
                    token_logits = self.token_classifier(current_hidden)
                    token_probs = F.softmax(token_logits, dim=-1)
                    token_id = torch.multinomial(token_probs, 1).item()
                    token = self.vocab.idx2token[token_id]
                    if token in ['<sep>', '<ctx_sep>', '<conclusion>']:
                        break
                    predicted_tokens.append(token)
                    token_tensor = torch.tensor([token_id], device=hidden_state.device)
                    token_embedding = self.vocab_embedding(token_tensor)
                    current_hidden = self.child_rnn(token_embedding, current_hidden)
                node.value = ' '.join(predicted_tokens)
        else:
            children = []
            num_children = len(target_node.children) if target_node else 2
            current_hidden = hidden_state
            for i in range(num_children):
                current_hidden = self.child_rnn(current_hidden, current_hidden)
                child_target = target_node.children[i] if target_node else None
                loss_child, child_node = self.forward(current_hidden, child_target)
                total_loss += loss_child
                children.append(child_node)
            node.children = children
        return total_loss, node

def classify_predicate(pred: str) -> str:
    pred = pred.strip()
    if pred.startswith("linkage"):
        return "linkage"
    elif pred.startswith("M_NLI") or pred.startswith("M_NLU") or pred.startswith("M_TEM"):
        return "ml_pred"
    elif pred.startswith("contra"):
        return "contra"
    elif re.match(r'.+_.+\..+\s*[=!<>]+', pred):  # x_y.attr op val
        return "edge_attr"
    elif re.match(r'.+\..+\s*[=!<>]+', pred):
        return "node_attr"
    elif re.match(r'.+_.+\.length\s*=', pred):
        return "edge_attr_list"
    elif re.match(r'.+\..+\s*!=\s*.+\..+', pred):
        return "node_attr_binary"
    else:
        return "unknown"

def generate_predicates_and_rules(data, doc1, doc2, x0, y0, ml_threshold=0.5, max_rules=10) -> Tuple[List[str], List[str]]:
    predicates = set()
    rules = []

    nodes_attrs = {}
    edges_attrs = {}
    text_spans = {}

    # 提取节点属性、边属性、text span
    for triple in data:
        n1 = triple["node_1"]["name"]
        n2 = triple["node_2"]["name"]

        for k, v in triple["node_1"].items():
            if k not in ["name", "text_span", "label"]:
                nodes_attrs.setdefault(n1, {})[k] = v
        for k, v in triple["node_2"].items():
            if k not in ["name", "text_span", "label"]:
                nodes_attrs.setdefault(n2, {})[k] = v

        if "text_span" in triple["node_1"]:
            text_spans[n1] = triple["node_1"]["text_span"]
        if "text_span" in triple["node_2"]:
            text_spans[n2] = triple["node_2"]["text_span"]

        edge_key = (n1, n2)
        edges_attrs.setdefault(edge_key, {})

        if "relationship" in triple and triple["relationship"]:
            edges_attrs[edge_key]["relationship"] = triple["relationship"]

        if "edge_properties" in triple:
            for ek, ev in triple["edge_properties"].items():
                edges_attrs[edge_key][ek] = ev

    # 生成 predicate 类型
    def gen_unary_preds():
        for node, attrs in nodes_attrs.items():
            for attr, val in attrs.items():
                if isinstance(val, (int, float)):
                    predicates.add(f"{node}.{attr} = {val}")
                    predicates.add(f"{node}.{attr} != {val}")
                    # 数值比较可以加偏移
                    predicates.add(f"{node}.{attr} < {val + 1}")
                    predicates.add(f"{node}.{attr} > {val - 1}")
                    predicates.add(f"{node}.{attr} <= {val}")
                    predicates.add(f"{node}.{attr} >= {val}")
                elif isinstance(val, str):
                    predicates.add(f'{node}.{attr} = "{val}"')
                    predicates.add(f'{node}.{attr} != "{val}"')

    def gen_binary_node_preds():
        nodes = list(nodes_attrs.keys())
        for x, y in combinations(nodes, 2):
            attrs_x = nodes_attrs[x]
            attrs_y = nodes_attrs[y]
            shared = set(attrs_x.keys()) & set(attrs_y.keys())
            for attr in shared:
                # 类型要相同
                if type(attrs_x[attr]) != type(attrs_y[attr]):
                    continue
                if isinstance(attrs_x[attr], (int, float)):
                    predicates.add(f"{x}.{attr} != {y}.{attr}")
                elif isinstance(attrs_x[attr], str):
                    predicates.add(f'{x}.{attr} != {y}.{attr}')

    def gen_unary_edge_preds():
        for (x, y), attrs in edges_attrs.items():
            for attr, val in attrs.items():
                if attr == "text_span":
                    continue
                if isinstance(val, (int, float)):
                    predicates.add(f"{x}_{y}.{attr} = {val}")
                    predicates.add(f"{x}_{y}.{attr} != {val}")
                    predicates.add(f"{x}_{y}.{attr} < {val + 1}")
                    predicates.add(f"{x}_{y}.{attr} > {val - 1}")
                    predicates.add(f"{x}_{y}.{attr} <= {val}")
                    predicates.add(f"{x}_{y}.{attr} >= {val}")
                elif isinstance(val, str):
                    predicates.add(f'{x}_{y}.{attr} = "{val}"')
                    predicates.add(f'{x}_{y}.{attr} != "{val}"')
                elif isinstance(val, list):
                    predicates.add(f"{x}_{y}.{attr}.length = {len(val)}")

    def gen_binary_edge_preds():
        edges = list(edges_attrs.keys())
        for (ex1, ey1), (ex2, ey2) in combinations(edges, 2):
            # 如果两个 relationship 存在并且不同
            rel1 = edges_attrs[(ex1, ey1)].get("relationship")
            rel2 = edges_attrs[(ex2, ey2)].get("relationship")
            if rel1 and rel2 and rel1 != rel2:
                predicates.add(f'{ex1}_{ey1}.relationship != {ex2}_{ey2}.relationship')

    def gen_contra_preds():
        # 简单做两节点 text_span 对比
        for x, y in combinations(text_spans.keys(), 2):
            span_x = text_spans[x]
            span_y = text_spans[y]
            # 假设 doc1/doc2 是标识 doc 来源
            # 这里只示例，不一定语义正确
            contra_pred = f'contra("{span_x}", "{span_y}")'
            predicates.add(contra_pred)

    def gen_ml_preds():
        # 示例：如果某节点属性向量相近或不同，就加 ML 型谓词
        for x, y in combinations(nodes_attrs.keys(), 2):
            # 这里 ml_score 模拟
            # 你应当替换成你真正的 ML 模型计算函数
            attrs_x = nodes_attrs[x]
            attrs_y = nodes_attrs[y]
            # 简单 heuristic：如果属性重叠少，则认为 score 小
            overlap = set(attrs_x.items()) & set(attrs_y.items())
            total = max(len(attrs_x), len(attrs_y), 1)
            score = len(overlap) / total
            if score < ml_threshold:
                predicates.add(f'M_NLI({x}.attrs, {y}.attrs)')
                predicates.add(f'M_NLU({x}.attrs, {y}.attrs)')
                predicates.add(f'M_TEM({x}.attrs, {y}.attrs)')

    # 执行生成
    gen_unary_preds()
    gen_binary_node_preds()
    gen_unary_edge_preds()
    gen_binary_edge_preds()
    gen_contra_preds()
    gen_ml_preds()

    # 清理谓词
    pred_list = list({p for p in predicates if p and isinstance(p, str)})

    # 输出 statistics
    stats = defaultdict(int)
    for p in pred_list:
        t = classify_predicate(p)
        stats[t] += 1
    print("📊 Predicate Type Statistics:")
    for t, cnt in stats.items():
        print(f" - {t}: {cnt}")

    # 生成规则
    rules_out = []
    used_heads = set()

    # 第一阶段：让每种类型至少一个 head
    type_buckets = defaultdict(list)
    for p in pred_list:
        t = classify_predicate(p)
        type_buckets[t].append(p)

    for t, bucket in type_buckets.items():
        if not bucket:
            continue
        head = random.choice(bucket)
        # 选择 body（至少一个谓词，不是 head）
        body_cands = [p for p in pred_list if p != head]
        if not body_cands:
            continue
        body = random.choice(body_cands)
        rule = f"Q[{x0},{y0}] ({body} ∧ {head} → {head})"
        rules_out.append(rule)
        used_heads.add(head)
        if len(rules_out) >= max_rules:
            break

    # 第二阶段：剩余规则
    all_pairs = list(combinations(pred_list, 2))
    random.shuffle(all_pairs)
    for (p1, p2) in all_pairs:
        if len(rules_out) >= max_rules:
            break
        head = p2
        if head in used_heads:
            continue
        # body 用 p1 ∧ p2 或两个不同类型
        rule = f"Q[{x0},{y0}] ({p1} ∧ {p2} → {head})"
        rules_out.append(rule)
        used_heads.add(head)

    return pred_list, rules_out

def load_predicates_and_rules(file_path: str) -> Tuple[List[str], List[str]]:
    predicates = []
    rules = []
    mode = None
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == "Predicates:":
                mode = "pred"
                continue
            elif line == "Rules:":
                mode = "rules"
                continue
            if not line or mode is None:
                continue
            if mode == "pred":
                predicates.append(line)
            elif mode == "rules":
                rules.append(line)
    return predicates, rules

def main():
    # ==== 1. 加载训练数据（用于 PCFG） ====
    train_preds, train_rules = load_predicates_and_rules("./star_outputs/dual_star_graph_1_Emily_Clark_rico.txt")

    # ==== 2. 加载目标图数据（用于生成新谓词） ====
    with open("../star_json/dual_graph/dual_star_graph_1_Emily_Clark.json", "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    # ==== 3. 基于图生成谓词（不生成规则） ====
    predicates_new, _ = generate_predicates_and_rules(
        data=graph_data,
        doc1="doc1", doc2="doc2", x0="x0", y0="y0",
        ml_threshold=0.5,
        max_rules=0  # 👈 只生成谓词
    )

    print("\n🧠 新生成的谓词：")
    for p in predicates_new:
        print(" -", p)

    # ==== 4. 构造 sec2tree 模型输入 ====
    input_sequence = " ; ".join(predicates_new)
    input_tokens = tokenize_predicate(input_sequence)

    vocab = Vocab.load("./star_outputs/vocab.pkl")
    input_ids = [vocab.token2idx.get(tok, vocab.token2idx["<unk>"]) for tok in input_tokens]
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    input_lengths = torch.tensor([len(input_ids)])

    # ==== 5. 加载训练好的模型 ====
    encoder = EncoderRNN(len(vocab.token2idx), hidden_size=256)
    decoder = ImprovedTreeDecoder(hidden_size=256 * 2, vocab=vocab)  # hidden_size * 2 because encoder is bidirectional

    encoder.load_state_dict(torch.load("./star_outputs/encoder.pth", map_location="cpu"))
    decoder.load_state_dict(torch.load("./star_outputs/decoder.pth", map_location="cpu"))

    encoder.eval()
    decoder.eval()

    # ==== 6. 使用 sec2tree 解码规则树 ====
    num_generate = 5  # 👈 想生成几条规则
    print(f"\n🌳 使用 sec2tree 随机生成 {num_generate} 条规则：")

    with open("./star_outputs/generated_rules_sec2tree.txt", "w", encoding="utf-8") as f_out:
        with torch.no_grad():
            for i in range(num_generate):
                hidden, _ = encoder(input_tensor, input_lengths)
                _, rule_tree = decoder(hidden, target_node=None)
                rule_str = rule_tree_to_str(rule_tree)
                print(f"[Rule #{i + 1}] {rule_str}")
                f_out.write(f"{rule_str}\n")


    # ==== 7. 可选：用 PCFG 采样新规则 ====
    # 可添加 PCFGGenerator(predictes_new, train_rules).generate_rules() 等逻辑
    # 留给你补

if __name__ == "__main__":
    main()
