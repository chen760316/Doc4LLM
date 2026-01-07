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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import re
from typing import List, Tuple, Optional
import json
import random
import re
import json
from collections import Counter, defaultdict
from typing import List, Tuple
from itertools import combinations

# —— 规则字符串解析成 AST 树 —— #
class RuleNode:
    def __init__(self, node_type: str, children: Optional[List['RuleNode']] = None, value: Optional[str] = None):
        self.type = node_type  # e.g. RULE, PREMISE, PREDICATE, CONCLUSION
        self.children = children or []
        self.value = value

    def __repr__(self):
        if self.value:
            return f"{self.type}({self.value})"
        else:
            return f"{self.type}({self.children})"

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


def parse_rule_to_tree(rule_str: str) -> RuleNode:
    """
    解析示例规则字符串：
    Q[x0,y0] ( p1 ∧ p2 → c )
    """
    m = re.match(r'Q\[(.*?)\]\s*\(\s*(.*)\s*→\s*(.*)\s*\)', rule_str)
    if not m:
        raise ValueError(f"Cannot parse rule string: {rule_str}")
    ctx = m.group(1).strip()
    premise_str = m.group(2).strip()
    conclusion_str = m.group(3).strip()

    premise_preds = [p.strip() for p in premise_str.split('∧')]

    premise_node = RuleNode("PREMISE", [RuleNode("PREDICATE", value=p) for p in premise_preds])
    conclusion_node = RuleNode("CONCLUSION", [RuleNode("PREDICATE", value=conclusion_str)])
    root = RuleNode("RULE", children=[premise_node, conclusion_node], value=ctx)
    return root



def tree_to_string(node: RuleNode) -> str:
    if node.type == "PREDICATE":
        return node.value
    elif node.type == "PREMISE":
        return " ∧ ".join([tree_to_string(c) for c in node.children])
    elif node.type == "CONCLUSION":
        return tree_to_string(node.children[0])
    elif node.type == "RULE":
        return f"Q[{node.value}] ({tree_to_string(node.children[0])} → {tree_to_string(node.children[1])})"
    else:
        raise ValueError(f"Unknown node type: {node.type}")


# —— 构建词汇表和编码器输入 —— #

class Vocab:
    def __init__(self):
        self.token2idx = {}
        self.idx2token = []

    def add_token(self, token):
        if token not in self.token2idx:
            self.idx2token.append(token)
            self.token2idx[token] = len(self.idx2token) - 1
        return self.token2idx[token]

    def __len__(self):
        return len(self.idx2token)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token2idx[t] for t in tokens if t in self.token2idx]

    def decode(self, indices: List[int]) -> List[str]:
        return [self.idx2token[i] for i in indices]


def tokenize_predicate(pred: str) -> List[str]:
    # 例如：拆分变量名、操作符、数字、字符串等为单独token
    tokens = re.findall(r'[A-Za-z_]+|\d+|[=!<>]+|".*?"|\(|\)|,', pred)
    return tokens


# —— Seq2Tree 模型实现 —— #

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, input_seq, input_lengths):
        embedded = self.embedding(input_seq)  # (batch, seq_len, embed_dim)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (hidden, cell) = self.lstm(packed)
        # hidden shape: (num_layers*2, batch, hidden_dim)
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (batch, hidden_dim*2)
        return hidden_cat  # 句子编码


# 树解码器用简单的递归LSTM结构，演示递归生成树

class TreeDecoder(nn.Module):
    def __init__(self, hidden_dim, vocab_size, vocab, max_children=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.vocab = vocab

        self.max_children = max_children

        self.node_type_list = ["RULE", "PREMISE", "CONCLUSION", "PREDICATE"]
        self.node_classifier = nn.Linear(hidden_dim, len(self.node_type_list))  # 节点类型分类
        self.token_classifier = nn.Linear(hidden_dim, vocab_size)  # PREDICATE节点词预测

        self.child_rnn = nn.LSTMCell(hidden_dim, hidden_dim)

    def forward(self, hidden_state, target_node: Optional[RuleNode]=None):
        """
        递归解码函数，训练时传入target_node，实现teacher forcing并计算loss。
        返回总loss和生成的RuleNode树。
        """

        # 预测当前节点类型logits
        node_logits = self.node_classifier(hidden_state)  # (1, node_type_num)
        node_log_prob = F.log_softmax(node_logits, dim=-1)

        if target_node is None:
            # 推理时用argmax预测节点类型
            node_type_id = node_logits.argmax(dim=-1).item()
        else:
            node_type_id = self.node_type_list.index(target_node.type)

        # 节点类型loss
        node_type_target = torch.tensor([node_type_id], device=hidden_state.device)
        loss_node_type = F.nll_loss(node_log_prob, node_type_target)

        # 创建节点对象
        node = RuleNode(target_node.type if target_node else self.node_type_list[node_type_id])

        total_loss = loss_node_type

        if node.type == "PREDICATE":
            # 预测词汇
            token_logits = self.token_classifier(hidden_state)  # (1, vocab_size)
            token_log_prob = F.log_softmax(token_logits, dim=-1)

            if target_node is None:
                token_id = token_logits.argmax(dim=-1).item()
                node.value = self.vocab.idx2token[token_id]
                return total_loss, node
            else:
                # target_node.value是词字符串，编码成id
                token_id = self.vocab.token2idx.get(target_node.value, 0)
                token_target = torch.tensor([token_id], device=hidden_state.device)
                loss_token = F.nll_loss(token_log_prob, token_target)
                total_loss += loss_token
                node.value = target_node.value
                return total_loss, node

        else:
            # 递归预测子节点
            hx, cx = hidden_state, torch.zeros_like(hidden_state)
            children = []
            num_children = len(target_node.children) if target_node else self.max_children
            for i in range(num_children):
                hx, cx = self.child_rnn(hx, (hx, cx))
                child_target = target_node.children[i] if target_node else None
                loss_child, child_node = self.forward(hx, child_target)
                total_loss += loss_child
                children.append(child_node)
            node.children = children
            return total_loss, node

def train_one_epoch(encoder, decoder, dataloader, optimizer, device):
    encoder.train()
    decoder.train()

    total_loss = 0.0
    for inputs_padded, lengths, roots in dataloader:
        inputs_padded = inputs_padded.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()

        encoder_out = encoder(inputs_padded, lengths)  # (batch, hidden_dim*2)

        batch_loss = 0.0
        batch_size = encoder_out.size(0)
        for i in range(batch_size):
            hidden = encoder_out[i].unsqueeze(0)  # (1, hidden_dim*2)
            target_tree = roots[i]
            loss, _ = decoder(hidden, target_tree)
            batch_loss += loss

        batch_loss /= batch_size
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# —— 简单数据集封装 —— #

class RuleDataset(Dataset):
    def __init__(self, rule_strs: List[str], vocab: Vocab):
        self.rule_strs = rule_strs
        self.vocab = vocab

    def __len__(self):
        return len(self.rule_strs)

    def __getitem__(self, idx):
        rule = self.rule_strs[idx]
        root = parse_rule_to_tree(rule)
        # 编码谓词序列作为 encoder 输入
        premise_preds = [c.value for c in root.children[0].children]
        premise_tokens = []
        for p in premise_preds:
            premise_tokens.extend(tokenize_predicate(p) + ['<sep>'])
        premise_tokens = premise_tokens[:-1]  # 去掉最后一个sep
        input_ids = self.vocab.encode(premise_tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        length = len(input_ids)
        # 目标是规则树 root，可以直接用 root 节点（这里简化，没做序列化）
        return input_ids, length, root


# —— 训练示范主函数 —— #

def collate_fn(batch):
    inputs, lengths, roots = zip(*batch)
    lengths = torch.tensor(lengths)
    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    return inputs_padded, lengths, roots

# 先定义辅助函数
def decode_tree_tokens(node: RuleNode, vocab: Vocab):
    if node.type == "PREDICATE":
        if isinstance(node.value, int):
            node.value = vocab.idx2token[node.value]
    else:
        for c in node.children:
            decode_tree_tokens(c, vocab)

def save_generated_rules(trees: List[RuleNode], vocab: Vocab, filename="generated_rules.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for tree in trees:
            decode_tree_tokens(tree, vocab)
            rule_str = tree_to_string(tree)
            f.write(rule_str + "\n")

def generate_rules(dataloader, encoder, decoder, vocab):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for inputs_padded, lengths, roots in dataloader:
            encoder_out = encoder(inputs_padded, lengths)
            for i in range(encoder_out.size(0)):
                hidden = encoder_out[i].unsqueeze(0)
                sample_tree = decoder.decode_node(hidden)
                decode_tree_tokens(sample_tree, vocab)
                rule_str = tree_to_string(sample_tree)
                print(f"Generated Rule: {rule_str}")
            break

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 读取已有谓词和规则文件
    train_preds_file = "./star_outputs/dual_star_graph_1_Emily_Clark_rico.txt"
    train_predicates, train_rules = load_predicates_and_rules(train_preds_file)

    # 读取图数据json
    with open("../star_json/dual_graph/dual_star_graph_1_Emily_Clark.json", "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    # 通过图数据生成新的谓词和规则
    x0, y0 = "x0", "y0"  # 你可根据上下文调整
    gen_predicates, gen_rules = generate_predicates_and_rules(graph_data, "doc1", "doc2", x0, y0)

    # 合并谓词和规则，去重
    all_predicates = list(set(train_predicates) | set(gen_predicates))
    all_rules = list(set(train_rules) | set(gen_rules))

    print(f"Total predicates: {len(all_predicates)}")
    print(f"Total rules: {len(all_rules)}")

    # 接下来用 all_predicates 和 all_rules 构造训练数据集和词汇表
    vocab = Vocab()
    for rule in all_rules:
        root = parse_rule_to_tree(rule)
        premise_preds = [c.value for c in root.children[0].children]
        for p in premise_preds:
            for token in tokenize_predicate(p):
                vocab.add_token(token)
            vocab.add_token('<sep>')
        conclusion_pred = root.children[1].children[0].value
        for token in tokenize_predicate(conclusion_pred):
            vocab.add_token(token)

    print(f"Vocab size: {len(vocab)}")

    dataset = RuleDataset(all_rules, vocab)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    embed_dim = 64
    hidden_dim = 128
    encoder = Encoder(len(vocab), embed_dim, hidden_dim).to(device)
    decoder = TreeDecoder(hidden_dim * 2, len(vocab), vocab).to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(encoder, decoder, dataloader, optimizer, device)
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

    # 推理示范，调用decoder递归生成树（无teacher forcing）
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for inputs_padded, lengths, roots in dataloader:
            inputs_padded = inputs_padded.to(device)
            lengths = lengths.to(device)
            encoder_out = encoder(inputs_padded, lengths)
            for i in range(encoder_out.size(0)):
                hidden = encoder_out[i].unsqueeze(0)
                _, sample_tree = decoder(hidden, target_node=None)
                decode_tree_tokens(sample_tree, vocab)
                print("Generated Rule:", tree_to_string(sample_tree))
            break

if __name__ == "__main__":
    main()
