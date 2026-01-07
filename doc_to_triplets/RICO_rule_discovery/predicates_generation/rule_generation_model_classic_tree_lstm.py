"""
用 Tree-LSTM 替换 PCFG 生成规则的思路

训练阶段：
使用已有的规则（树结构）作为训练样本。
用 Tree-LSTM 编码规则的树结构，学习节点之间的组合关系和语义。
训练一个解码器（通常也是基于 Tree-LSTM 或 RNN）来生成新的规则树。

生成阶段：
给定初始条件（如起始非终结符或部分谓词），用训练好的 Tree-LSTM 解码器生成新的规则树。
通过采样策略（beam search、贪婪等）生成符合训练数据分布的新规则。

实现步骤建议
（1）准备训练数据（树结构化规则）
规则需要解析成树结构（例如，每条规则解析成抽象语法树AST）。
例如：Q[x0,y0] (p1 ∧ p2 → p3) 解析成根节点Q，左子树是前提集合，右子树是结论。

（2）定义 Tree-LSTM 编码器
参考论文：Tai et al., 2015的 Child-Sum Tree-LSTM
输入：节点特征（谓词或符号的embedding）。
输出：根节点的隐状态向量，表示整个规则。

（3）定义 Tree-LSTM 解码器
从根开始递归生成子节点，直到叶节点。
利用隐状态预测下一层节点的符号/谓词。
结合采样策略产生多条规则。

（4）训练过程
最小化解码器生成树与真实树结构之间的损失（交叉熵或结构化损失）。
用已知规则树做监督。

（5）生成规则
输入初始条件或随机向量，解码器递归生成规则树。
将树结构转换为文本规则格式。

具体说明:
1.训练阶段:
先用 load predicates and_rules 加载已有的谓词(predicates)和规则(rules)。
将规则文本用 parse_ruletotree解析成树结构(RuleTreeNode)
通过 TreeLsTMEncoder 递归编码每棵规则树，得到树的隐藏表示(隐状态向量)。
TreeLsTMDecoder 根据编码器输出的隐藏状态，学习解码生成树结构(递归生成节点符号和结构)
用交叉熵损失(nn.crossEntropyLoss)训练编码器+解码器，使其能重构训练数据的规则树结构
2生成阶段:
训练好后，给解码器一个初始隐状态(h_init ，此处是零向量)，解码器递归生成规则树(调用decode 函数)。
生成的规则树可转换成文本规则(treetorule_str)。
生成多条规则样例并打印，同时保存到文件。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import re
from typing import List, Tuple, Optional
from itertools import combinations

import re

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


# —— 数据加载函数 —— #
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

# —— 规则树结构定义 —— #
class RuleTreeNode:
    def __init__(self, symbol: str, children: Optional[List['RuleTreeNode']] = None):
        self.symbol = symbol
        self.children = children if children else []

    def __repr__(self):
        if not self.children:
            return self.symbol
        return f"{self.symbol}({', '.join(repr(c) for c in self.children)})"

# —— 词表 —— #
class Vocab:
    def __init__(self, tokens: List[str]):
        self.token2idx = {tok: i for i, tok in enumerate(tokens)}
        self.idx2token = {i: tok for i, tok in enumerate(tokens)}
        self.unk = "<UNK>"
        if self.unk not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[self.unk] = idx
            self.idx2token[idx] = self.unk

    def encode(self, token: str) -> int:
        return self.token2idx.get(token, self.token2idx[self.unk])

    def decode(self, idx: int) -> str:
        return self.idx2token.get(idx, self.unk)

    def __len__(self):
        return len(self.token2idx)

# —— TreeLSTM 单元 —— #
class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super().__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.ioux = nn.Linear(in_dim, 3 * mem_dim)
        self.iouh = nn.Linear(mem_dim, 3 * mem_dim)
        self.fh = nn.Linear(mem_dim, mem_dim)

    def forward(self, inputs, child_c, child_h):
        # inputs: (batch, in_dim) here batch usually 1
        if len(child_c) == 0:
            child_c_sum = torch.zeros(inputs.size(0), self.mem_dim, device=inputs.device)
            child_h_sum = torch.zeros(inputs.size(0), self.mem_dim, device=inputs.device)
        else:
            child_c_sum = torch.stack(child_c, dim=0).sum(0)
            child_h_sum = torch.stack(child_h, dim=0).sum(0)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.chunk(iou, 3, dim=1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)

        f_terms = []
        for h in child_h:
            f_terms.append(torch.sigmoid(self.fh(h)))
        if len(f_terms) == 0:
            fc = 0
        else:
            f_stack = torch.stack(f_terms, dim=0)
            c_stack = torch.stack(child_c, dim=0)
            fc = (f_stack * c_stack).sum(0)

        c = i * u + fc
        h = o * torch.tanh(c)
        return c, h

# —— 编码器 —— #
class TreeLSTMEncoder(nn.Module):
    def __init__(self, vocab: Vocab, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.vocab = vocab
        self.embedding = nn.Embedding(len(vocab), embed_dim)
        self.cell = ChildSumTreeLSTMCell(embed_dim, hidden_dim)

    def forward(self, node: RuleTreeNode) -> Tuple[torch.Tensor, torch.Tensor]:
        # 递归编码
        # 把 symbol 编码为索引，再 embedding
        idx = self.vocab.encode(node.symbol)
        embed = self.embedding(torch.tensor([idx]))
        child_states = [ self.forward(child) for child in node.children ]
        child_c = [c for c, h in child_states]
        child_h = [h for c, h in child_states]

        c, h = self.cell(embed, child_c, child_h)
        return c, h

# —— 解码器 —— #
class TreeLSTMDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, vocab: Vocab, max_children: int = 3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.cell = ChildSumTreeLSTMCell(embed_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.vocab = vocab
        self.max_children = max_children

        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.child_num_predictor = nn.Linear(hidden_dim, max_children + 1)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.output_layer(h)
        child_num_logits = self.child_num_predictor(h)
        return logits, child_num_logits

    def decode(self, h: torch.Tensor, depth: int = 0, max_depth: int = 4) -> Optional[RuleTreeNode]:
        if depth > max_depth:
            return None

        # 固定根节点为 Q
        root = RuleTreeNode("Q")

        # 生成前提和结论子树
        # —— PREMISE —— #
        premise_node = RuleTreeNode("PREMISE")
        num_premises = random.randint(1, 3)  # 可改为 child_num_predictor

        for _ in range(num_premises):
            # 随机选一个 predicate
            pred_idx = random.randint(0, self.vocab_size - 1)
            pred_symbol = self.vocab.decode(pred_idx)
            predicate_node = RuleTreeNode(pred_symbol)
            premise_node.children.append(predicate_node)

        # —— CONCLUSION —— #
        # 再生成结论谓词
        concl_idx = random.randint(0, self.vocab_size - 1)
        concl_symbol = self.vocab.decode(concl_idx)
        conclusion_node = RuleTreeNode(concl_symbol)

        root.children = [premise_node, conclusion_node]
        return root


# —— 规则解析函数，把规则字符串解析为树结构 —— #
def parse_rule_to_tree(rule_str: str) -> RuleTreeNode:
    """
    假设规则形如：
      Q[x0,y0] (p1 ∧ p2 → p3)
    """
    m = re.match(r'Q\[[^\]]+\]\s*\((.*)→(.*)\)', rule_str)
    if not m:
        # 如果格式不匹配，可以返回一个简单节点或者 raise
        return RuleTreeNode(rule_str.strip())
    premise = m.group(1).strip()
    conclusion = m.group(2).strip()
    # 分割前提部分
    premise_preds = [p.strip() for p in premise.split("∧") if p.strip()]
    premise_node = RuleTreeNode("PREMISE", [RuleTreeNode(p) for p in premise_preds])
    root = RuleTreeNode("Q", [premise_node, RuleTreeNode(conclusion)])
    return root

# —— 训练 epoch —— #
def train_epoch(encoder, decoder, train_trees: List[RuleTreeNode],
                optimizer, criterion_symbol, criterion_childnum, vocab: Vocab):
    encoder.train()
    decoder.train()
    total_loss = 0.0
    for tree in train_trees:
        optimizer.zero_grad()
        c, h = encoder(tree)  # encode
        # 解码部分，用根 h 作预测节点symbol 和子节点数量
        logits, child_num_logits = decoder.forward(h)
        # 标签
        target_symbol = torch.tensor([vocab.encode(tree.symbol)])
        target_child_num = torch.tensor([len(tree.children)])
        loss_symbol = criterion_symbol(logits, target_symbol)
        loss_childnum = criterion_childnum(child_num_logits, target_child_num)
        loss = loss_symbol + loss_childnum
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_trees)

# —— 工具：树转文本 —— #
def tree_to_rule_str(node: RuleTreeNode, vars: str = "x0,y0") -> str:
    if node is None:
        return ""
    if not node.children:
        return node.symbol
    if node.symbol == "Q" and len(node.children) == 2 and node.children[0].symbol == "PREMISE":
        premise_preds = [tree_to_rule_str(c) for c in node.children[0].children]
        conclusion = tree_to_rule_str(node.children[1])
        return f"Q[{vars}] ({' ∧ '.join(premise_preds)} → {conclusion})"
    return node.symbol + "(" + ", ".join(tree_to_rule_str(c) for c in node.children) + ")"

# 遍历新规则树的所有谓词节点，确保它们格式正确
def validate_tree(tree: RuleTreeNode) -> bool:
    if tree is None:
        return True
    # 用 classify_predicate 判断节点符号
    pred_type = classify_predicate(tree.symbol)
    if pred_type == "unknown":
        print(f"发现非法谓词: {tree.symbol}")
        return False
    # 递归检查所有子节点
    return all(validate_tree(child) for child in tree.children)


# —— 主流程 —— #
def main():
    # 1. 加载 predicates 和 rules
    train_preds, train_rules = load_predicates_and_rules("./star_outputs/dual_star_graph_1_Emily_Clark_rico.txt")

    # 2. 基于规则文本生成训练树
    train_trees = []
    for r in train_rules:
        tree = parse_rule_to_tree(r)
        train_trees.append(tree)

    # 3. 构造词表：包括规则里的所有符号 + special ones
    #    也可以加上你 train_preds 中的 predicate 表示为符号
    symbols = set()
    for tree in train_trees:
        def dfs(n: RuleTreeNode):
            symbols.add(n.symbol)
            for ch in n.children:
                dfs(ch)
        dfs(tree)
    # 可加 “PREMISE”,"Q" 等 if not already
    # 如果你的 predicates 有额外节点，也可以加
    symbols = list(symbols)
    vocab = Vocab(symbols)

    # 4. 初始化模型
    EMBED_DIM = 128
    HIDDEN_DIM = 128
    LR = 0.01
    EPOCHS = 50

    encoder = TreeLSTMEncoder(vocab, EMBED_DIM, HIDDEN_DIM)
    decoder = TreeLSTMDecoder(vocab_size=len(vocab), embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, vocab=vocab, max_children=3)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)
    criterion_symbol = nn.CrossEntropyLoss()
    criterion_childnum = nn.CrossEntropyLoss()

    # 5. 训练
    for epoch in range(EPOCHS):
        loss = train_epoch(encoder, decoder, train_trees, optimizer, criterion_symbol, criterion_childnum, vocab)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} Loss: {loss:.4f}")

    # 6. 生成新规则
    encoder.eval()
    decoder.eval()
    with open("./star_outputs/generated_rules_lstm_tree.txt", "w", encoding="utf-8") as f:
        with torch.no_grad():
            h_init = torch.zeros(1, HIDDEN_DIM)
            num_rules = 5  # 你想生成的规则数
            for i in range(num_rules):
                new_tree = decoder.decode(h_init, max_depth=3)
                print("*" * 100)
                print(f"\n生成的新规则树 #{i + 1}：")
                print(new_tree)
                print("对应规则文本：")
                print(tree_to_rule_str(new_tree))
                f.write(tree_to_rule_str(new_tree) + "\n")

if __name__ == "__main__":
    main()

