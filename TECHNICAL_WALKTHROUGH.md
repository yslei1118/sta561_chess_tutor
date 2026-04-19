# Chess Tutor 项目技术报告

> STA 561D Probabilistic Machine Learning Final Project, Duke University
>
> 本文档完整描述项目从问题定义到最终结果的每一步实现过程，包括方法选择的理由、具体的设计细节和实验结论。适合尚未接触本项目的读者阅读。

---

## 1. 问题定义

### 1.1 现实痛点

国际象棋引擎（如 Stockfish）是世界上最强的棋手，但它是一个糟糕的老师。

当一个 ELO 1200 的初学者问"我这步为什么不好？"，Stockfish 的回答是 `Nf5 (+2.3)`——一个走法加一个分数，和它给 ELO 2000 的高手说的完全一样。初学者不知道为什么 Nf5 好，也没办法在下一盘棋中复现这个思路。

一个好的人类教练会做不同的事：对初学者说"你的马没有保护，会被白吃"，对高手说"这步可行但长期会削弱你的王翼结构"。教练会**根据学生水平**调整建议的内容和表达方式。

### 1.2 我们要做什么

构建一个自适应的国际象棋教练系统：

1. **理解不同水平人类的走法**——给定棋盘和目标 ELO，预测该水平的人最可能走什么
2. **生成可理解的反馈**——用学生能听懂的语言解释局面
3. **学习哪种反馈最有效**——自动选择最适合当前学生和当前局面的反馈类型
4. **提供交互式体验**——用户可以在 Notebook 中评估局面和与 bot 对弈

### 1.3 课程的 A+ 要求

教授明确要求两个功能才能拿 A+：

- 用户可以设置任意棋局，让教练按指定 ELO 给出评估
- 用户可以和 bot 对弈，bot 要有 running commentary

---

## 2. 数据采集与处理

### 2.1 数据来源

我们使用了 Lichess 开源数据库（https://database.lichess.org/），选取了 2013 年 1 月的数据。选这个月的原因：文件大小适中（解压后 ~93MB），包含足够多的低段和中段对局。

原始数据是 PGN 格式（Portable Game Notation），每局棋记录了：
- 双方 ELO 分数
- 完整的走法序列
- 胜负结果

### 2.2 ELO 分段设计

我们将数据按 ELO 划分为 5 个段：

| 段位 | 范围 | 代表水平 |
|------|------|---------|
| 1100 | 1050–1150 | 纯初学者 |
| 1300 | 1250–1350 | 有基础 |
| 1500 | 1450–1550 | 业余中等 |
| 1700 | 1650–1750 | 业余强手 |
| 1900 | 1850–1950 | 准专业 |

每段的宽度是 ±50 ELO。太宽（比如 ±200）会导致段内差异大，太窄则数据量不够。±50 在这两个需求之间取得平衡。

### 2.3 局面采样策略

不是每一步都拿来训练的。我们做了三个过滤：

1. **跳过前 4 步**——开局前几步几乎所有人都走一样的（1.e4 或 1.d4），没有区分度
2. **跳过第 40 步之后**——残局数据在各段分布不均匀
3. **每 5 步采一次**——连续的局面高度相关，每 5 步采一次减少冗余

最终得到约 10 万个训练局面。

### 2.4 实现细节

数据处理的 pipeline 是：

```
PGN 文件 → parse_pgn.py（逐局解析，记录 FEN/走法/ELO）
         → extract_features.py（提取 40 维特征向量）
         → dataset.py（组装成 numpy 数组，保存到 data/processed/）
```

代码位于 `chess_tutor/data/` 目录下。

---

## 3. 特征工程

### 3.1 设计思路

我们没有使用神经网络端到端学特征，而是手工设计了 40 维特征向量。原因有三：

1. **可解释性**——教练需要说清楚"为什么你这步不好"，手工特征让我们能直接指出"你的王安全分数很低"
2. **先例**——2025 年 BP-Chess 论文证明手工特征 + 简单分类器在走法预测上可以和 Maia 这类深度模型打平
3. **可复现性**——在 CPU 上跑得动，教授在笔记本电脑上就能验证结果

### 3.2 30 维棋盘特征

每个棋盘局面提取以下 30 个数值：

**子力统计（12 维）**：白方和黑方各有多少个兵、马、象、车、后、王。这是最基本的局面信息——子力是否平衡直接决定了局势。

**物质差（1 维）**：用 centipawn（百分之一兵的价值）衡量的子力差。子力价值为：兵=100, 马=320, 象=330, 车=500, 后=900。计算方式：

```math
\text{material\_balance} = \sum_{p} \text{count}_\text{white}(p) \cdot v(p) - \sum_{p} \text{count}_\text{black}(p) \cdot v(p)
```

**机动性（1 维）**：当前方有多少合法走法。走法越多说明棋子越活跃，局面越灵活。

**王安全（2 维）**：
- 兵盾分数：王前面有几个保护性的兵（0-3）
- 王区域攻击：对方有多少次攻击落在王周围 8 格

这两个特征直接反映了王是否安全——初学者最常犯的错误之一就是在王不安全时发起进攻。

**中心控制（1 维）**：e4/d4/e5/d5 四个中心格被己方棋子占据或攻击的程度。中心控制是开局和中局的核心概念。

**兵形结构（4 维）**：
- 孤立兵数量——没有同色兵在相邻列保护的兵
- 叠兵数量——同一列有两个或以上己方兵
- 通路兵数量——前方没有对方兵阻挡的兵
- 兵岛数量——兵被分成几组

兵形是战略评估的核心，高手对兵形的理解远超初学者。

**出子程度（1 维）**：有多少个马和象已经离开了起始位置。出子是开局的首要目标。

**王车易位权（4 维）**：四个二值特征，表示白方/黑方是否还能进行王翼/后翼王车易位。

**比赛阶段（3 维）**：开局/中局/残局的 one-hot 编码。通过棋盘上剩余子力总量判断：

```math
\text{phase} = \begin{cases} \text{opening} & \text{if total\_material} > 6200 \\ \text{middlegame} & \text{if total\_material} > 3200 \\ \text{endgame} & \text{otherwise} \end{cases}
```

**悬挂子（1 维）**：有多少个己方棋子既被攻击又没有保护——这是失误的直接指标。

### 3.3 10 维走法特征

针对具体走法再补充 10 个特征：
- 是否吃子（1 维）
- 是否将军（1 维）
- 走的棋子类型（6 维 one-hot：兵/马/象/车/后/王）
- 这步的 centipawn 损失（1 维，用 Stockfish 标注）
- 归一化的步数（1 维，当前步数/80）

### 3.4 代码实现

核心函数在 `chess_tutor/data/extract_features.py`：

- `extract_board_features(board)` → 返回 30 维 numpy 数组
- `extract_move_features(board, move)` → 返回 10 维 numpy 数组
- `detect_game_phase(board)` → 返回 "opening"/"middlegame"/"endgame"

每个函数都直接操作 `python-chess` 的 `chess.Board` 对象。

---

## 4. 走法预测模型

### 4.1 核心任务

输入：一个棋盘局面 + 一个目标 ELO 分数
输出：该 ELO 水平的人最可能走的前 k 步

这不是要预测"最佳走法"，而是预测"人类走法"——不同水平的人面对同一个局面会做不同的选择。

### 4.2 数据组织方式

我们把走法预测转化为 binary classification（candidate ranking）问题：

对每个训练局面：
1. 列出所有合法走法（通常 25-35 步）
2. 实际走的那步标为 label=1，其余标为 label=0
3. 每步的特征 = 30 维棋盘特征 + 10 维走法特征 = 40 维

这样一个局面产生 ~30 条训练数据。10 万局面 → ~300 万条训练数据。

预测时，对所有合法走法都算一遍 P(human_played | features)，按概率从高到低排序，取 top-k。

脚本 `scripts/build_candidate_dataset.py` 负责构建这个数据集。

### 4.3 三种架构

#### Architecture A：每段一个模型（Per-bracket）

思路最简单：把 1100 段的数据只喂给 1100 段的模型，1300 段的数据只喂给 1300 段的模型。预测时根据目标 ELO 找最近的那个模型。

这是 Maia 论文（McIlroy-Young et al., KDD 2020）的思路。优点是每个模型只需要拟合一个段位的分布。缺点是只能在 5 个离散 ELO 点上查询——如果目标是 1640 分，只能用 1700 段的模型近似。

#### Architecture B：单一模型加 ELO 特征（Pooled）

把所有段的数据合并，训练一个模型，把归一化的 ELO 值作为第 41 个特征追加进去。理论上模型可以学到"当 ELO 特征偏低时，倾向于预测更简单的走法"。

实际效果中等——模型倾向于忽略 ELO 特征，因为棋盘特征的预测力更强。

#### Architecture C：Nadaraya-Watson 核插值（连续 ELO 查询的方法论贡献）

训练阶段和 Architecture A 一样——每段一个独立的 Random Forest。差别在预测阶段：不是找最近的一个模型，而是用高斯核把 5 个模型的预测做加权平均。

具体来说，给定目标 ELO $s^*$，每个段位中心 $s_k$ 的权重由高斯核决定：

```math
w_k = \frac{K_h(s^* - s_k)}{\sum_j K_h(s^* - s_j)}, \quad K_h(d) = \exp\!\left(-\frac{d^2}{2h^2}\right)
```

其中 $h$ 是 bandwidth（带宽）参数。最终的插值预测为：

```math
P(m \mid x,\, s^*) = \sum_k w_k \cdot P_k(m \mid x)
```

**直觉**：离目标 ELO 越近的段位权重越大。如果 bandwidth 很小，退化为最近邻（=Architecture A）；如果 bandwidth 很大，所有段权重相等（=忽略 ELO 差异）。最优 bandwidth 在中间。

**Bandwidth 选择**：用 leave-one-bracket-out 交叉验证、candidate-ranking 评估（`NadarayaWatsonELO.select_bandwidth_cv(..., pos_idx=played_pos)`）。我们遍历 bandwidth ∈ {25, 50, 75, 100, 150, 200, 300}。

**真实 CV 结果** — 每个 bandwidth 下的 **mean ± 1 SEM**（5-fold LOBO，per-fold std ≈ 0.008，SEM ≈ 0.004）：

| bandwidth | mean top-1 | 1 SEM |
|-----------|-----------|-------|
| 25  | 0.1522 | ± 0.004 |
| 50-100  | 0.1522-0.1523 | ± 0.004 |
| 150 | 0.1527 | ± 0.004 |
| 200 | 0.1539 | ± 0.004 |
| **300** | **0.1545** | ± 0.004 |
| 500 | 0.1541 | ± 0.004 |
| 1000 | 0.1539 | ± 0.004 |

CV point-estimate 选 **h = 300**，但注意：h ∈ {150, 200, 300, 500, 1000} 的 mean 全部在 **一个 standard error** 内，统计上不可区分。实际含义是：**任何足够宽的 kernel 都 OK**——当 bandwidth 远大于 bracket 间距时，核权重趋于均匀，Architecture C 退化成 per-bracket 模型的简单平均。这一观察与 §4.6 ablation 中 B (pooled training) 优于 A/C 的结果一致：pooled-style 聚合本身就是最有效的（B 直接用单个更大训练集，C bw→∞ 隐式做同样的聚合）。

（早期文档错误地称 "CV 选出 h=100"——那是当 `select_bandwidth_cv` 使用 argmax-on-binary-output 而非正确的 ranking 评估时错误得到的值。我们修复了该函数并重新运行。）

**C 的核心价值**：不在于 top-1 最高（在 ablation 里 C 的 top-1 与 A、B 接近，详见 §4.6），而在于它提供了**连续 ELO 插值**的能力——能回答 ELO=1640 这样不在训练段上的查询，而 A 只能做到最近邻近似。这是一个 inference-time 的方法论扩展，不是一个"更准"的模型。

这个方法的代码在 `chess_tutor/models/kernel_interpolation.py`。核心类 `NadarayaWatsonELO` 实现了 `kernel()`、`kernel_weights()`、`interpolate()` 和 `select_bandwidth_cv()` 四个方法。

### 4.4 分类器选择

我们用 Random Forest 作为每个段的基础分类器：

```python
RandomForestClassifier(
    n_estimators=500,    # 500 棵决策树
    max_depth=20,        # 最大深度 20
    min_samples_leaf=5,  # 叶子最少 5 个样本
    random_state=42      # 固定随机种子
)
```

为什么用 RF 而不是 XGBoost、SVM 或神经网络？

- RF 训练速度快（CPU 几分钟），可复现性强
- Feature importance 直接可读——方便做可解释性分析
- 我们做了 ablation：RF vs GBT vs LogReg vs Ridge，RF 在 top-1 和 top-5 准确率上综合最好

### 4.5 数据划分

**按 position 划分**——同一个局面的所有 candidate moves 要么全在训练集要么全在测试集。如果按 row 随机切分，同一个局面的正例（实际走法）可能出现在训练集而负例在测试集，造成数据泄漏。

切分比例：80% 训练 / 20% 测试，用 `RandomState(42)` 固定。

### 4.6 结果

**Ablation 实测 top-1（来自 `results/ablation_table.csv`）**：

| Architecture | Classifier | Top-1 |
|--------------|-----------|-------|
| **B (Pooled+ELO)** | RF | **0.1582** ← 最高 |
| C (Kernel, bw=200) | RF | 0.1533 |
| C (Kernel, bw=150) | RF | 0.1532 |
| C (Kernel, bw=75) | RF | 0.1522 |
| C (Kernel, bw=300, CV-selected) | RF | 0.1545 |
| A (Per-bracket) | RF | 0.1516 |
| C (Kernel, bw=50) | RF | 0.1514 |
| A (Per-bracket) | GBT | 0.1477 |
| A (Per-bracket) | LogReg | 0.1246 |

**诚实的 takeaway**：

1. **B (pooled + ELO feature) 在 top-1 上最好**（0.1582）。这出乎意料但是真实数据。原因可能是：pooled model 能看到全部 ELO 段的信息互相 regularize，而 A/C 按段训练导致每个 sub-model 的样本量更少。
2. **C 的各 bandwidth 与 A 非常接近**（0.1514–0.1533 区间），最高的 C（bw=200）比 A 只高 1.1%。核插值在 pure top-1 上没有给出显著提升。
3. **CV 选出的 bw=300 是候选集中最大的**——kernel 几乎变 uniform，C 退化为近似 pooled model。这和 B 在 top-1 上超过 C 的结果一致。
4. **Top-5 的差距比 top-1 小**，per-bracket 范围 **34.7-42.9%**。

**关于 top-1 只有 ~15% 的解释**：每个局面平均有 30+ 合法走法，纯随机猜中的概率是 ~3%。我们达到了 15%，是随机的 5 倍。更重要的指标是 top-5——教练不需要猜中学生的唯一走法，只需要缩小候选集以生成相关反馈。Maia 使用深度 CNN 在上百万对局上训练达到 50% top-1，我们用 RF + ~25k 局面（candidate ranking 展开后约 73 万行）达到 15% 是合理的。

**Cross-ELO 混淆矩阵**：5 个段里有 4 个段呈现对角占优——用同段模型预测准确率最高。这证明模型确实区分了不同水平，而不是对所有水平做相同的预测。

**所以我们选 C 作为主要 "contribution" 的理由是方法论**（连续 ELO 查询）**而非 pure accuracy**。如果 final bot 只需要在训练段上查询，B 是更简单且更强的选择；C 的价值体现在 1640、1750 这种非离散 ELO 的应用场景。

训练和评估的完整脚本在 `scripts/train_and_evaluate.py` 和 `scripts/run_final_experiment.py`。

---

## 5. Blunder 分析（Negative Finding）

### 5.1 尝试与结果

我们尝试用前面的 40 维手工特征直接训练一个 RF 二分类器，预测某一步是否为 blunder（定义：Stockfish depth-12 评估走后相对走前 cp_loss > 100）。

为了得到诚实的 ground truth，我们在 22,712 个真实走法上跑了 Stockfish（depth 12，8 进程并行约 7 分钟），保存在 `data/processed/real_cp_losses.npy` 和 `real_blunder_labels.npy`。

**结果 AUC ≈ 0.51**——接近随机。去掉 `mobility` 和 `hanging_pieces` 两维再跑，AUC 几乎不变。ELO 单特征的 AUC 是 0.511，ELO + mobility + hanging 的 AUC 是 0.502。

### 5.2 诚实的解读

这是一个**有价值的 negative finding**：

- Blunder 的本质是**局面特定的战术错觉**——"看起来赢子但其实被反将杀"这种东西必须通过实际搜索（engine lookahead）才能识别，不是静态特征能刻画的。
- 我们验证了"handcrafted features can predict blunders"这个常见假设不成立，提醒了 handcrafted approach 的能力边界。
- 相反，**cp_loss 作为连续信号是稳定的**：每段的 mean cp_loss ≈ 50，blunder rate ≈ 10%，反映了 Lichess 对手匹配系统让每个水平玩家的"错误暴露度"保持一致。

### 5.3 结论

我们**没有**把 blunder detection 作为系统的一个功能模块。这部分工作的产出是：

1. 一个诚实的 cp_loss 数据集（22,712 个走法真实标注）
2. 一个 negative finding，帮我们约束 feedback 生成系统的 claim
3. 真实数据显示各 ELO 段 blunder rate ≈ **10%**，而我们 simulator 的 `blunder_prob = 0.3` **偏高**（高出 3 倍）。提醒我们在未来的 pilot 中应该把这个 default 降到 0.10-0.15 左右以匹配真实分布

---

## 6. 反馈生成系统

### 6.1 反馈类型设计

我们定义了 7 种反馈类型，覆盖了国际象棋教学的主要维度：

| 编号 | 类型 | 适用场景 | 示例 |
|------|------|---------|------|
| F1 | Tactical Alert | 局面存在战术机会 | "你的棋子可以攻击两个目标" |
| F2 | Strategic Nudge | 需要改善长期布局 | "考虑改善你最差的那个子的位置" |
| F3 | Blunder Warning | 学生刚犯了或即将犯失误 | "这步看起来自然但会丢子" |
| F4 | Pattern Recognition | 局面符合经典模式 | "这个局面有典型的中局结构" |
| F5 | Move Comparison | 学生走法 vs 建议走法 | "你的走法和引擎推荐的差异在于..." |
| F6 | Encouragement | 学生走了好棋 | "好棋！这正是引擎的第一选择" |
| F7 | Simplification | 复杂局面可以简化 | "考虑兑换进入简化残局" |

这 7 个类型的 enum 定义在 `chess_tutor/feedback/taxonomy.py`。

### 6.2 ELO 自适应模板

同一种反馈类型，对不同 ELO 的学生用不同的语言：

**ELO 1100（初学者）**——使用简单词汇，直接指令，不用术语：
> "提示：你有一个棋子可以同时攻击两个目标，你能找到吗？"

**ELO 1500（中等）**——可以用一些术语，需要学生自己思考：
> "考虑某个棋子到关键位置的强制走法——这里有一个战术机会。"

**ELO 1900（高手）**——使用完整术语，期望具体分析：
> "局面需要精确计算——有一个战术序列从某步开始。"

模板系统在 `chess_tutor/feedback/templates.py`。

### 6.3 动态反馈生成

反馈文本不是完全静态的模板——`generator.py` 里的 `FeedbackGenerator` 会根据实际局面填充具体内容：

1. `_find_key_piece_and_square()` ——在局面中找到最重要的棋子和格子（比如悬挂的马在 e5）
2. `_detect_tactic_type()` ——判断是否存在 fork、pin、discovered attack
3. `_observe_position()` ——观察局面的关键特征（物质优势、mobility 低等）
4. `_suggest_plan()` ——根据局面特征建议下一步计划

选择反馈类型的逻辑也是动态的。`select_best_feedback_type()` 给每种类型打分：

- 如果有将军或吃子机会 → Tactical Alert 加分
- 如果有悬挂子 → Blunder Warning 加分
- 如果是残局 → Simplification 加分
- 如果 mobility 高、无悬挂子 → Encouragement 或 Strategic Nudge 加分
- 最后加一点随机噪声，避免每次都选同一个类型

---

## 7. Contextual Thompson Sampling

### 7.1 问题形式化

每次给学生反馈时，我们面临一个选择：7 种反馈类型里选哪一个？

这个问题可以建模为 **contextual multi-armed bandit**：

- **Arms（臂）**：7 种反馈类型
- **Context（上下文）**：描述当前"学生 × 局面"状态的特征向量
- **Reward（奖励）**：反馈是否有效（反馈类型与局面/学生状态的匹配度）
- **目标**：最小化 cumulative regret（累计遗憾），即与最优策略的差距

### 7.2 Context 向量设计

20 维上下文向量包含四类信息（定义在 `chess_tutor/teaching/context.py`）：

**棋盘信息 [0:4]**：物质差（归一化）、mobility（归一化）、王安全兵盾、王安全攻击数

**位置信息 [4:6]**：局面复杂度（合法走法数 / 40）、失误概率估计

**学生信息 [6:14]**：
- 学生 ELO（归一化到 [0,1]）
- 最近 10 步的平均 cp_loss
- 三个弱点分数（tactics/strategy/endgame）
- 步数、时间压力、累计失误次数

**学生趋势 [14:20]**：improving/stable/declining 的 one-hot 编码、胜率估计、两个保留位

### 7.3 Thompson Sampling 算法

每个 arm $a$ 维护一个 Bayesian 线性回归模型的后验分布：

```math
r = \theta_a^\top x + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0,\, \sigma^2)
```

后验参数的更新规则：

```math
B_a = I + \sum_{t} x_t x_t^\top, \quad f_a = \sum_{t} r_t \cdot x_t, \quad \hat{\mu}_a = B_a^{-1} f_a
```

```math
\theta_a \sim \mathcal{N}\!\left(\hat{\mu}_a,\; v^2 B_a^{-1}\right)
```

**选择 arm**（每次交互时）：

```math
\tilde{\theta}_a \sim \mathcal{N}(\hat{\mu}_a,\; v^2 B_a^{-1}), \quad a^* = \arg\max_a \; \tilde{\theta}_a^\top x
```

**更新后验**（观察到 reward $r$ 后）：

```math
B_a \leftarrow B_a + x x^\top, \quad f_a \leftarrow f_a + r \cdot x, \quad \hat{\mu}_a \leftarrow B_a^{-1} f_a
```

这是 Agrawal & Goyal (ICML 2013) 的 Linear Thompson Sampling。理论 regret bound 为 $R(T) = \tilde{O}(d\sqrt{T})$，是 sublinear 的——意味着平均每步的遗憾趋于零。

实现在 `chess_tutor/teaching/bandit.py`。

### 7.4 Reward 设计

我们设计了两种 reward 函数：

**Analytic reward**（`teaching/reward.py`）：用于理论分析

```math
r = 0.5 \cdot \sigma(\Delta\text{cp}) + 0.3 \cdot \mathbb{1}[\text{blunder\_avoided}] + 0.2 \cdot \mathbb{1}[\text{continued\_play}]
```

需要 Stockfish 标注的 cp_loss，live play 中不可用。

**Simulation reward**（`simulation/runner.py::_empirical_reward`）：**纯 empirical，无 context-arm alignment**

```math
r = \underbrace{\max\!\left(0,\; 1 - \frac{\text{cp\_loss}}{200}\right)}_{\text{empirical only}} + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, 0.05^2)
```

关键：reward 公式里**没有 arm**。`_empirical_reward(context, arm, cp_loss)` 里 `context` 和 `arm` 两个参数在函数体第一行就 `del context, arm` 显式不用。

Arm 如何影响 reward？**通过 student simulator 的 concept 动态**，不是通过 reward 公式：

1. Bandit 选择 arm （比如 TACTICAL_ALERT）
2. `StudentSimulator.update_state(arm=TACTICAL_ALERT, ...)` 提升学生的 `tactics` mastery
3. 下一回合 `_sample_real_cp_loss(student, board)` 按 **当前局面相关的 concept 的 mastery** 选 percentile：
   - 如果下一个局面是战术性的（`is_capture` count ≥ 2 或 is_check），用 `tactics` mastery
   - 如果是残局，用 `endgame` mastery
   - 如果是开局，用 `opening + strategy` mastery
4. 提升了的 tactics mastery → 战术局面 cp_loss 的 percentile 下降 → 更低的 cp_loss 样本 → 更高的 reward

这整个链路是 **realistic pedagogical mechanism**：
- Arm → 某 concept 的 mastery 提升（`FEEDBACK_CONCEPT_MAP` 给出映射）
- 下回合相关 concept 主导 cp_loss 采样
- cp_loss 变低 → reward 变高

Reward 函数**完全不引用 arm**。Bandit 唯一能影响 reward 的通道是 simulator 的 concept 动态。

**历史沿革**：
- v1 (original): `_context_dependent_reward` = `0.4 base + 0.6 alignment(ctx, arm) + noise`，arm 在 reward 公式里硬编码——self-referential
- v2: 重命名为 `_pedagogical_prior_reward`，base 换成 empirical cp_loss，但 alignment 项保留——只修了一半
- **v3 (current)**: `_empirical_reward`，alignment 项完全删除；arm differentiation 通过 simulator 的 concept dynamics

**仍存在的假设**：
- `FEEDBACK_CONCEPT_MAP`（在 `chess_tutor/feedback/taxonomy.py`）：哪个 feedback type 提升哪个 concept
- `_relevant_concepts_for_board`（在 `runner.py`）：哪个局面激活哪个 concept
- 这两个映射是我们根据国际象棋教学直觉手写的，**但它们在 simulator 里，不在 reward 里**。bandit 学的是 "给定 20 维 context，选哪个 arm 最能让 downstream cp_loss 低"——而 simulator 的动态决定这个关系

> **⚠️ 真正的限制**：仍需要真人 user pilot 来验证 `FEEDBACK_CONCEPT_MAP` 的现实相关性。绝对 reward 数字不代表真人教学效果。详见 FAQ Q5。

### 7.5 对比策略

我们实现了 5 种策略做对比（全部在 `bandit.py`）：

| 策略 | 核心思想 |
|------|---------|
| **Thompson Sampling** | 采样后验，自然平衡探索与利用 |
| ε-Greedy (ε=0.1) | 90% 选历史最优，10% 随机探索 |
| LinUCB (α=1) | UCB 风格的置信上界，Li et al. (WWW 2010) |
| Random | 均匀随机 |
| Rule-Based | 手写规则（blunder_prob 高→警告，complexity 高→简化） |

---

## 8. 学生模拟器

### 8.1 为什么需要模拟

Bandit 需要几百到上千次交互才能收敛。我们不可能在一学期内找到足够多的真人学生做实验。所以我们构建了一个符合教育学理论的学生模拟器做离线评估。

### 8.2 学生模型

每个模拟学生有以下状态（定义在 `chess_tutor/student/model.py`）：

- **ELO**：当前水平（1000-2000）
- **Weakness profile**：5 个概念维度的 mastery 分数
  - tactics（战术）、strategy（战略）、endgame（残局）、opening（开局）、calculation（计算）
  - 每个 mastery 在 [0, 1] 之间，0 = 完全不会，1 = 完全掌握
- **Recent cp_losses**：最近 10 步的失误记录
- **Trend**：根据最近表现自动判断 improving / stable / declining

### 8.3 学习规则

采用 Zone of Proximal Development (ZPD) 模型：

```math
p_{\text{learn}} = \eta \cdot \text{relevance}(f, s) \cdot (1 - m_s)
```

其中 $\eta$ 是 base learning rate，$\text{relevance}(f, s)$ 是反馈类型 $f$ 对学生弱点 $s$ 的相关度，$m_s$ 是当前 mastery。

核心思想：
- 反馈必须**相关**才有效——给一个战略弱的学生战术提醒收益不大
- **掌握度越低**进步越快——新手刚学到的概念进步最明显
- 这不是线性的——随着 mastery 接近 1.0，进步越来越难

### 8.4 StudentPopulation

`StudentPopulation.generate(n_students=50, random_state=42)` 生成一批多样的学生：

- ELO 从 1000 到 1800 均匀分布
- 每个学生的 weakness profile 随机生成但保持合理（初学者各维度都弱，高手只在部分维度弱）

代码在 `chess_tutor/simulation/student_simulator.py`。

---

## 9. 实验设计与结果

### 9.1 走法预测实验

**设置**：80/20 按 position 切分，3 种架构 × 5 个 ELO 段。

**结果**（详见 §4.6）：
- B (pooled + ELO) top-1 = 0.1582，是 ablation 里的最佳
- C (kernel, best bandwidth=200) top-1 = 0.1533，与 A (0.1516) 接近
- Cross-ELO 混淆矩阵 **3/5 对角占优**（1100/1300 反而被 1700 段模型预测最准）
- **C 的价值在方法论**（连续 ELO 查询），不在 raw top-1 数字

**产出图表**：
- `cross_elo_heatmap.png`：5×5 的跨段准确率热力图
- `feature_importance.png`：RF 的 top 15 特征重要度
- `kernel_weights.png`：3 种 bandwidth 下的高斯核权重分布
- `bandwidth_cv.png`：bandwidth 和准确率的关系曲线

### 9.2 Blunder 数据分析（negative finding）

- 用真实 Stockfish cp_loss 标注 22,712 个走法
- Handcrafted features 预测 blunder 的 AUC ≈ 0.51（近似随机）
- 各 ELO 段的 blunder rate ≈ 10%，相当稳定
- 详细讨论见第 5 章

### 9.3 Bandit 策略对比实验（pure empirical reward，无 alignment）

**Primary experiment**：50 个模拟学生 × 300 episode × 30 interactions = 450,000 次交互，5 种策略（default hyperparameters），**纯 empirical reward + concept-aware simulator**。

**真实结果**（来自 `results/bandit_comparison.csv`，20 维 context 所有 dims 均 active）：

| Policy | Mean Reward | Std | ELO Gain | Arm Entropy |
|--------|-------------|-----|----------|-------------|
| **Thompson Sampling** | **25.649** | 4.16 | 26.5 | **2.81** |
| ε-Greedy (ε=0.1) | 25.646 | 4.22 | 26.7 | 1.03 |
| LinUCB (α=1) | 25.597 | 4.08 | 25.4 | 2.78 |
| Random | 25.592 | 4.22 | 26.2 | **2.81** |
| Rule-Based | 25.521 | 4.32 | 26.2 | 1.18 |

**Hyperparameter sweep**（`results/hyperparam_sweep.csv`）：ε-Greedy 包揽前 4，TS 最佳 v=0.5 排第 6，LinUCB α=0.1/0.5 最差（排 13-14）。完整 14-config 表见 FAQ Q4。

**诚实 takeaway**：

1. **5 种 policy 在统计上无差别**——spread 只有 0.5%（25.52-25.65），而每个 policy 的 std 约 4.2。任何 ranking 都在噪声范围内。这是项目最诚实的 finding：**在真正 empirical 的 reward 下（无 hand-crafted alignment），bandit 没有统计意义地打败 Random**。
2. **Thompson Sampling 和 Random 的 arm entropy 同为 2.81**——signal 不够强时 TS 的 posterior 保持接近 uniform prior，采样分布和 Random 一样。TS 不是"实现错了"，而是它在**诚实汇报信号强度**。
3. **Pseudo-ranking 会在 runs 之间翻转**——前一轮（dim 18/19 作为 reserved=0 时）ε-Greedy 领先 2.5%，现在（激活 phase 特征后）TS 反超 0.01%。gap 在这种量级时，"赢家"取决于你用的随机种子。
4. **Rule-Based 一贯垫底**（25.52）——硬写的规则在 simulator concept 动态下是负担。
5. **ELO Gain spread 5%**（25.4 - 26.7），仍在 policy std 内。

**和之前版本的对比**：

旧版用 hand-crafted alignment term 时，TS/LinUCB 比 Random 高 8-12%。那些数字本质上是 bandit 在 **recover 我们自己写进 reward 的规则**——真实但 tautological。去掉 alignment 后，大部分优势消失。**这才是真实的 empirical picture**。

**我们在 Section 10 live demo 中仍然选 TS 作 default 的理由**：

1. **理论保证**：Agrawal-Goyal (2013) 的 $\tilde{O}(d\sqrt{T})$ Bayesian regret bound 是本项目唯一有严格理论保证的 policy。
2. **Arm entropy 最高（2.81）**：学生看到最多样的 feedback 类型。ε-Greedy 的 1.05 说明它很快塌缩到一个主导 arm——对教学是不可取的（学生只看到一类反馈）。
3. **无 hyperparameter**：v=1 是标准选择。
4. **Graceful degradation**：signal 强时（真人 pilot，feedback 效应可观测）TS 的 Bayesian posterior 自然累积证据；signal 弱时（现在的 simulator）TS 行为等同 random，是安全 default。

这不等于"TS 是最好的"——在 pure empirical reward 上 ε-Greedy 赢。我们诚实地报告。

**产出图表**：
- `regret_curves.png`：5 种策略的 regret 曲线（含 confidence bands）
- `arm_distribution.png`：各策略选择每种反馈类型的频率
- `elo_gain.png`：策略对学生 ELO gain 的 boxplot

### 9.4 Ablation 实验

- **Bandwidth ablation**：{25, 50, 75, 100, 150, 200, 300}，最优 100
- **Classifier ablation**：RF vs GBT vs LogReg vs Ridge，RF 综合最优
- **Feature ablation**：去掉 king_safety 降 3%，去掉 phase 降 2%

产出：`ablation_table.png`

实验脚本在 `scripts/run_final_experiment.py`，一次运行生成所有图表和数据表，总耗时约 35 分钟（CPU）。

---

## 10. 交互式 Demo

### 10.1 Section 9：位置评估器

**用途**：用户粘贴任意 FEN 字符串，选一个 ELO，看教练给出的评估。

**界面设计**：
- `widgets.Text`：输入 FEN
- `widgets.IntSlider`：选 ELO（800-2200，步长 100）
- `widgets.Button`：触发评估

**输出内容**：
1. SVG 棋盘渲染
2. 位置评估：assessment（equal/slight advantage/...）+ confidence + suggested plan
3. 预测走法：该 ELO 最可能走的 top-3
4. 7 种反馈文本（标注出 bandit 推荐的那种）
5. 跨 ELO 对比：1100/1300/1500/1700/1900 各自的预测和反馈并列展示

### 10.2 Section 10：与 Bot 对弈

**界面设计**（经过 5 轮迭代最终定型）：

用户通过点击 SVG 棋盘选子和走棋，不需要输入任何棋谱格式。技术实现：

1. `chess.svg.board()` 渲染棋盘为 SVG
2. 注入 JavaScript 点击事件处理器
3. JavaScript 从点击坐标计算出格名（如 "e4"）
4. 通过一个隐藏的 `widgets.Text` 把格名传回 Python
5. Python 端 `.observe()` 回调处理走棋逻辑

点击交互流程：
- 第一次点击选中一个己方棋子 → 高亮为蓝色，合法落点显示为绿色
- 第二次点击到绿色目标 → 执行走棋
- 点到其他位置 → 取消选择

**每步走棋后的系统反馈**：
1. `CommentaryGenerator.comment_on_student_move()` —— 评论用户这步
2. `build_context()` —— 构建 20 维 context 向量
3. `LinearThompsonSampling.select_arm()` —— bandit 选择最优反馈类型
4. `FeedbackGenerator.generate()` —— 生成对应的反馈文本
5. `bandit.update()` —— 用 reward 更新 bandit 后验

然后 bot 走棋：
1. `ChessTutorBot.play_move()` —— bot 在目标 ELO 下选走法
2. `CommentaryGenerator.comment_on_bot_move()` —— bot 解释自己为什么走这步
3. `evaluate_position()` —— 显示当前局势评估

**关键修复**：`is_game_over(claim_draw=False)` 防止 50-move rule 提前结束对局。

### 10.3 Engine vs. Tutor 对比

用 4 个典型局面（开局/战术/安静中局/残局），并排展示：

- **Engine 输出**：对所有人都一样——一个走法 + 一个分数
- **Tutor 在 ELO 1100 的输出**：简单语言的评估 + 基础计划 + 初级反馈
- **Tutor 在 ELO 1800 的输出**：深入的评估 + 高级计划 + 进阶反馈

这个 section 直接回应教授的要求——证明系统比 raw engine 更有用。

---

## 11. 提交材料

| 材料 | 文件 | 状态 |
|------|------|------|
| Executive Summary（2 页，无数学/术语） | `EXECUTIVE_SUMMARY.md` | 已完成 |
| FAQ（12 个问题） | `FAQ.md` | 已完成 |
| Technical Appendix | `README.md` 的后半段 | 已完成 |
| Jupyter Demo | `chess_tutor/demo/chess_tutor_demo.ipynb` | 已完成（10 sections + 2 interactive） |
| 可复现代码 | 整个 `chess_tutor/` + `scripts/` | 已完成 |
| 离线 demo 数据 | `data/demo_cache/`（137 KB） | 已完成 |

---

## 12. 关键数据总览

| 指标 | 值 |
|------|-----|
| 代码量 | 35 模块，~3900 行 |
| 训练数据 | 22,712 个 unique positions × 5 ELO 段 → ~73 万 candidate rows |
| 特征维度 | 30（棋盘）+ 10（走法）= 40 |
| 走法预测 top-1（最佳 = B: Pooled+ELO） | 0.1582 |
| 走法预测 top-1（C: Kernel, 最佳 bw=200） | 0.1533 |
| 走法预测 top-1（A: Per-bracket） | 0.1516 |
| Cross-ELO 对角占优 | **3/5 brackets**（1500/1700/1900）；1100 和 1300 反而被 1700 段模型预测最准——真实的 cross-bracket generalization finding |
| Stockfish-标注真实 blunder rate（全段） | ~10% |
| Bandit 表现（pure empirical reward） | 5 policy 在 25.52-25.65 范围内统计持平（std ≈ 4.2） |
| TS arm entropy（反馈多样性） | 2.81（与 Random 并列最高） |
| Context dimensions | 20 维，全部 active（dim 18/19 为 phase_opening/phase_endgame，替换掉旧的 reserved=0 死特征） |
| Regret 类型 | Sublinear（√T） |
| 反馈类型 | 7 种 |
| Bandit context 维度 | 20 |
| Pipeline 总运行时间 | ~35 分钟（CPU） |
| 图表 | 10 张 |
