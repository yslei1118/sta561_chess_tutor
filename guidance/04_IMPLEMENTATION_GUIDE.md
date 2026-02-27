# Chess Tutor: Implementation Guide

> Step-by-step instructions for coding each phase. Follow API_SPECS.md for exact signatures.

---

## §1 Phase 1: Data Pipeline

### Step 1.1: Setup

```bash
pip install python-chess stockfish pandas numpy scikit-learn matplotlib ipywidgets zstandard pyarrow huggingface_hub
```

Create `requirements.txt`:
```
python-chess>=1.10.0
stockfish>=3.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
ipywidgets>=8.0.0
zstandard>=0.21.0
pyarrow>=14.0.0
huggingface_hub>=0.19.0
```

### Step 1.2: Download Data

```python
# data/download.py — key pattern
import zstandard
import urllib.request
import os

def download_lichess_pgn(year, month, output_dir="data/raw/"):
    url = f"https://database.lichess.org/standard/lichess_db_standard_rated_{year}-{month:02d}.pgn.zst"
    zst_path = os.path.join(output_dir, f"lichess_{year}_{month:02d}.pgn.zst")
    pgn_path = os.path.join(output_dir, f"lichess_{year}_{month:02d}.pgn")
    
    if os.path.exists(pgn_path):
        return pgn_path
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Stream download (files are large)
    urllib.request.urlretrieve(url, zst_path)
    
    # Stream decompress
    dctx = zstandard.ZstdDecompressor()
    with open(zst_path, 'rb') as ifh, open(pgn_path, 'wb') as ofh:
        dctx.copy_stream(ifh, ofh)
    
    os.remove(zst_path)  # Clean up compressed file
    return pgn_path
```

**GOTCHA:** Start with a small file first! `lichess_db_standard_rated_2013-06.pgn.zst` is only ~40MB. Later months are 20+ GB.

### Step 1.3: Parse PGN

```python
# data/parse_pgn.py — key pattern
import chess.pgn
import io

def parse_pgn_file(pgn_path, elo_brackets=[1100,1300,1500,1700,1900], 
                   elo_tolerance=50, max_games_per_bracket=100000, ...):
    records = []
    bracket_counts = {b: 0 for b in elo_brackets}
    
    with open(pgn_path) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            
            # Filter: rated, standard/rapid, both ELOs present
            headers = game.headers
            white_elo = int(headers.get("WhiteElo", 0))
            black_elo = int(headers.get("BlackElo", 0))
            if white_elo == 0 or black_elo == 0:
                continue
            
            # Check if either player falls in a target bracket
            target_bracket = None
            for b in elo_brackets:
                avg_elo = (white_elo + black_elo) / 2
                if abs(avg_elo - b) <= elo_tolerance:
                    target_bracket = b
                    break
            if target_bracket is None:
                continue
            if bracket_counts[target_bracket] >= max_games_per_bracket:
                continue
            
            bracket_counts[target_bracket] += 1
            
            # Walk through the game
            board = game.board()
            move_num = 0
            for node in game.mainline():
                move_num += 1
                move = node.move
                
                # Sample positions
                if move_num >= min_move and move_num <= max_move:
                    if move_num % sample_interval == 0:
                        player_elo = white_elo if board.turn == chess.WHITE else black_elo
                        records.append({
                            'game_id': headers.get("Site", ""),
                            'move_number': move_num,
                            'fen': board.fen(),
                            'white_elo': white_elo,
                            'black_elo': black_elo,
                            'side_to_move': 'white' if board.turn == chess.WHITE else 'black',
                            'player_elo': player_elo,
                            'move_uci': move.uci(),
                            'move_san': board.san(move),
                            'result': headers.get("Result", "*"),
                            'time_control': headers.get("TimeControl", ""),
                        })
                
                board.push(move)
            
            # Check if all brackets are full
            if all(c >= max_games_per_bracket for c in bracket_counts.values()):
                break
    
    return pd.DataFrame(records)
```

**GOTCHA:** `chess.pgn.read_game` is slow on huge files. For 100K+ games, consider parallel processing or streaming with offsets.

### Step 1.4: Feature Extraction

```python
# data/extract_features.py — key patterns

PIECE_VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}

def compute_material_balance(board):
    balance = 0
    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        balance += len(board.pieces(piece_type, chess.WHITE)) * PIECE_VALUES[piece_type]
        balance -= len(board.pieces(piece_type, chess.BLACK)) * PIECE_VALUES[piece_type]
    return balance

def compute_king_safety(board, color):
    king_sq = board.king(color)
    if king_sq is None:
        return 0, 0
    
    # Pawn shield: count friendly pawns in front of king
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    shield_files = [max(0, king_file-1), king_file, min(7, king_file+1)]
    shield_rank = king_rank + (1 if color == chess.WHITE else -1)
    
    shield_count = 0
    for f in shield_files:
        if 0 <= shield_rank <= 7:
            sq = chess.square(f, shield_rank)
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                shield_count += 1
    
    # King zone attacks: count opponent attacks on king neighborhood
    king_zone = chess.SquareSet(chess.BB_KING_ATTACKS[king_sq])
    king_zone.add(king_sq)
    attacks = sum(1 for sq in king_zone if board.is_attacked_by(not color, sq))
    
    return shield_count, attacks

def detect_game_phase(board):
    # Count total material (excluding kings)
    total = 0
    for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        total += len(board.pieces(pt, chess.WHITE)) * PIECE_VALUES[pt]
        total += len(board.pieces(pt, chess.BLACK)) * PIECE_VALUES[pt]
    
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
    
    if total > 6000 and queens >= 2:
        return 'opening' if board.fullmove_number <= 10 else 'middlegame'
    elif total < 2500 or queens == 0:
        return 'endgame'
    else:
        return 'middlegame'
```

**GOTCHA:** `chess.BB_KING_ATTACKS` might not exist in all python-chess versions. Use `chess.KING_ATTACKS` or compute manually via `board.attacks(king_sq)`.

### Step 1.5: Dataset Assembly

Use `ChessTutorDataset.build_from_pgn()` to chain all steps. Save as parquet for fast loading.

```python
# Quick validation: load 1K games, extract features, verify shapes
dataset = ChessTutorDataset()
dataset.build_from_pgn(["data/raw/lichess_2013_06.pgn"])
assert dataset.n_positions > 5000
X, y = dataset.get_features_and_labels(elo_bracket=1500)
assert X.shape[1] == 30  # board features only
```

---

## §2 Phase 2: Move Prediction

### Step 2.1: Move Encoding

**Key decision:** How to represent the target move `y`.

**Approach 1 (Recommended): Top-K candidate ranking**
- For each position, get top 10 legal moves by Stockfish
- Encode as multi-class classification over these 10 candidates
- Features = board_features + candidate_move_features (per candidate)
- Train as binary: P(this_candidate = human_move | features)

**Approach 2: Direct move index**
- Map each possible UCI move to an index (large vocabulary ~4000)
- Multi-class classification
- Works but very sparse; most moves never seen in training

**Use Approach 1.** It's how BP-Chess works and achieves strong results.

### Step 2.2: Training Architecture A (Per-Bracket)

```python
# For each bracket, train a separate model
for elo in [1100, 1300, 1500, 1700, 1900]:
    X_bracket, y_bracket = dataset.get_features_and_labels(elo_bracket=elo)
    model = RandomForestClassifier(n_estimators=500, max_depth=20, n_jobs=-1)
    model.fit(X_bracket, y_bracket)
    models[elo] = model
```

### Step 2.3: Training Architecture C (Kernel Interpolation)

```python
# Train per-bracket models (same as A)
bracket_models = {}
for elo in ELO_BRACKETS:
    X, y = dataset.get_features_and_labels(elo_bracket=elo)
    model = RandomForestClassifier(...)
    model.fit(X, y)
    bracket_models[elo] = model

# At inference time, combine with Nadaraya-Watson
nw = NadarayaWatsonELO(bandwidth=100)

# For a query at ELO 1400:
bracket_preds = {elo: model.predict_proba(X_test) for elo, model in bracket_models.items()}
interpolated = nw.interpolate(bracket_preds, target_elo=1400)
```

### Step 2.4: Bandwidth Selection

```python
# Leave-one-bracket-out cross-validation
best_bw = nw.select_bandwidth_cv(
    X=X_all, y=y_all, elos=elos_all,
    bracket_models=bracket_models,
    bandwidths=[25, 50, 75, 100, 150, 200, 300]
)
```

### Step 2.5: Cross-ELO Evaluation Matrix

```python
# For each pair (train_bracket, test_bracket), compute accuracy
matrix = np.zeros((5, 5))
for i, train_elo in enumerate(ELO_BRACKETS):
    for j, test_elo in enumerate(ELO_BRACKETS):
        preds = bracket_models[train_elo].predict(X_test[test_elo])
        matrix[i, j] = accuracy_score(y_test[test_elo], preds)

# Diagonal should be highest → model works best when matched to ELO
# Plot as heatmap with seaborn or matplotlib
```

---

## §3 Phase 3: Position Evaluation & Feedback

### Step 3.1: Blunder Detection

```python
# Binary classification on ALL positions (not per-bracket)
X_all = np.concatenate([board_features, move_features], axis=1)  # (n, 40)
y_blunder = (cp_losses > 100).astype(int)

blunder_det = BlunderDetector(classifier='rf')
blunder_det.fit(X_all, y_blunder)
print(f"AUC: {blunder_det.score(X_test, y_test)['auc']}")
```

### Step 3.2: Feedback Templates

```python
# feedback/templates.py — example templates
TEMPLATES = {
    FeedbackType.TACTICAL_ALERT: [
        "Look carefully — there's a {tactic_type} opportunity on {square}!",
        "This position has a tactical shot. Can you find the {tactic_type}?",
        "Don't miss it: {piece} to {square} creates a {tactic_type}.",
    ],
    FeedbackType.BLUNDER_WARNING: [
        "Be careful with {move}! It loses {cp_loss} centipawns.",
        "That move looks natural, but {explanation}. Consider {alternative} instead.",
        "Watch out — {move} drops material because of {threat}.",
    ],
    FeedbackType.ENCOURAGEMENT: [
        "Nice move! {move} was the engine's top choice too.",
        "Good instinct! That move improves your position by controlling {feature}.",
        "Well played — you found the best move in a tricky position.",
    ],
    # ... etc for all 7 types
}
```

### Step 3.3: Tactic Detection (for feedback relevance)

```python
# Simple tactic detection using Stockfish eval swings
def detect_tactics(board, stockfish):
    """Check if position contains a tactic (sharp eval swing after best move)."""
    eval_before = stockfish.evaluate(board)
    
    # Play the best move
    best_move = chess.Move.from_uci(eval_before['best_move'])
    board.push(best_move)
    eval_after = stockfish.evaluate(board)
    board.pop()
    
    # Large eval swing = tactic exists
    swing = abs(eval_after['score_cp'] - eval_before['score_cp'])
    if swing > 200:
        return True, categorize_tactic(board, best_move)
    return False, None
```

---

## §4 Phase 4: Teaching Engine

### Step 4.1: Linear Thompson Sampling

```python
# teaching/bandit.py — core implementation

class LinearThompsonSampling:
    def __init__(self, n_arms=7, context_dim=20, exploration_v=1.0, prior_variance=1.0):
        self.n_arms = n_arms
        self.d = context_dim
        self.v = exploration_v
        
        # Per-arm parameters
        self.B = [np.eye(context_dim) * (1.0 / prior_variance) for _ in range(n_arms)]
        self.f = [np.zeros(context_dim) for _ in range(n_arms)]
        self.mu_hat = [np.zeros(context_dim) for _ in range(n_arms)]
        self.n_pulls = np.zeros(n_arms, dtype=int)
        self._cumulative_reward = 0.0
    
    def select_arm(self, context):
        scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            # Sample from posterior
            cov = self.v**2 * np.linalg.inv(self.B[a])
            theta_sample = np.random.multivariate_normal(self.mu_hat[a], cov)
            scores[a] = theta_sample @ context
        return int(np.argmax(scores))
    
    def update(self, arm, context, reward):
        self.B[arm] += np.outer(context, context)
        self.f[arm] += reward * context
        self.mu_hat[arm] = np.linalg.solve(self.B[arm], self.f[arm])
        self.n_pulls[arm] += 1
        self._cumulative_reward += reward
```

**GOTCHA:** Use `np.linalg.solve` instead of `np.linalg.inv` for numerical stability. Use Cholesky decomposition for sampling if context_dim is large.

### Step 4.2: Student Simulator

```python
class StudentSimulator:
    def __init__(self, elo, weakness_profile=None, move_predictor=None, learning_rate=0.01):
        self.elo = elo
        self.weakness_profile = weakness_profile or self._default_profile(elo)
        self.move_predictor = move_predictor
        self.lr = learning_rate
        self.recent_cp_losses = []
        self.blunder_count = 0
        self.moves_played = 0
    
    def _default_profile(self, elo):
        """Generate a random weakness profile correlated with ELO."""
        base = (elo - 800) / 1200  # normalize to [0, 1]-ish
        return {
            'tactics': np.clip(base + np.random.normal(0, 0.15), 0, 1),
            'strategy': np.clip(base - 0.1 + np.random.normal(0, 0.15), 0, 1),
            'endgame': np.clip(base - 0.05 + np.random.normal(0, 0.15), 0, 1),
            'opening': np.clip(base + np.random.normal(0, 0.1), 0, 1),
            'calculation': np.clip(base + np.random.normal(0, 0.15), 0, 1),
        }
    
    def respond_to_position(self, board, feedback_type=None, feedback_text=None):
        # Base: sample from human move distribution at this ELO
        # If feedback given: improve with probability based on relevance
        features = extract_board_features(board)
        base_move = self._sample_move(board, features)
        
        if feedback_type is not None:
            relevance = self._compute_relevance(feedback_type, board)
            concept = FEEDBACK_CONCEPT_MAP[feedback_type]
            mastery = np.mean([self.weakness_profile.get(c, 0.5) for c in concept]) if concept else 0.5
            p_learn = self.lr * relevance * (1 - mastery)
            
            if np.random.random() < p_learn:
                # Upgrade: pick a better move (from higher ELO model or engine)
                better_elo = min(self.elo + 200, 1900)
                base_move = self._sample_move(board, features, elo_override=better_elo)
        
        return base_move
```

### Step 4.3: Running Experiments

```python
# Compare all policies
policies = {
    'Thompson Sampling': LinearThompsonSampling(n_arms=7, context_dim=20),
    'ε-Greedy (ε=0.1)': EpsilonGreedy(n_arms=7, epsilon=0.1),
    'LinUCB (α=1)': LinUCB(n_arms=7, context_dim=20, alpha=1.0),
    'Random': RandomPolicy(n_arms=7),
    'Rule-Based': RuleBasedPolicy(),
}

students = StudentPopulation.generate(n_students=50)
results = run_experiment(students, policies, n_episodes=200, n_interactions_per_episode=20)

# Plot regret curves
for name, res in results.items():
    plt.plot(res['regret_curves'].mean(axis=0), label=name)
plt.legend()
plt.xlabel('Interaction')
plt.ylabel('Cumulative Regret')
```

---

## §5 Phase 5: Interactive Demo

### Step 5.1: Position Evaluator Widget

```python
# demo/tutor_demo.ipynb — Cell: Position Evaluator
import ipywidgets as widgets
from IPython.display import display, SVG
import chess
import chess.svg

fen_input = widgets.Text(value='rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1',
                         description='FEN:')
elo_slider = widgets.IntSlider(value=1500, min=800, max=2000, step=100, description='ELO:')
eval_button = widgets.Button(description='Evaluate')
output = widgets.Output()

def on_evaluate(b):
    with output:
        output.clear_output()
        board = chess.Board(fen_input.value)
        display(SVG(chess.svg.board(board, size=400)))
        
        evaluation = tutor_bot.evaluate_position(board, elo_slider.value)
        print(f"\n=== Evaluation for ELO {elo_slider.value} ===")
        print(f"Assessment: {evaluation['assessment']}")
        print(f"Suggested plan: {evaluation['suggested_plan']}")
        print(f"Top predicted moves at this ELO: {evaluation['move_predictions']}")

eval_button.on_click(on_evaluate)
display(widgets.VBox([fen_input, elo_slider, eval_button, output]))
```

### Step 5.2: Playing Bot

```python
# demo/tutor_demo.ipynb — Cell: Play Against Bot
board = chess.Board()
bot = ChessTutorBot(move_predictor=model_c, stockfish_evaluator=sf)
commentary = CommentaryGenerator(...)

move_input = widgets.Text(description='Your move (SAN):')
play_button = widgets.Button(description='Play')
game_output = widgets.Output()

def on_play(b):
    with game_output:
        # Parse student move
        try:
            student_move = board.parse_san(move_input.value)
        except ValueError:
            print("Invalid move! Try again.")
            return
        
        # Comment on student move
        comment = commentary.comment_on_student_move(board, student_move, student_elo=elo_slider.value, ...)
        board.push(student_move)
        print(f"You: {move_input.value} — {comment}")
        
        # Bot responds
        bot_move = bot.play_move(board, target_elo=elo_slider.value)
        bot_comment = commentary.comment_on_bot_move(board, bot_move, target_elo=elo_slider.value, ...)
        board.push(bot_move)
        print(f"Bot: {board.san(bot_move)} — {bot_comment}")
        
        display(SVG(chess.svg.board(board, size=400)))
```

### Step 5.3: Board Rendering Helper

```python
# utils/visualization.py
import chess.svg
from IPython.display import SVG, display

def render_board(board, last_move=None, arrows=None, size=400):
    """Render board with optional highlights."""
    kwargs = {'size': size}
    if last_move:
        kwargs['lastmove'] = last_move
    if arrows:
        kwargs['arrows'] = arrows
    return SVG(chess.svg.board(board, **kwargs))

def plot_cross_elo_matrix(matrix, elo_brackets, save_path=None):
    """Plot the cross-ELO accuracy heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='.3f', 
                xticklabels=elo_brackets, yticklabels=elo_brackets,
                cmap='YlOrRd', ax=ax)
    ax.set_xlabel('Test ELO Bracket')
    ax.set_ylabel('Model Trained on ELO Bracket')
    ax.set_title('Cross-ELO Move Matching Accuracy')
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig
```

---

## Common Gotchas & Tips

1. **python-chess board state:** `board.push()` modifies in place. Always `.copy()` before branching.
2. **Stockfish process:** Use `with StockfishEvaluator() as sf:` to ensure cleanup. Don't leave orphan processes.
3. **Feature consistency:** Always extract features from the position BEFORE the move is played.
4. **Memory management:** Lichess data is huge. Use generators/streaming, don't load everything into RAM.
5. **Move encoding:** Legal move sets differ per position. Use candidate-based encoding, not fixed vocabulary.
6. **Numerical stability:** In Thompson Sampling, add small regularization to B matrices to avoid singular.
7. **Random seeds:** Set `np.random.seed(42)` and `random_state=42` everywhere for reproducibility.
8. **Stockfish availability:** Not all systems have Stockfish installed. Make it optional (use HuggingFace evals as fallback).
