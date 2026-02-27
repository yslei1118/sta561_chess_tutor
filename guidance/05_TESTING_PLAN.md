# Chess Tutor: Testing Plan

> Tests to run after each phase. Must all pass before moving to the next phase.

---

## §1 Phase 1 Tests: Data Pipeline

### Unit Tests

```python
# tests/test_data.py

def test_download():
    """Verify download function works on small file."""
    path = download_lichess_pgn(2013, 6, output_dir="data/test/")
    assert os.path.exists(path)
    assert os.path.getsize(path) > 1_000_000  # At least 1MB

def test_parse_pgn_basic():
    """Parse a small PGN and verify output schema."""
    df = parse_pgn_file("data/test/lichess_2013_06.pgn", max_games_per_bracket=100)
    required_cols = ['game_id', 'move_number', 'fen', 'white_elo', 'black_elo',
                     'side_to_move', 'player_elo', 'move_uci', 'move_san', 'result']
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
    assert len(df) > 100
    assert df['player_elo'].between(800, 3000).all()

def test_parse_pgn_elo_filtering():
    """Verify ELO bracket filtering works."""
    df = parse_pgn_file("data/test/lichess_2013_06.pgn", 
                        elo_brackets=[1500], elo_tolerance=50, max_games_per_bracket=50)
    assert df['player_elo'].between(1450, 1550).mean() > 0.8  # Most should be near 1500

def test_extract_board_features():
    """Verify feature extraction on known positions."""
    board = chess.Board()  # Starting position
    features = extract_board_features(board)
    assert features.shape == (30,)
    assert features[0] == 8  # White has 8 pawns
    assert features[6] == 8  # Black has 8 pawns
    assert features[12] == 0  # Material balance is 0
    assert features[13] == 20  # 20 legal moves for white in starting position

def test_extract_board_features_endgame():
    """Test features in an endgame position."""
    board = chess.Board("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1")  # K+P vs K
    features = extract_board_features(board)
    phase_idx = 26  # game phase one-hot starts at index 26
    assert features[phase_idx + 2] == 1  # endgame flag should be set

def test_extract_move_features():
    """Verify move feature extraction."""
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    features = extract_move_features(board, move)
    assert features.shape == (10,)
    assert features[0] == 0  # Not a capture
    assert features[1] == 0  # Not a check
    # Pawn one-hot at index [2]
    assert features[2] == 1  # It's a pawn move

def test_extract_move_features_capture():
    """Test on a capture move."""
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
    move = chess.Move.from_uci("e4d5")  # exd5
    features = extract_move_features(board, move)
    assert features[0] == 1  # Is capture

def test_stockfish_evaluator():
    """Verify Stockfish wrapper works."""
    with StockfishEvaluator() as sf:
        board = chess.Board()
        result = sf.evaluate(board)
        assert 'score_cp' in result
        assert 'best_move' in result
        assert abs(result['score_cp']) < 100  # Starting position is near equal

def test_dataset_roundtrip():
    """Build, save, load dataset and verify consistency."""
    dataset = ChessTutorDataset(data_dir="data/test_processed/")
    dataset.build_from_pgn(["data/test/lichess_2013_06.pgn"])
    dataset.save("test_data")
    
    dataset2 = ChessTutorDataset(data_dir="data/test_processed/")
    dataset2.load("test_data")
    assert dataset2.n_positions == dataset.n_positions
```

### Phase 1 Checkpoint Criteria
- [ ] ≥10,000 positions extracted across all ELO brackets
- [ ] Feature vectors have correct dimensions (30 board, 10 move)
- [ ] No NaN values in feature arrays
- [ ] ELO brackets are balanced (within 2× of each other)
- [ ] Stockfish evaluations available for ≥90% of positions
- [ ] Dataset saves and loads correctly as parquet

---

## §2 Phase 2 Tests: Move Prediction

### Unit Tests

```python
def test_move_predictor_arch_a():
    """Train and evaluate Architecture A (per-bracket)."""
    predictor = MovePredictor(architecture='A', base_classifier='rf')
    predictor.fit(X_train, y_train, elos_train)
    scores = predictor.score(X_test, y_test, elos_test)
    assert scores['accuracy_top1'] > 0.15  # Significantly above random
    assert len(scores['per_bracket_accuracy']) == len(ELO_BRACKETS)

def test_move_predictor_arch_b():
    """Train and evaluate Architecture B (pooled)."""
    predictor = MovePredictor(architecture='B', base_classifier='rf')
    predictor.fit(X_train, y_train, elos_train)
    scores = predictor.score(X_test, y_test, elos_test)
    assert scores['accuracy_top1'] > 0.15

def test_move_predictor_arch_c():
    """Train and evaluate Architecture C (kernel interpolation)."""
    predictor = MovePredictor(architecture='C', base_classifier='rf', kernel_bandwidth=100)
    predictor.fit(X_train, y_train, elos_train)
    scores = predictor.score(X_test, y_test, elos_test)
    assert scores['accuracy_top1'] > 0.15

def test_kernel_interpolation():
    """Test NW kernel produces valid probability distributions."""
    nw = NadarayaWatsonELO(bandwidth=100)
    bracket_preds = {
        1100: np.array([0.5, 0.3, 0.2]),
        1300: np.array([0.3, 0.4, 0.3]),
        1500: np.array([0.2, 0.3, 0.5]),
    }
    result = nw.interpolate(bracket_preds, target_elo=1200)
    assert abs(result.sum() - 1.0) < 1e-6  # Must sum to 1
    assert (result >= 0).all()  # All non-negative

def test_kernel_interpolation_at_center():
    """At a bracket center, kernel should heavily weight that bracket."""
    nw = NadarayaWatsonELO(bandwidth=50)
    weights = nw.kernel_weights(1300, [1100, 1300, 1500])
    assert weights[1] > 0.8  # Weight on 1300 bracket should dominate

def test_cross_elo_matrix_diagonal():
    """Diagonal of cross-ELO matrix should be highest in each column."""
    predictor = MovePredictor(architecture='A', base_classifier='rf')
    predictor.fit(X_train, y_train, elos_train)
    matrix = predictor.cross_elo_evaluation(X_test, y_test, elos_test)
    for j in range(matrix.shape[1]):
        assert matrix[j, j] >= matrix[:, j].mean()  # Diagonal ≥ column mean

def test_bandwidth_cv():
    """Bandwidth selection should return a positive value."""
    nw = NadarayaWatsonELO()
    bw = nw.select_bandwidth_cv(X_train, y_train, elos_train, bracket_models,
                                bandwidths=[50, 100, 200])
    assert bw in [50, 100, 200]
```

### Phase 2 Checkpoint Criteria
- [ ] Architecture A: top-1 accuracy >25% (above random ~5-10%)
- [ ] Architecture C: top-1 accuracy ≥ Architecture A
- [ ] Cross-ELO matrix shows diagonal dominance
- [ ] Kernel interpolation produces valid probability distributions
- [ ] Feature importance plots generated
- [ ] All 3 architectures compared in a summary table

---

## §3 Phase 3 Tests: Position Evaluation & Feedback

```python
def test_blunder_detector_auc():
    """Blunder detector achieves meaningful AUC."""
    bd = BlunderDetector(classifier='rf')
    bd.fit(X_train_all, y_blunder_train)
    result = bd.score(X_test_all, y_blunder_test)
    assert result['auc'] > 0.70  # Target: 0.75

def test_feedback_generator_all_types():
    """Every feedback type produces non-empty text."""
    fg = FeedbackGenerator(...)
    board = chess.Board("r1bqkb1r/pppppppp/2n5/4P3/8/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 3")
    for ft in FeedbackType:
        text = fg.generate(board, student_elo=1500, feedback_type=ft)
        assert isinstance(text, str)
        assert len(text) > 10

def test_feedback_elo_adaptation():
    """Feedback differs for different ELO levels."""
    fg = FeedbackGenerator(...)
    board = chess.Board()
    text_1200 = fg.generate(board, student_elo=1200, feedback_type=FeedbackType.STRATEGIC_NUDGE)
    text_1800 = fg.generate(board, student_elo=1800, feedback_type=FeedbackType.STRATEGIC_NUDGE)
    assert text_1200 != text_1800  # Should give different advice

def test_position_complexity():
    """Complex positions score higher than simple ones."""
    pc = PositionComplexity(model='rf')
    pc.fit(X_board_train, y_complexity_train)
    
    simple = chess.Board("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1")  # K+P vs K
    complex = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N2N2/PP2BPPP/R1BQ1RK1 w - - 0 8")
    
    feat_simple = extract_board_features(simple).reshape(1, -1)
    feat_complex = extract_board_features(complex).reshape(1, -1)
    
    assert pc.predict(feat_complex)[0] > pc.predict(feat_simple)[0]
```

### Phase 3 Checkpoint Criteria
- [ ] Blunder detection AUC >0.70 (target 0.75)
- [ ] All 7 feedback types generate meaningful text
- [ ] Feedback adapts to ELO level
- [ ] Position complexity correlates with human intuition
- [ ] Template system is extensible (easy to add new templates)

---

## §4 Phase 4 Tests: Teaching Engine

```python
def test_thompson_sampling_basic():
    """TS selects arms and updates without errors."""
    ts = LinearThompsonSampling(n_arms=7, context_dim=20)
    context = np.random.randn(20)
    arm = ts.select_arm(context)
    assert 0 <= arm < 7
    ts.update(arm, context, reward=0.5)
    assert ts.total_pulls[arm] == 1

def test_thompson_sampling_convergence():
    """On a simple bandit problem, TS should converge to best arm."""
    ts = LinearThompsonSampling(n_arms=3, context_dim=5)
    
    # Arm 0 has highest reward for all contexts
    true_theta = [np.array([1, 0, 0, 0, 0]),   # arm 0: reward = x[0]
                  np.array([0, 0.5, 0, 0, 0]),  # arm 1: reward = 0.5*x[1]
                  np.array([0, 0, 0.3, 0, 0])]  # arm 2: reward = 0.3*x[2]
    
    for t in range(500):
        context = np.abs(np.random.randn(5))
        arm = ts.select_arm(context)
        reward = true_theta[arm] @ context + np.random.normal(0, 0.1)
        ts.update(arm, context, reward)
    
    # After 500 rounds, arm 0 should be pulled most
    assert ts.total_pulls[0] > ts.total_pulls[1]
    assert ts.total_pulls[0] > ts.total_pulls[2]

def test_student_simulator_improves():
    """Student should improve over time with good feedback."""
    student = StudentSimulator(elo=1200, learning_rate=0.05)
    initial_elo = student.elo
    
    # Give relevant feedback for 100 interactions
    for _ in range(100):
        board = chess.Board()  # Use starting position for simplicity
        student.respond_to_position(board, feedback_type=FeedbackType.TACTICAL_ALERT)
        student.update_state(FeedbackType.TACTICAL_ALERT, move_quality=50, 
                           position_concepts=['tactics'])
    
    # Student should have some improvement in tactics skill
    assert student.weakness_profile['tactics'] > 0.0  # Some learning happened

def test_episode_runner():
    """Episode runner produces valid output."""
    student = StudentSimulator(elo=1200)
    policy = RandomPolicy(n_arms=7)
    result = run_episode(student, policy, feedback_generator, context_builder_fn,
                         reward_fn, positions, n_interactions=10)
    assert len(result['rewards']) == 10
    assert len(result['arms_selected']) == 10
    assert all(0 <= a < 7 for a in result['arms_selected'])

def test_experiment_comparison():
    """Full experiment runs without errors and produces comparable results."""
    policies = {
        'TS': LinearThompsonSampling(n_arms=7, context_dim=20),
        'Random': RandomPolicy(n_arms=7),
    }
    students = StudentPopulation.generate(n_students=5)
    results = run_experiment(students, policies, n_episodes=10, n_interactions_per_episode=10)
    
    assert 'TS' in results
    assert 'Random' in results
    # TS should eventually outperform random
    assert results['TS']['mean_cumulative_reward'] >= results['Random']['mean_cumulative_reward'] * 0.8

def test_regret_sublinear():
    """Cumulative regret should grow sub-linearly for TS."""
    ts = LinearThompsonSampling(n_arms=7, context_dim=20)
    # ... run for 1000 steps ...
    regret = cumulative_regret(np.array(rewards))
    # Check that regret growth rate decreases
    first_half_rate = regret[499] / 500
    second_half_rate = (regret[999] - regret[499]) / 500
    assert second_half_rate < first_half_rate  # Sub-linear: rate decreases
```

### Phase 4 Checkpoint Criteria
- [ ] Thompson Sampling runs without numerical errors
- [ ] TS converges to reasonable arm distribution
- [ ] Student simulator shows learning over time
- [ ] Regret is sub-linear
- [ ] TS outperforms random baseline (>20% improvement)
- [ ] TS matches or beats ε-greedy and rule-based
- [ ] All 5 baseline policies implemented and compared

---

## §5 Phase 5 Tests: Interactive Demo

```python
def test_bot_plays_legal_moves():
    """Bot always plays legal moves."""
    bot = ChessTutorBot(move_predictor=model_c)
    board = chess.Board()
    for _ in range(30):
        move = bot.play_move(board, target_elo=1500)
        assert move in board.legal_moves
        board.push(move)
        if board.is_game_over():
            break

def test_bot_elo_calibration():
    """Bot at different ELOs should play different quality moves."""
    bot = ChessTutorBot(move_predictor=model_c)
    board = chess.Board("r1bqkb1r/pppppppp/2n5/4P3/8/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 3")
    
    moves_1200 = [bot.play_move(board.copy(), target_elo=1200) for _ in range(50)]
    moves_1800 = [bot.play_move(board.copy(), target_elo=1800) for _ in range(50)]
    
    # Moves should differ across ELO levels
    set_1200 = set(m.uci() for m in moves_1200)
    set_1800 = set(m.uci() for m in moves_1800)
    assert set_1200 != set_1800  # Different move distributions

def test_position_evaluator():
    """Position evaluator produces valid output."""
    bot = ChessTutorBot(move_predictor=model_c)
    board = chess.Board()
    result = bot.evaluate_position(board, target_elo=1500)
    assert 'assessment' in result
    assert 'suggested_plan' in result
    assert result['assessment'] in ['winning', 'equal', 'losing', 'unclear']

def test_commentary_non_empty():
    """Commentary generator produces non-empty strings."""
    cg = CommentaryGenerator(...)
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    comment = cg.comment_on_student_move(board, move, student_elo=1500, engine_eval={...})
    assert len(comment) > 20
```

### Phase 5 Checkpoint Criteria
- [ ] Bot plays legal moves in all tested positions
- [ ] Bot plays differently at different ELO levels
- [ ] Position evaluator produces valid assessments
- [ ] Commentary generator produces readable text
- [ ] Jupyter demo notebook runs end-to-end without errors
- [ ] Board rendering works with ipywidgets

---

## §6 Final Evaluation: Plots & Tables

### Required Plots
1. **Cross-ELO accuracy heatmap** (Phase 2) — 5×5 matrix showing move-matching accuracy
2. **Feature importance bar chart** (Phase 2) — Top 15 features by RF Gini importance
3. **Kernel weight visualization** (Phase 2) — Gaussian weights for different bandwidths
4. **Bandwidth CV curve** (Phase 2) — Accuracy vs bandwidth
5. **Blunder detection ROC curve** (Phase 3) — ROC with AUC annotated
6. **Regret curves** (Phase 4) — All 5 policies on same plot, mean ± std
7. **Simulated ELO trajectories** (Phase 4) — ELO over teaching episodes, per policy
8. **Arm selection distribution** (Phase 4) — Bar chart of feedback type frequencies per policy
9. **Ablation table** (Phase 2) — Model A vs B vs C × RF/SVM/LogReg/Ridge
10. **Teaching effectiveness** (Phase 4) — Box plot of ELO gain per policy

### Required Tables
1. **Model comparison** — Architecture × Classifier × Top-1/3/5 accuracy
2. **Ablation: feature subsets** — Accuracy with cumulative features
3. **Ablation: data size** — Accuracy vs training set size
4. **Bandit comparison** — Policy × Cumulative reward × ELO gain × Entropy
5. **Feedback type usage** — Which types each policy selects most
