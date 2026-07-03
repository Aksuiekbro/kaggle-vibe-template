# Strategy Reference: Optimization Competitions

Combinatorial optimization problems (bin packing, TSP, scheduling, assignment). This is reference material — read it, then develop your own approach in your workspace STRATEGY.md.

## Approach Order (start simple, add complexity)

### 1. Greedy Baseline (first 30 minutes)
- Implement the simplest greedy heuristic for the problem
- Submit immediately — this is your floor score
- Examples: largest-first placement, nearest-neighbor for TSP, first-fit for bin packing
- Don't over-engineer. A working baseline in 30 minutes beats a perfect one in 3 hours.

### 2. Local Search (next phase)
- Take greedy solution, apply local moves (swaps, translations, rotations)
- Accept moves that improve score, reject others (hill climbing)
- This alone often reaches top 50%
- Key: define the right neighborhood of moves for the problem

### 3. Simulated Annealing (primary approach)
SA is the dominant approach for Kaggle optimization competitions.

**Key decisions:**
- Temperature schedule: geometric cooling (T *= alpha) with alpha in [0.999, 0.99999]
- Move neighborhood: what perturbations to allow — this is the most important design decision
- Acceptance: Metropolis criterion (accept worse with probability exp(-delta/T))
- Stopping: time-based or iteration-based

**Implementation tips:**
- Implement in C++ for speed — move evaluation must be fast
- Incremental evaluation (delta scoring) is critical — don't recompute full score each move
- Multi-start: run SA from multiple random initializations, keep the best
- Reheat: if stuck, increase temperature briefly then cool again

### 4. Advanced Techniques (if SA plateaus)
- Genetic algorithms / evolutionary strategies
- Constraint programming (for feasibility-heavy problems)
- Problem-specific decomposition (solve subproblems independently)
- Hybrid: SA + problem-specific moves (e.g., LKH-style for TSP)
- Late acceptance hill climbing (LAHC) — simpler than SA, sometimes competitive

## Implementation Guidelines

- **Language**: C++ strongly preferred. The difference between C++ and Python for optimization can be 100x+ in speed, which directly translates to solution quality.
- **Parallelism**: Multi-thread across independent subproblems or multi-start runs
- **Checkpointing**: Save best solution every N iterations — if the process crashes, you don't lose everything
- **Logging**: Track score over time to detect convergence. If score hasn't improved in 10% of total iterations, you're likely stuck.
- **Time management**: Know how much time you have. A 1-hour SA run and a 10-hour SA run can produce very different results.

## Common Pitfalls

- Spending too long on code quality instead of improving score
- Not testing on held-out instances (see Anti-Overfitting Protocol in RULES.md)
- Over-tuning hyperparameters on specific instances
- Using Python when C++ would be 100x faster
- Too-small move neighborhood (gets stuck in local optima)
- Too-large move neighborhood (random walk, slow convergence)
- Not implementing incremental evaluation (full recomputation is too slow)

## What Worked in Past Competitions

- Santa 2025 (tree packing): Custom SA with translation moves (14th place confirmed)
- Santa 2024: Greedy + SA + multi-threading
- TSP variants: LKH-style moves, Or-opt, 2-opt/3-opt
- Bin packing: Bottom-left-first placement, Max Rects algorithm as baseline
- Scheduling: Priority-based construction + SA with swap/insert moves
