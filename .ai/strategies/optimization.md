# Strategy Reference: Optimization Competitions

Combinatorial optimization problems (bin packing, TSP, scheduling, assignment). This is reference material — read it, then develop your own approach in your workspace STRATEGY.md.

## Approach Order (start simple, add complexity)

### 1. Greedy Baseline (first 30 minutes)
- Implement the simplest greedy heuristic for the problem
- Submit immediately — this is your floor score
- Examples: largest-first placement, nearest-neighbor for TSP, first-fit for bin packing
- Don't over-engineer. A working baseline in 30 minutes beats a perfect one in 3 hours.

### 2. GPU-Accelerated Exact Solvers (cuOpt)
Before investing in custom heuristics, check whether the problem can be formulated as a standard mathematical program. NVIDIA cuOpt can convert natural-language optimization problems into LP/MILP/QP formulations and solve them on GPU with massive speedups over CPU solvers.

**cuOpt Numerical Optimization (LP/MILP/QP):**
- Converts problem descriptions into linear programs (LP), mixed-integer linear programs (MILP), or quadratic programs (QP)
- GPU-accelerated solving — orders of magnitude faster than CPU-based solvers for large instances
- Install: `npx skills add nvidia/skills/cuopt-numerical-optimization-formulation`

**cuOpt Routing Solver (VRP/TSP/PDP):**
- Purpose-built for vehicle routing problems: TSP, VRP, pickup-and-delivery (PDP), and variants with time windows, capacity constraints, etc.
- Handles real-world constraints (fleet heterogeneity, breaks, precedence) out of the box
- Install: `npx skills add nvidia/skills/cuopt-routing-optimization`

**When to use:**
- The problem can be cleanly expressed as an LP, MILP, or QP (assignment, facility location, flow, scheduling with linear constraints)
- The problem is a routing/TSP/VRP variant — cuOpt routing is specifically built for these
- You want a strong solution quickly without writing custom C++ search code
- Instance sizes are large enough that GPU acceleration provides meaningful speedup

**When NOT to use:**
- The objective function or constraints are non-linear, black-box, or too complex for standard mathematical formulations
- The problem has highly custom combinatorial structure that does not map to LP/MILP/QP (e.g., irregular packing with rotations)
- The scoring function involves simulation or evaluation that cannot be expressed algebraically
- Small instances where a CPU solver (Gurobi, CPLEX, OR-Tools) already finishes in seconds

**Workflow tip:** Even if cuOpt cannot solve the full problem optimally, use it to generate a strong initial solution, then refine with local search or SA (sections 3-4 below).

### 3. Local Search (next phase)
- Take greedy solution, apply local moves (swaps, translations, rotations)
- Accept moves that improve score, reject others (hill climbing)
- This alone often reaches top 50%
- Key: define the right neighborhood of moves for the problem

### 4. Simulated Annealing (primary approach)
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

### 5. Advanced Techniques (if SA plateaus)
- Genetic algorithms / evolutionary strategies
- Constraint programming (for feasibility-heavy problems)
- Problem-specific decomposition (solve subproblems independently)
- Hybrid: SA + problem-specific moves (e.g., LKH-style for TSP)
- Late acceptance hill climbing (LAHC) — simpler than SA, sometimes competitive

**GPU-Accelerated Advanced Techniques:**
- **TileGym / cuTile for custom GPU kernel autotuning**: When your scoring function is compute-bound (e.g., evaluating millions of packing configurations, large matrix operations in the objective), use TileGym/cuTile to auto-tune custom CUDA kernels for your specific evaluation logic. This can yield 10-50x speedups over naive GPU implementations by optimizing tile sizes, memory access patterns, and thread block configurations for your exact workload.
- **Multi-objective Pareto exploration (cuopt-multi-objective-exploration)**: For problems with multiple competing objectives (e.g., minimize cost AND maximize coverage), use cuOpt's multi-objective exploration to efficiently map the Pareto frontier on GPU. Install: `npx skills add nvidia/skills/cuopt-multi-objective-exploration`. This is particularly useful when the competition scoring function is a weighted combination of objectives and you want to explore the tradeoff space before committing to specific weights.

## Implementation Guidelines

- **Language**: C++ strongly preferred. The difference between C++ and Python for optimization can be 100x+ in speed, which directly translates to solution quality.
- **GPU solvers as a C++ alternative**: For problems that map to standard formulations (LP/MILP/QP) or routing, cuOpt can match or exceed hand-written C++ performance without the implementation effort. Consider cuOpt first for routing and mathematically-formulable problems; fall back to C++ for custom heuristic search where GPU solver formulations do not apply.
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
