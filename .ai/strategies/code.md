# Strategy Reference: Code Competitions

Competitive programming, algorithm design, and code generation competitions. This is reference material — read it, then develop your own approach in your workspace STRATEGY.md.

## Approach Order (start simple, add complexity)

### 1. Problem Analysis (first 30 minutes)
- Understand the problem constraints: input size, time limit, memory limit
- Classify the problem type: graph, DP, greedy, math, simulation, search
- Identify the expected complexity from constraints (n≤10^5 → O(n log n), n≤10^3 → O(n^2))
- Write a brute-force solution first for correctness testing

### 2. Correct Solution
- Implement the algorithm that matches the expected complexity
- Test against provided examples
- Generate edge cases: minimum input, maximum input, degenerate cases
- Validate against brute-force on small inputs

### 3. Optimization
- Profile for bottlenecks
- Use appropriate data structures (segment tree, union-find, hash maps)
- Optimize I/O (fast input/output in C++)
- Constant factor optimization if needed (cache locality, SIMD)

### 4. Multi-Approach (if time allows)
- Try completely different algorithms for the same problem
- Some problems have multiple valid approaches with different trade-offs
- Use the one with best worst-case performance

## Implementation Guidelines

- **Language**: C++ for speed (most competitive programming is C++)
- **Template**: Use a competitive programming template with common macros
- **Testing**: Always test against examples before submitting
- **Edge cases**: Empty input, single element, maximum constraints, negative values

## Problem Type Quick Reference

| Type | Key Technique | Typical Complexity |
|------|--------------|-------------------|
| Sorting/Greedy | Custom comparator, greedy choice | O(n log n) |
| Dynamic Programming | State definition, transitions | O(n*k) varies |
| Graph (shortest path) | Dijkstra, BFS, Bellman-Ford | O(E log V) |
| Graph (connectivity) | Union-Find, DFS | O(n α(n)) |
| String matching | KMP, Z-algorithm, suffix array | O(n) |
| Range queries | Segment tree, BIT | O(n log n) |
| Geometry | Convex hull, sweep line | O(n log n) |
| Math/Number theory | Sieve, modular arithmetic | varies |

## Common Pitfalls

- Integer overflow (use long long in C++)
- Off-by-one errors in loops and array indexing
- Not handling edge cases (empty input, n=1)
- TLE from wrong algorithm complexity (brute force on large input)
- MLE from unnecessary memory allocation
- Wrong output format (trailing spaces, newlines)

## What Worked in Past Competitions

- Read the constraints carefully — they tell you the expected complexity
- Start with the simplest correct solution, optimize only if needed
- Competitive programmers who win use standard algorithms, not novel ones
- Testing against brute-force catches most bugs
