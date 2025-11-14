# MCTS-Based Game Agent for River Game

## Overview

This C++ implementation provides an AI agent for a two-player board game using **Monte Carlo Tree Search (MCTS)** with several optimizations. The agent uses a compact board representation for computational efficiency and includes smart heuristics for move evaluation.

## Architecture

### 1. Compact Board Representation

Instead of using string-based maps, the board uses a compact `uint8_t` array where each cell is encoded using bitwise operations:

- **Bits 0-1**: Side type (0=empty, 1=stone, 2=river)
- **Bit 2**: Owner (0=circle, 1=square)
- **Bit 3**: Orientation (0=horizontal, 1=vertical) for rivers

This reduces memory usage from ~100+ bytes per cell to just 1 byte, providing 10-100x speedup.

### 2. MCTS Algorithm

The agent implements four-phase MCTS:

#### **Selection**
- Traverses from root to leaf using UCB1 (Upper Confidence Bound)
- Formula: `wins/visits + C * sqrt(log(parent_visits)/visits)`
- Balances exploitation (high win rate) vs exploration (less visited nodes)

#### **Expansion**
- **Smart Expansion**: Moves are sorted by heuristic evaluation before trying
- Best moves are expanded first instead of random selection
- Significantly improves tree quality

#### **Simulation (Playout)**
- **Epsilon-Greedy Strategy**: 80% greedy, 20% random
- Greedy phase evaluates top 10 moves and picks the best
- **Adaptive Depth**: 30 moves when far from winning, 15 when close
- **Early Termination**: Stops if evaluation is extreme (>0.95 or <0.05) or stable

#### **Backpropagation**
- Updates visit counts and win statistics up the tree
- Alternates result (1.0 - result) for opponent nodes

### 3. Evaluation Function

Uses exponential distance-based heuristic:

For each stone:
- Calculates Manhattan distance to target scoring row
- Calculates nearest distance to scoring columns
- Applies exponential reward (closer pieces score exponentially higher)
- Returns normalized score between [0, 1]

### 4. Move History Tracking

Maintains a sliding window of recent moves to avoid repetition:
- Configurable history size (default: 3)
- Filters out moves in history when generating legal moves
- Prevents simple move loops

## Adjustable Parameters

### MCTS Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `exploration_constant` | `MCTSNode::ucb1()` | 1.414 | UCB1 exploration constant. Higher = more exploration |
| `time_ms` | `StudentAgent::choose()` | min(400, max(100, my_time*100)) | Time limit per move in milliseconds |

### Simulation Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `epsilon` | `simulate()` | 0.8 | Greedy move probability (0.8 = 80% greedy, 20% random) |
| `check_count` | `simulate()` | 10 | Number of top moves to evaluate in greedy phase |
| `max_depth_far` | `simulate()` | 30 | Simulation depth when far from winning |
| `max_depth_near` | `simulate()` | 15 | Simulation depth when close to winning |
| `stones_threshold` | `simulate()` | 3 | Threshold for "close to winning" |
| `eval_check_interval` | `simulate()` | 5 | How often to check evaluation for early termination |
| `extreme_eval_threshold` | `simulate()` | 0.95/0.05 | Threshold for extreme position evaluation |
| `stable_eval_threshold` | `simulate()` | 0.05 | Threshold for position stability |

### Evaluation Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `distance_factor` | `evaluate_position()` | 0.3 | Exponential decay factor for distance scoring |

### Move Generation Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `history_size` | `StudentAgent` constructor | 3 | Number of recent moves to track and avoid |

### Board-Specific Constants

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `top_score_row()` | Global | 2 | Row index for circle's scoring area |
| `bottom_score_row(rows)` | Global | rows - 3 | Row index for square's scoring area |

## How to Tune Parameters

### For Stronger Play (More Computation)

```cpp
// Increase thinking time
int time_ms = std::min(5000, std::max(100, (int)(my_time * 100)));

// More greedy simulations
if (epsilon_dist(rng) < 0.9 && moves.size() > 1)  // 90% greedy

// Check more moves in greedy phase
size_t check_count = std::min(moves.size(), size_t(20));

// Deeper simulations
int max_depth = stones_needed > 3 ? 50 : 25;
```

### For Faster Play (Less Computation)

```cpp
// Reduce thinking time
int time_ms = std::min(500, std::max(50, (int)(my_time * 50)));

// More random simulations
if (epsilon_dist(rng) < 0.6 && moves.size() > 1)  // 60% greedy

// Check fewer moves
size_t check_count = std::min(moves.size(), size_t(5));

// Shallower simulations
int max_depth = stones_needed > 3 ? 15 : 8;
```

### For More Exploration

``` cpp
// Increase UCB1 constant
double ucb1(double exploration_constant = 2.0) const

// More random simulations
if (epsilon_dist(rng) < 0.5 && moves.size() > 1)  // 50% greedy
```
### For Better Positional Play

``` cpp
// Adjust distance factor (higher = more aggressive)
myScore += std::exp(-dist * 0.5);

// Disable move history to allow repositioning
StudentAgent("circle", 0)  // history_size = 0
```

## Usage

### Initialization

``` cpp
// Default initialization (history size = 3)
StudentAgent agent("circle");

// Custom history size
StudentAgent agent("square", 5);  // Track last 5 moves
```

### Making a move
``` cpp
Move move = agent.choose(board, rows, cols, score_cols, my_time, opp_time);
```

### Generating Legal Moves
``` cpp
std::vector<Move> moves = agent.generate_all_moves(board, rows, cols, score_cols);
```
## Performance Characteristics
- Memory: ~1 byte per cell (vs ~100+ for string maps)
- Speed: 10-100x faster than string-based implementation
- MCTS Iterations: Typically 1000-5000 iterations per second
- Move Quality: Smart expansion and biased simulation improve quality significantly

## Key Optimizations
- Compact Representation: Bitwise encoding for minimal memory footprint
- Move Ordering: Best moves expanded first in MCTS tree
- Biased Simulation: Greedy playouts instead of pure random
- Adaptive Depth: Adjusts simulation depth based on game state
- Early Termination: Stops simulation when position is decided
- Move History: Avoids repetitive moves


## Future Improvements
Potential enhancements:

- RAVE (Rapid Action Value Estimation): Share statistics across tree
- Transposition Tables: Cache evaluated positions
- Progressive Widening: Gradually expand more children
- Parallel MCTS: Multi-threaded tree search
- Opening Book: Pre-computed best opening moves
Endgame Database: Perfect play in simple endgames