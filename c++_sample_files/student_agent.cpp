#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <map>
#include <array>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cmath>
#include <random>
#include <memory>
#include <limits>
#include <cstring>

namespace py = pybind11;

struct Move
{
    std::string action;
    std::vector<int> from;
    std::vector<int> to;
    std::vector<int> pushed_to;
    std::string orientation;
};

namespace
{
    // Compact board representation: each cell is a uint8_t
    // bits 0-1: side (0=empty, 1=stone, 2=river)
    // bit 2: owner (0=circle, 1=square)
    // bit 3: orientation (0=horizontal, 1=vertical) - only for rivers

    constexpr uint8_t EMPTY = 0;
    constexpr uint8_t STONE = 1;
    constexpr uint8_t RIVER = 2;
    constexpr uint8_t SIDE_MASK = 3;
    constexpr uint8_t OWNER_CIRCLE = 0;
    constexpr uint8_t OWNER_SQUARE = 4;
    constexpr uint8_t OWNER_MASK = 4;
    constexpr uint8_t ORIENT_HORIZONTAL = 0;
    constexpr uint8_t ORIENT_VERTICAL = 8;
    constexpr uint8_t ORIENT_MASK = 8;

    using CompactBoard = std::vector<uint8_t>;

    inline int top_score_row() { return 2; }
    inline int bottom_score_row(int rows) { return rows - 3; }
    inline bool in_bounds(int x, int y, int rows, int cols) { return 0 <= x && x < cols && 0 <= y && y < rows; }

    // Convert string board to compact representation
    inline CompactBoard to_compact(const std::vector<std::vector<std::map<std::string, std::string>>> &board,
                                   int rows, int cols)
    {
        CompactBoard compact(rows * cols, EMPTY);
        for (int y = 0; y < rows; ++y)
        {
            for (int x = 0; x < cols; ++x)
            {
                const auto &cell = board[y][x];
                if (cell.empty())
                    continue;

                uint8_t val = 0;
                auto side_it = cell.find("side");
                if (side_it != cell.end())
                {
                    if (side_it->second == "stone")
                        val |= STONE;
                    else if (side_it->second == "river")
                        val |= RIVER;
                }

                auto owner_it = cell.find("owner");
                if (owner_it != cell.end() && owner_it->second == "square")
                    val |= OWNER_SQUARE;

                auto orient_it = cell.find("orientation");
                if (orient_it != cell.end() && orient_it->second == "vertical")
                    val |= ORIENT_VERTICAL;

                compact[y * cols + x] = val;
            }
        }
        return compact;
    }

    // Convert compact board back to string representation
    inline std::vector<std::vector<std::map<std::string, std::string>>> from_compact(
        const CompactBoard &compact, int rows, int cols)
    {
        std::vector<std::vector<std::map<std::string, std::string>>> board(rows,
                                                                           std::vector<std::map<std::string, std::string>>(cols));

        for (int y = 0; y < rows; ++y)
        {
            for (int x = 0; x < cols; ++x)
            {
                uint8_t val = compact[y * cols + x];
                if (val == EMPTY)
                    continue;

                auto &cell = board[y][x];
                uint8_t side = val & SIDE_MASK;
                if (side == STONE)
                    cell["side"] = "stone";
                else if (side == RIVER)
                    cell["side"] = "river";

                cell["owner"] = (val & OWNER_MASK) ? "square" : "circle";

                if (side == RIVER)
                    cell["orientation"] = (val & ORIENT_MASK) ? "vertical" : "horizontal";
            }
        }
        return board;
    }

    inline bool is_opponent_score_cell(int x, int y, uint8_t player, int rows, int cols, const std::vector<int> &score_cols)
    {
        if (player == OWNER_CIRCLE)
            return y == bottom_score_row(rows) && std::find(score_cols.begin(), score_cols.end(), x) != score_cols.end();
        return y == top_score_row() && std::find(score_cols.begin(), score_cols.end(), x) != score_cols.end();
    }

    inline std::vector<std::pair<int, int>> river_flow_destinations(
        const CompactBoard &board,
        int rx, int ry, int sx, int sy, uint8_t player,
        int rows, int cols, const std::vector<int> &score_cols, bool river_push)
    {
        std::vector<std::pair<int, int>> dest;
        std::vector<uint8_t> visited(rows * cols, 0);
        std::vector<std::pair<int, int>> q{{rx, ry}};
        size_t qi = 0;

        auto cell_at = [&](int x, int y) -> uint8_t
        {
            if (!in_bounds(x, y, rows, cols))
                return EMPTY;
            return board[y * cols + x];
        };

        while (qi < q.size())
        {
            auto [x, y] = q[qi++];

            // Check visited and bounds BEFORE accessing
            if (!in_bounds(x, y, rows, cols))
                continue;
            int id = y * cols + x;
            if (visited[id])
                continue;

            visited[id] = 1;

            // Determine which cell to check
            uint8_t cell = (river_push && x == rx && y == ry) ? cell_at(sx, sy) : cell_at(x, y);

            if (cell == EMPTY)
            {
                if (is_opponent_score_cell(x, y, player, rows, cols, score_cols))
                {
                    // Block entering opponent score - do nothing (don't add to destinations)
                }
                else
                {
                    dest.emplace_back(x, y);
                }
                continue;
            }

            uint8_t side = cell & SIDE_MASK;
            if (side != RIVER)
                continue;

            bool horiz = !(cell & ORIENT_MASK);
            std::array<std::pair<int, int>, 2> dirs = horiz ? std::array<std::pair<int, int>, 2>{{{1, 0}, {-1, 0}}}
                                                            : std::array<std::pair<int, int>, 2>{{{0, 1}, {0, -1}}};

            for (auto [dx, dy] : dirs)
            {
                int nx = x + dx, ny = y + dy;
                while (in_bounds(nx, ny, rows, cols))
                {
                    if (is_opponent_score_cell(nx, ny, player, rows, cols, score_cols))
                        break;

                    uint8_t next_cell = board[ny * cols + nx];

                    if (next_cell == EMPTY)
                    {
                        dest.emplace_back(nx, ny);
                        nx += dx;
                        ny += dy;
                        continue;
                    }

                    // Skip the source cell (sx, sy)
                    if (nx == sx && ny == sy)
                    {
                        nx += dx;
                        ny += dy;
                        continue;
                    }

                    if ((next_cell & SIDE_MASK) == RIVER)
                    {
                        q.emplace_back(nx, ny);
                        break;
                    }

                    break;
                }
            }
        }

        // Remove duplicates
        std::vector<uint8_t> seen(rows * cols, 0);
        std::vector<std::pair<int, int>> out;
        out.reserve(dest.size());
        for (auto &d : dest)
        {
            int id = d.second * cols + d.first;
            if (!seen[id])
            {
                seen[id] = 1;
                out.push_back(d);
            }
        }
        return out;
    }

    struct Targets
    {
        std::vector<std::pair<int, int>> moves;
        std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> pushes;
    };

    inline Targets compute_valid_targets(const CompactBoard &board,
                                         int sx, int sy, uint8_t player,
                                         int rows, int cols, const std::vector<int> &score_cols)
    {
        Targets t;
        if (!in_bounds(sx, sy, rows, cols))
            return t;

        uint8_t cell = board[sy * cols + sx];
        if (cell == EMPTY || (cell & OWNER_MASK) != player)
            return t;

        bool isStone = (cell & SIDE_MASK) == STONE;
        std::array<std::pair<int, int>, 4> dirs{{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};

        for (auto [dx, dy] : dirs)
        {
            int tx = sx + dx, ty = sy + dy;
            if (!in_bounds(tx, ty, rows, cols))
                continue;
            if (is_opponent_score_cell(tx, ty, player, rows, cols, score_cols))
                continue;

            uint8_t target = board[ty * cols + tx];
            if (target == EMPTY)
            {
                t.moves.emplace_back(tx, ty);
            }
            else
            {
                uint8_t target_side = target & SIDE_MASK;
                if (target_side == RIVER)
                {
                    auto dests = river_flow_destinations(board, tx, ty, sx, sy, player, rows, cols, score_cols, false);
                    for (auto &d : dests)
                        t.moves.push_back(d);
                }
                else
                {
                    if (isStone)
                    {
                        int px = tx + dx, py = ty + dy;
                        if (!in_bounds(px, py, rows, cols))
                            continue;
                        if (board[py * cols + px] != EMPTY)
                            continue;

                        // Check ownership of the pushed piece
                        uint8_t pushed_owner = target & OWNER_MASK;

                        // If pushing our own piece, can't push into opponent's scoring area
                        // If pushing opponent's piece, can't push into our scoring area
                        if (pushed_owner == player)
                        {
                            // Pushing our own piece - check opponent's scoring area
                            if (is_opponent_score_cell(px, py, player, rows, cols, score_cols))
                                continue;
                        }
                        else
                        {
                            // Pushing opponent's piece - check our scoring area
                            uint8_t opponent = (player == OWNER_CIRCLE) ? OWNER_SQUARE : OWNER_CIRCLE;
                            if (is_opponent_score_cell(px, py, opponent, rows, cols, score_cols))
                                continue;

                            
                        }

                        t.pushes.emplace_back(std::make_pair(tx, ty), std::make_pair(px, py));
                    }
                    else
                    {
                        auto dests = river_flow_destinations(board, sx, sy, tx, ty, player, rows, cols, score_cols, true);
                        for (auto &d : dests)
                            t.pushes.emplace_back(std::make_pair(tx, ty), d);
                    }
                }
            }
        }

        std::vector<uint8_t> seen(rows * cols, 0);
        std::vector<std::pair<int, int>> uniq;
        uniq.reserve(t.moves.size());
        for (auto &m : t.moves)
        {
            int id = m.second * cols + m.first;
            if (!seen[id])
            {
                seen[id] = 1;
                uniq.push_back(m);
            }
        }
        t.moves.swap(uniq);
        return t;
    }

    static void apply_move(CompactBoard &b, const Move &m, int rows, int cols)
    {
        int from_idx = m.from[1] * cols + m.from[0];
        uint8_t &from_cell = b[from_idx];

        if (m.action == "move")
        {
            int to_idx = m.to[1] * cols + m.to[0];
            b[to_idx] = from_cell;
            from_cell = EMPTY;
        }
        else if (m.action == "push")
        {
            bool pusher_river = (from_cell & SIDE_MASK) == RIVER;
            if (pusher_river)
            {
                from_cell = (from_cell & OWNER_MASK) | STONE;
            }
            int to_idx = m.to[1] * cols + m.to[0];
            int pushed_idx = m.pushed_to[1] * cols + m.pushed_to[0];

            // Move the pushed piece to its destination
            b[pushed_idx] = b[to_idx];

            // Move the pusher to the vacated cell
            b[to_idx] = from_cell;
            from_cell = EMPTY;
        }
        else if (m.action == "flip")
        {
            if ((from_cell & SIDE_MASK) == STONE)
            {
                uint8_t owner = from_cell & OWNER_MASK;
                uint8_t orient = (m.orientation == "vertical") ? ORIENT_VERTICAL : ORIENT_HORIZONTAL;
                from_cell = RIVER | owner | orient;
            }
            else
            {
                from_cell = (from_cell & OWNER_MASK) | STONE;
            }
        }
        else if (m.action == "rotate")
        {
            from_cell ^= ORIENT_MASK;
        }
    }

    inline int stones_in_SA(const CompactBoard &board, uint8_t player,
                            int rows, int cols, const std::vector<int> &score_cols)
    {
        int row = (player == OWNER_CIRCLE) ? top_score_row() : bottom_score_row(rows);
        int cnt = 0;
        for (int x : score_cols)
        {
            uint8_t c = board[row * cols + x];
            if (c != EMPTY && (c & OWNER_MASK) == player && (c & SIDE_MASK) == STONE)
                ++cnt;
        }
        return cnt;
    }

    // ...existing code...

    inline double evaluate_position(const CompactBoard &board, uint8_t root,
                                    int rows, int cols, const std::vector<int> &score_cols, int win_count)
    {
        uint8_t opp = (root == OWNER_CIRCLE) ? OWNER_SQUARE : OWNER_CIRCLE;

        auto targetRowMe = (root == OWNER_CIRCLE) ? top_score_row() : bottom_score_row(rows);
        auto targetRowOpp = (root == OWNER_CIRCLE) ? bottom_score_row(rows) : top_score_row();

        auto nearest_dx = [&](int x)
        {
            int best = 1e9;
            for (int c : score_cols)
                best = std::min(best, std::abs(x - c));
            return best;
        };

        // Helper function to evaluate river strategic value
        auto evaluate_river = [&](int rx, int ry, uint8_t river_owner) -> double
        {
            double score = 0.0;

            // Count adjacent pieces that can interact with this river
            int my_adjacent = 0;
            int opp_adjacent = 0;
            std::array<std::pair<int, int>, 4> dirs{{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};

            for (auto [dx, dy] : dirs)
            {
                int ax = rx + dx, ay = ry + dy;
                if (!in_bounds(ax, ay, rows, cols))
                    continue;

                uint8_t adj = board[ay * cols + ax];
                if (adj == EMPTY || (adj & SIDE_MASK) == RIVER)
                    continue;

                if ((adj & OWNER_MASK) == river_owner)
                {
                    my_adjacent++;
                }
                else
                {
                    opp_adjacent++;
                }
            }

            // Score based on adjacent pieces
            //score += my_adjacent * 0.5;  // More of my pieces can use this river
            //score -= opp_adjacent * 0.3; // Opponent can also use it

            // Evaluate how close pieces can reach to opponent's scoring area via this river
            auto dests = river_flow_destinations(board, rx, ry, rx, ry, river_owner, rows, cols, score_cols, false);

            double best_reach = 0.0;
            int opponent_target_row = (river_owner == OWNER_CIRCLE) ? top_score_row() : bottom_score_row(rows);

            for (auto [dx_dest, dy_dest] : dests)
            {
                // How close is this destination to opponent's scoring row?
                int dist_to_target = std::abs(dy_dest - opponent_target_row);
                int dist_to_score_col = nearest_dx(dx_dest);
                double total_dist = dist_to_target + dist_to_score_col;

                // Better rivers reach closer to opponent's scoring area
                double reach_value = std::exp(-total_dist * 0.4);
                best_reach = std::max(best_reach, reach_value);
            }

            score += best_reach * my_adjacent * 0.5; // Weight for strategic positioning

            return score;
        };

        double myScore = 0.0, oppScore = 0.0;
        for (int y = 0; y < rows; ++y)
        {
            for (int x = 0; x < cols; ++x)
            {
                uint8_t c = board[y * cols + x];
                if (c == EMPTY)
                    continue;

                uint8_t side = c & SIDE_MASK;
                uint8_t owner = c & OWNER_MASK;

                if (side == STONE)
                {
                    if (owner == root)
                    {
                        int dy = std::abs(y - targetRowMe);
                        int dx = nearest_dx(x);
                        double dist = dy + dx;
                        myScore += std::exp(-dist * 0.3);
                    }
                    else
                    {
                        int dy = std::abs(y - targetRowOpp);
                        int dx = nearest_dx(x);
                        double dist = dy + dx;
                        oppScore += std::exp(-dist * 0.3);
                    }
                }
                else if (side == RIVER)
                {
                    // Evaluate river's strategic value
                    double river_value = evaluate_river(x, y, owner);

                    if (owner == root)
                    {
                        myScore += river_value;
                    }
                    else
                    {
                        oppScore += river_value;
                    }
                }
            }
        }

        double total = myScore + oppScore;
        if (total < 1e-9)
            return 0.5;

        return myScore / total;
    }

    // ...existing code...

    struct MCTSNode
    {
        CompactBoard board;
        Move move;
        MCTSNode *parent;
        std::vector<std::unique_ptr<MCTSNode>> children;
        int visits;
        double wins;
        bool is_fully_expanded;
        uint8_t player_to_move;
        std::vector<Move> untried_moves;

        MCTSNode(const CompactBoard &b, const Move &m, MCTSNode *p, uint8_t player)
            : board(b), move(m), parent(p), visits(0), wins(0.0),
              is_fully_expanded(false), player_to_move(player) {}

        bool is_terminal(int rows, int cols, const std::vector<int> &score_cols, int win_count) const
        {
            int circleSA = stones_in_SA(board, OWNER_CIRCLE, rows, cols, score_cols);
            int squareSA = stones_in_SA(board, OWNER_SQUARE, rows, cols, score_cols);
            return circleSA >= win_count || squareSA >= win_count;
        }

        double ucb1(double exploration_constant = 1.414) const
        {
            if (visits == 0)
                return std::numeric_limits<double>::infinity();
            return (wins / visits) + exploration_constant * std::sqrt(std::log(parent->visits) / visits);
        }

        MCTSNode *select_child()
        {
            MCTSNode *best = nullptr;
            double best_ucb = -std::numeric_limits<double>::infinity();
            for (auto &child : children)
            {
                double ucb = child->ucb1();
                if (ucb > best_ucb)
                {
                    best_ucb = ucb;
                    best = child.get();
                }
            }
            return best;
        }
    };

    std::vector<Move> generate_all_moves_internal(const CompactBoard &board, uint8_t player,
                                                  int rows, int cols, const std::vector<int> &score_cols)
    {
        std::vector<Move> out;
        out.reserve(512);

        for (int y = 0; y < rows; ++y)
        {
            for (int x = 0; x < cols; ++x)
            {
                uint8_t cell = board[y * cols + x];
                if (cell == EMPTY || (cell & OWNER_MASK) != player)
                    continue;

                Targets tg = compute_valid_targets(board, x, y, player, rows, cols, score_cols);
                for (auto &mv : tg.moves)
                {
                    Move m;
                    m.action = "move";
                    m.from = {x, y};
                    m.to = {mv.first, mv.second};
                    out.push_back(m);
                }
                for (auto &pp : tg.pushes)
                {
                    Move m;
                    m.action = "push";
                    m.from = {x, y};
                    m.to = {pp.first.first, pp.first.second};
                    m.pushed_to = {pp.second.first, pp.second.second};
                    out.push_back(m);
                }

                if ((cell & SIDE_MASK) == STONE)
                {
                    for (auto ori : {"horizontal", "vertical"})
                    {
                        Move m;
                        m.action = "flip";
                        m.from = {x, y};
                        m.orientation = ori;
                        out.push_back(m);
                    }
                }
                else
                {
                    Move m;
                    m.action = "flip";
                    m.from = {x, y};
                    out.push_back(m);

                    Move r;
                    r.action = "rotate";
                    r.from = {x, y};
                    out.push_back(r);
                }
            }
        }
        return out;
    }

    MCTSNode *expand(MCTSNode *node, int rows, int cols, const std::vector<int> &score_cols, int win_count)
    {
        if (node->untried_moves.empty())
        {
            node->untried_moves = generate_all_moves_internal(node->board, node->player_to_move, rows, cols, score_cols);
            if (node->untried_moves.empty())
            {
                node->is_fully_expanded = true;
                return node;
            }

            // Sort moves by heuristic evaluation (best first)
            std::sort(node->untried_moves.begin(), node->untried_moves.end(),
                      [&](const Move &a, const Move &b)
                      {
                          CompactBoard board_a = node->board, board_b = node->board;
                          apply_move(board_a, a, rows, cols);
                          apply_move(board_b, b, rows, cols);

                          double eval_a = evaluate_position(board_a, node->player_to_move, rows, cols, score_cols, win_count);
                          double eval_b = evaluate_position(board_b, node->player_to_move, rows, cols, score_cols, win_count);

                          return eval_a > eval_b;
                      });
        }

        // Take the best untried move (front of sorted list)
        Move move = node->untried_moves.front();
        node->untried_moves.erase(node->untried_moves.begin());

        if (node->untried_moves.empty())
            node->is_fully_expanded = true;

        auto new_board = node->board;
        apply_move(new_board, move, rows, cols);

        uint8_t next_player = (node->player_to_move == OWNER_CIRCLE) ? OWNER_SQUARE : OWNER_CIRCLE;
        auto child = std::make_unique<MCTSNode>(new_board, move, node, next_player);
        MCTSNode *child_ptr = child.get();
        node->children.push_back(std::move(child));

        return child_ptr;
    }

    double simulate(const CompactBoard &start_board, uint8_t current_player, uint8_t root_player,
                    int rows, int cols, const std::vector<int> &score_cols, int win_count)
    {
        auto board = start_board;
        static std::mt19937 rng(std::random_device{}());

        // Adaptive depth based on how close we are to winning
        int circleSA = stones_in_SA(start_board, OWNER_CIRCLE, rows, cols, score_cols);
        int squareSA = stones_in_SA(start_board, OWNER_SQUARE, rows, cols, score_cols);

        int max_stones = std::max(circleSA, squareSA);
        int stones_needed = win_count - max_stones;

        // More depth if we're far from winning, less if close
        //int max_depth = stones_needed > 3 ? 30 : 15;
        int max_depth = stones_needed > 3 ? 3 : 3;

        double prev_eval = evaluate_position(board, root_player, rows, cols, score_cols, win_count);

        for (int depth = 0; depth < max_depth; ++depth)
        {
            circleSA = stones_in_SA(board, OWNER_CIRCLE, rows, cols, score_cols);
            squareSA = stones_in_SA(board, OWNER_SQUARE, rows, cols, score_cols);

            if (circleSA >= win_count)
                return (root_player == OWNER_CIRCLE) ? 1.0 : 0.0;
            if (squareSA >= win_count)
                return (root_player == OWNER_SQUARE) ? 1.0 : 0.0;

            auto moves = generate_all_moves_internal(board, current_player, rows, cols, score_cols);
            if (moves.empty())
                break;

            // Epsilon-greedy strategy (80% greedy, 20% random)
            Move move;
            std::uniform_real_distribution<double> epsilon_dist(0.0, 1.0);

            if (epsilon_dist(rng) < 1.0 && moves.size() > 1)
            {
                // Greedy: pick best move according to heuristic
                double best_eval = -1.0;
                size_t best_idx = 0;

                // Check top 10 moves to balance speed and quality
                size_t check_count = std::min(moves.size(), size_t(10));
                for (size_t i = 0; i < check_count; ++i)
                {
                    CompactBoard temp = board;
                    apply_move(temp, moves[i], rows, cols);
                    double eval = evaluate_position(temp, current_player, rows, cols, score_cols, win_count);

                    if (eval > best_eval)
                    {
                        best_eval = eval;
                        best_idx = i;
                    }
                }
                move = moves[best_idx];
            }
            else
            {
                // Random exploration
                std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
                move = moves[dist(rng)];
            }

            apply_move(board, move, rows, cols);
            current_player = (current_player == OWNER_CIRCLE) ? OWNER_SQUARE : OWNER_CIRCLE;

            // Early termination based on evaluation
            if (depth % 5 == 0 && depth > 0)
            {
                double curr_eval = evaluate_position(board, root_player, rows, cols, score_cols, win_count);

                // If evaluation is extreme (very good or very bad), stop early
                if (curr_eval > 0.95 || curr_eval < 0.05)
                    return curr_eval;

                // If evaluation hasn't changed much, the position is stable
                if (std::abs(curr_eval - prev_eval) < 0.05)
                    return curr_eval;

                prev_eval = curr_eval;
            }
        }

        return evaluate_position(board, root_player, rows, cols, score_cols, win_count);
    }

    void backpropagate(MCTSNode *node, double result)
    {
        while (node != nullptr)
        {
            node->visits++;
            node->wins += result;
            result = 1.0 - result;
            node = node->parent;
        }
    }

    Move mcts_search(const std::vector<std::vector<std::map<std::string, std::string>>> &board_str,
                     const std::string &player_str, int rows, int cols, const std::vector<int> &score_cols,
                     int win_count, std::chrono::milliseconds time_limit)
    {
        // Convert to compact representation
        CompactBoard board = to_compact(board_str, rows, cols);
        uint8_t player = (player_str == "circle") ? OWNER_CIRCLE : OWNER_SQUARE;

        auto deadline = std::chrono::steady_clock::now() + time_limit;

        Move dummy;
        dummy.action = "move";
        dummy.from = {0, 0};
        dummy.to = {0, 0};

        MCTSNode root(board, dummy, nullptr, player);

        int iterations = 0;
        while (std::chrono::steady_clock::now() < deadline)
        {
            MCTSNode *node = &root;
            while (!node->is_terminal(rows, cols, score_cols, win_count) && node->is_fully_expanded)
            {
                node = node->select_child();
                if (node == nullptr)
                    break;
            }

            if (node == nullptr)
                break;

            if (!node->is_terminal(rows, cols, score_cols, win_count) && !node->is_fully_expanded)
            {
                node = expand(node, rows, cols, score_cols, win_count);
            }

            double result = simulate(node->board, node->player_to_move, player, rows, cols, score_cols, win_count);

            backpropagate(node, result);

            iterations++;
        }

        std::cout << "MCTS iterations: " << iterations << std::endl;

        if (root.children.empty())
        {
            auto moves = generate_all_moves_internal(board, player, rows, cols, score_cols);
            return moves.empty() ? dummy : moves[0];
        }

        MCTSNode *best = nullptr;
        int best_visits = -1;
        for (auto &child : root.children)
        {
            if (child->visits > best_visits)
            {
                best_visits = child->visits;
                best = child.get();
            }
        }

        return best ? best->move : dummy;
    }

} // namespace

class StudentAgent
{
public:
    explicit StudentAgent(std::string side, int history_size = 3)
        : side(std::move(side)), max_history_size(history_size), move_counter(0) {
            // Define the opening moves here based on size
            if (side == "circle") {
                opening_moves = {
                    
                };
            } else {
                opening_moves = {
                    
                };
            }

        }

    std::vector<Move> generate_all_moves(const std::vector<std::vector<std::map<std::string, std::string>>> &board,
                                         int rows, int cols, const std::vector<int> &score_cols) const
    {
        CompactBoard compact = to_compact(board, rows, cols);
        uint8_t player = (side == "circle") ? OWNER_CIRCLE : OWNER_SQUARE;
        auto moves = generate_all_moves_internal(compact, player, rows, cols, score_cols);

        std::vector<Move> filtered;
        filtered.reserve(moves.size());
        for (const auto &m : moves)
        {
            if (!is_in_history(m))
                filtered.push_back(m);
        }

        return filtered.empty() ? moves : filtered;
    }

    Move choose(const std::vector<std::vector<std::map<std::string, std::string>>> &boardIn,
                int rows, int cols, const std::vector<int> &score_cols,
                float my_time, float /*opp_time*/)
    {
        int win_count = (int)score_cols.size();

        // Convert to compact representation for move generation
        CompactBoard board = to_compact(boardIn, rows, cols);
        uint8_t player = (side == "circle") ? OWNER_CIRCLE : OWNER_SQUARE;

        if(move_counter < (int)opening_moves.size())
        {
            Move intended_move = opening_moves[move_counter];
            if(is_valid_move(board, intended_move, player, rows, cols, score_cols) )
            {
                move_counter++;
                add_to_history(intended_move);
                std::cout << "OPENING MOVE\n";
                return intended_move;
            }
            move_counter = 999 ; //skip rest of opening moves if one fails
        }

        // Greedy check: if any move increases our stones-in-SA count, play it
        int before_SA = stones_in_SA(board, player, rows, cols, score_cols);
        auto all_moves = generate_all_moves_internal(board, player, rows, cols, score_cols);

        for (const auto &m : all_moves)
        {
            CompactBoard temp = board;
            apply_move(temp, m, rows, cols);
            int after_SA = stones_in_SA(temp, player, rows, cols, score_cols);
            if (after_SA > before_SA)
            {
                std::cout << "GREEDY: Found move that increases scoring count!\n";
                return m;
            }
        }

        // No immediate scoring-increase move found, proceed with MCTS
        int time_ms = std::min(300, std::max(100, (int)(my_time * 100)));
        Move chosen_move = mcts_search(boardIn, side, rows, cols, score_cols, win_count, std::chrono::milliseconds(time_ms));
        add_to_history(chosen_move);
        return chosen_move;
    }

private:
    std::string side;
    int max_history_size;
    std::vector<Move> move_history;
    int move_counter;
    vector<Move> opening_moves;

    bool moves_equal(const Move &a, const Move &b) const
    {
        return a.action == b.action &&
               a.from == b.from &&
               a.to == b.to &&
               a.pushed_to == b.pushed_to &&
               a.orientation == b.orientation;
    }

    bool is_in_history(const Move &m) const
    {
        for (const auto &hist_move : move_history)
        {
            if (moves_equal(m, hist_move))
                return true;
        }
        return false;
    }

    void add_to_history(const Move &m)
    {
        move_history.push_back(m);
        if ((int)move_history.size() > max_history_size)
        {
            move_history.erase(move_history.begin());
        }
    }
};

PYBIND11_MODULE(student_agent_module, m)
{
    py::class_<Move>(m, "Move")
        .def_readonly("action", &Move::action)
        .def_readonly("from_pos", &Move::from)
        .def_readonly("to_pos", &Move::to)
        .def_readonly("pushed_to", &Move::pushed_to)
        .def_readonly("orientation", &Move::orientation);

    py::class_<StudentAgent>(m, "StudentAgent")
        .def(py::init<std::string>())
        .def(py::init<std::string, int>(), py::arg("side"), py::arg("history_size") = 3)
        .def("generate_all_moves", &StudentAgent::generate_all_moves)
        .def("choose", &StudentAgent::choose);
}