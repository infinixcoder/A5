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
#include <queue>

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
    inline int top_score_row() { return 2; }
    inline int bottom_score_row(int rows) { return rows - 3; }
    inline bool in_bounds(int x, int y, int rows, int cols) { return 0 <= x && x < cols && 0 <= y && y < rows; }

    inline bool is_opponent_score_cell(int x, int y, const std::string &player, int rows, int cols, const std::vector<int> &score_cols)
    {
        if (player == "circle")
            return y == bottom_score_row(rows) && std::find(score_cols.begin(), score_cols.end(), x) != score_cols.end();
        return y == top_score_row() && std::find(score_cols.begin(), score_cols.end(), x) != score_cols.end();
    }

    inline bool is_own_score_cell(int x, int y, const std::string &player, int rows, int cols, const std::vector<int> &score_cols)
    {
        if (player == "circle")
            return y == top_score_row() && std::find(score_cols.begin(), score_cols.end(), x) != score_cols.end();
        return y == bottom_score_row(rows) && std::find(score_cols.begin(), score_cols.end(), x) != score_cols.end();
    }

    inline std::vector<std::pair<int, int>> river_flow_destinations(
        const std::vector<std::vector<std::map<std::string, std::string>>> &board,
        int rx, int ry, int sx, int sy, const std::string &player,
        int rows, int cols, const std::vector<int> &score_cols, bool river_push)
    {
        std::vector<std::pair<int, int>> dest;
        std::vector<uint8_t> visited(rows * cols, 0);
        std::vector<std::pair<int, int>> q{{rx, ry}};
        size_t qi = 0;

        auto cell_at = [&](int x, int y) -> const std::map<std::string, std::string> *
        {
            if (!in_bounds(x, y, rows, cols))
                return nullptr;
            const auto &m = board[y][x];
            return m.empty() ? nullptr : &m;
        };

        while (qi < q.size())
        {
            auto [x, y] = q[qi++];
            if (!in_bounds(x, y, rows, cols))
                continue;
            int id = y * cols + x;
            if (visited[id])
                continue;
            visited[id] = 1;

            const std::map<std::string, std::string> *cell = (river_push && x == rx && y == ry) ? cell_at(sx, sy) : cell_at(x, y);
            if (!cell)
            {
                if (!is_opponent_score_cell(x, y, player, rows, cols, score_cols))
                    dest.emplace_back(x, y);
                continue;
            }
            auto itSide = cell->find("side");
            if (itSide == cell->end() || itSide->second != "river")
                continue;

            bool horiz = cell->at("orientation") == "horizontal";
            std::array<std::pair<int, int>, 2> dirs = horiz ? std::array<std::pair<int, int>, 2>{{{1, 0}, {-1, 0}}}
                                                            : std::array<std::pair<int, int>, 2>{{{0, 1}, {0, -1}}};
            for (auto [dx, dy] : dirs)
            {
                int nx = x + dx, ny = y + dy;
                while (in_bounds(nx, ny, rows, cols))
                {
                    if (is_opponent_score_cell(nx, ny, player, rows, cols, score_cols))
                        break;
                    const auto &nc = board[ny][nx];
                    if (!nc.empty())
                        break;
                    q.emplace_back(nx, ny);
                    nx += dx;
                    ny += dy;
                }
            }
        }

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

    inline Targets compute_valid_targets(const std::vector<std::vector<std::map<std::string, std::string>>> &board,
                                         int sx, int sy, const std::string &player,
                                         int rows, int cols, const std::vector<int> &score_cols)
    {
        Targets t;
        if (!in_bounds(sx, sy, rows, cols))
            return t;
        const auto &cell = board[sy][sx];
        if (cell.empty() || cell.at("owner") != player)
            return t;

        bool isStone = cell.at("side") == "stone";
        std::array<std::pair<int, int>, 4> dirs{{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
        for (auto [dx, dy] : dirs)
        {
            int tx = sx + dx, ty = sy + dy;
            if (!in_bounds(tx, ty, rows, cols))
                continue;
            if (is_opponent_score_cell(tx, ty, player, rows, cols, score_cols))
                continue;
            const auto &target = board[ty][tx];
            if (target.empty())
            {
                t.moves.emplace_back(tx, ty);
            }
            else
            {
                if (target.at("side") == "river")
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
                        if (in_bounds(px, py, rows, cols) && !is_opponent_score_cell(px, py, player, rows, cols, score_cols) && board[py][px].empty())
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

    static void apply_move(std::vector<std::vector<std::map<std::string, std::string>>> &b, const Move &m)
    {
        auto &from = b[m.from[1]][m.from[0]];
        if (m.action == "move")
        {
            auto piece = from;
            from.clear();
            b[m.to[1]][m.to[0]] = piece;
        }
        else if (m.action == "push")
        {
            bool pusher_river = (!from.empty() && from.at("side") == "river");
            if (pusher_river)
            {
                from["side"] = "stone";
                from.erase("orientation");
            }
            auto pushed = b[m.to[1]][m.to[0]];
            b[m.to[1]][m.to[0]].clear();
            b[m.pushed_to[1]][m.pushed_to[0]] = pushed;
        }
        else if (m.action == "flip")
        {
            if (from.at("side") == "stone")
            {
                from["side"] = "river";
                from["orientation"] = m.orientation;
            }
            else
            {
                from["side"] = "stone";
                from.erase("orientation");
            }
        }
        else if (m.action == "rotate")
        {
            from["orientation"] = (from.at("orientation") == "horizontal") ? "vertical" : "horizontal";
        }
    }

    inline int stones_in_SA(const std::vector<std::vector<std::map<std::string, std::string>>> &board,
                            const std::string &me, int rows, int cols, const std::vector<int> &score_cols)
    {
        int row = (me == "circle") ? top_score_row() : bottom_score_row(rows);
        int cnt = 0;
        for (int x : score_cols)
        {
            const auto &c = board[row][x];
            if (!c.empty() && c.at("owner") == me && c.at("side") == "stone")
                ++cnt;
        }
        return cnt;
    }

    // BFS to find shortest path distance to scoring area
    inline int bfs_distance_to_SA(const std::vector<std::vector<std::map<std::string, std::string>>> &board,
                                  int sx, int sy, const std::string &player,
                                  int rows, int cols, const std::vector<int> &score_cols)
    {
        std::vector<std::vector<int>> dist(rows, std::vector<int>(cols, -1));
        std::queue<std::pair<int, int>> q;
        q.push({sx, sy});
        dist[sy][sx] = 0;

        while (!q.empty())
        {
            auto [x, y] = q.front();
            q.pop();

            // Check if we reached scoring area
            if (is_own_score_cell(x, y, player, rows, cols, score_cols))
                return dist[y][x];

            // Try all adjacent cells and river flows
            std::array<std::pair<int, int>, 4> dirs{{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
            for (auto [dx, dy] : dirs)
            {
                int nx = x + dx, ny = y + dy;
                if (!in_bounds(nx, ny, rows, cols) || dist[ny][nx] != -1)
                    continue;
                if (is_opponent_score_cell(nx, ny, player, rows, cols, score_cols))
                    continue;

                const auto &cell = board[ny][nx];
                if (cell.empty())
                {
                    dist[ny][nx] = dist[y][x] + 1;
                    q.push({nx, ny});
                }
                else if (cell.at("side") == "river")
                {
                    // Can use river flow
                    auto dests = river_flow_destinations(board, nx, ny, x, y, player, rows, cols, score_cols, false);
                    for (auto [rx, ry] : dests)
                    {
                        if (dist[ry][rx] == -1)
                        {
                            dist[ry][rx] = dist[y][x] + 1;
                            q.push({rx, ry});
                        }
                    }
                }
            }
        }
        return 1000; // No path found
    }

    // Check if a river at (x, y) with given orientation creates a useful path
    inline double evaluate_river_placement(const std::vector<std::vector<std::map<std::string, std::string>>> &board,
                                           int x, int y, const std::string &orientation, const std::string &player,
                                           int rows, int cols, const std::vector<int> &score_cols)
    {
        double score = 0.0;

        // Create temporary board with this river
        auto test_board = board;
        test_board[y][x]["side"] = "river";
        test_board[y][x]["orientation"] = orientation;
        test_board[y][x]["owner"] = player;

        // Check how many of our stones can now reach SA faster
        int improvement_count = 0;
        double total_improvement = 0.0;

        for (int py = 0; py < rows; ++py)
        {
            for (int px = 0; px < cols; ++px)
            {
                const auto &cell = board[py][px];
                if (cell.empty() || cell.at("owner") != player || cell.at("side") != "stone")
                    continue;

                int old_dist = bfs_distance_to_SA(board, px, py, player, rows, cols, score_cols);
                int new_dist = bfs_distance_to_SA(test_board, px, py, player, rows, cols, score_cols);

                if (new_dist < old_dist)
                {
                    improvement_count++;
                    total_improvement += (old_dist - new_dist);
                }
            }
        }

        // Bonus for creating river networks (rivers that connect)
        int connected_rivers = 0;
        std::array<std::pair<int, int>, 4> dirs{{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
        for (auto [dx, dy] : dirs)
        {
            int nx = x + dx, ny = y + dy;
            if (!in_bounds(nx, ny, rows, cols))
                continue;
            const auto &neighbor = board[ny][nx];
            if (!neighbor.empty() && neighbor.at("owner") == player && neighbor.at("side") == "river")
                connected_rivers++;
        }

        // Bonus for rivers that point toward scoring area
        int target_row = (player == "circle") ? top_score_row() : bottom_score_row(rows);
        bool points_toward_sa = false;
        if (orientation == "vertical")
        {
            points_toward_sa = true; // Vertical rivers always help movement toward SA
        }
        else // horizontal
        {
            // Horizontal rivers help if we're in a scoring column or adjacent
            for (int sc : score_cols)
            {
                if (std::abs(x - sc) <= 2)
                {
                    points_toward_sa = true;
                    break;
                }
            }
        }

        score += improvement_count * 2.0;
        score += total_improvement * 0.5;
        score += connected_rivers * 1.5;
        score += points_toward_sa ? 1.0 : 0.0;

        // Bonus for rivers closer to scoring area
        int dist_to_sa = std::abs(y - target_row);
        score += (rows - dist_to_sa) * 0.1;

        return score;
    }

    inline double evaluate_position(const std::vector<std::vector<std::map<std::string, std::string>>> &board,
                                    const std::string &root, int rows, int cols, const std::vector<int> &score_cols, int win_count)
    {
        std::string opp = (root == "circle") ? "square" : "circle";
        int mySA = stones_in_SA(board, root, rows, cols, score_cols);
        int opSA = stones_in_SA(board, opp, rows, cols, score_cols);

        if (mySA >= win_count)
            return 1.0;
        if (opSA >= win_count)
            return 0.0;

        double score = 0.5;

        // Scoring area advantage (highest priority)
        score += (mySA - opSA) * 0.1;

        // Count stones with fast paths to SA via rivers
        int my_reachable = 0, opp_reachable = 0;
        for (int y = 0; y < rows; ++y)
        {
            for (int x = 0; x < cols; ++x)
            {
                const auto &c = board[y][x];
                if (c.empty() || c.at("side") != "stone")
                    continue;

                if (c.at("owner") == root)
                {
                    int dist = bfs_distance_to_SA(board, x, y, root, rows, cols, score_cols);
                    if (dist <= 3) // Reachable in 3 moves or less
                        my_reachable++;
                    score += 0.01 / (1.0 + dist); // Proximity bonus
                }
                else
                {
                    int dist = bfs_distance_to_SA(board, x, y, opp, rows, cols, score_cols);
                    if (dist <= 3)
                        opp_reachable++;
                    score -= 0.01 / (1.0 + dist);
                }
            }
        }

        score += (my_reachable - opp_reachable) * 0.05;

        // River network evaluation
        int my_rivers = 0, my_strategic_rivers = 0;
        int opp_rivers = 0;

        for (int y = 0; y < rows; ++y)
        {
            for (int x = 0; x < cols; ++x)
            {
                const auto &c = board[y][x];
                if (c.empty() || c.at("side") != "river")
                    continue;

                if (c.at("owner") == root)
                {
                    my_rivers++;
                    // Check if this river creates a strategic path
                    double river_value = evaluate_river_placement(board, x, y, c.at("orientation"), root, rows, cols, score_cols);
                    if (river_value > 2.0)
                        my_strategic_rivers++;
                    score += river_value * 0.005;
                }
                else
                {
                    opp_rivers++;
                }
            }
        }

        // Bonus for maintaining rivers while having stones
        int my_stones = 0;
        for (int y = 0; y < rows; ++y)
        {
            for (int x = 0; x < cols; ++x)
            {
                const auto &c = board[y][x];
                if (!c.empty() && c.at("owner") == root && c.at("side") == "stone")
                    my_stones++;
            }
        }

        // Encourage having rivers when you have stones to move
        if (my_stones > 0 && my_rivers > 0)
            score += std::min(my_rivers, my_stones / 2) * 0.02;

        return std::max(0.0, std::min(1.0, score));
    }

    // MCTS Node
    struct MCTSNode
    {
        std::vector<std::vector<std::map<std::string, std::string>>> board;
        Move move;
        MCTSNode *parent;
        std::vector<std::unique_ptr<MCTSNode>> children;
        int visits;
        double wins;
        bool is_fully_expanded;
        std::string player_to_move;
        std::vector<Move> untried_moves;

        MCTSNode(const std::vector<std::vector<std::map<std::string, std::string>>> &b,
                 const Move &m, MCTSNode *p, const std::string &player)
            : board(b), move(m), parent(p), visits(0), wins(0.0),
              is_fully_expanded(false), player_to_move(player) {}

        bool is_terminal(int rows, int cols, const std::vector<int> &score_cols, int win_count) const
        {
            int circleSA = stones_in_SA(board, "circle", rows, cols, score_cols);
            int squareSA = stones_in_SA(board, "square", rows, cols, score_cols);
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

    std::vector<Move> generate_all_moves_internal(const std::vector<std::vector<std::map<std::string, std::string>>> &board,
                                                  const std::string &player, int rows, int cols, const std::vector<int> &score_cols)
    {
        std::vector<Move> out;
        out.reserve(512);
        for (int y = 0; y < rows; ++y)
        {
            for (int x = 0; x < cols; ++x)
            {
                const auto &cell = board[y][x];
                if (cell.empty() || cell.at("owner") != player)
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

                if (cell.at("side") == "stone")
                {
                    // Prioritize strategic river placements
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

    MCTSNode *expand(MCTSNode *node, int rows, int cols, const std::vector<int> &score_cols)
    {
        if (node->untried_moves.empty())
        {
            node->untried_moves = generate_all_moves_internal(node->board, node->player_to_move, rows, cols, score_cols);
            if (node->untried_moves.empty())
            {
                node->is_fully_expanded = true;
                return node;
            }
        }

        // Prioritize strategic moves during expansion
        static std::mt19937 rng(std::random_device{}());

        // Separate moves by type and evaluate flips
        std::vector<size_t> strategic_flips;
        std::vector<size_t> other_moves;

        for (size_t i = 0; i < node->untried_moves.size(); ++i)
        {
            const auto &move = node->untried_moves[i];
            if (move.action == "flip" && !move.orientation.empty())
            {
                // Evaluate this river placement
                double value = evaluate_river_placement(node->board, move.from[0], move.from[1],
                                                        move.orientation, node->player_to_move,
                                                        rows, cols, score_cols);
                if (value > 2.0) // Strategic threshold
                    strategic_flips.push_back(i);
                else
                    other_moves.push_back(i);
            }
            else
            {
                other_moves.push_back(i);
            }
        }

        // Prefer strategic flips 40% of the time
        size_t idx;
        if (!strategic_flips.empty() && (other_moves.empty() || std::uniform_real_distribution<>(0, 1)(rng) < 0.4))
        {
            std::uniform_int_distribution<size_t> dist(0, strategic_flips.size() - 1);
            idx = strategic_flips[dist(rng)];
        }
        else if (!other_moves.empty())
        {
            std::uniform_int_distribution<size_t> dist(0, other_moves.size() - 1);
            idx = other_moves[dist(rng)];
        }
        else
        {
            std::uniform_int_distribution<size_t> dist(0, node->untried_moves.size() - 1);
            idx = dist(rng);
        }

        Move move = node->untried_moves[idx];
        node->untried_moves.erase(node->untried_moves.begin() + idx);

        if (node->untried_moves.empty())
            node->is_fully_expanded = true;

        auto new_board = node->board;
        apply_move(new_board, move);

        std::string next_player = (node->player_to_move == "circle") ? "square" : "circle";
        auto child = std::make_unique<MCTSNode>(new_board, move, node, next_player);
        MCTSNode *child_ptr = child.get();
        node->children.push_back(std::move(child));

        return child_ptr;
    }

    double simulate(const std::vector<std::vector<std::map<std::string, std::string>>> &start_board,
                    std::string current_player, const std::string &root_player,
                    int rows, int cols, const std::vector<int> &score_cols, int win_count,
                    int max_depth = 30)
    {
        auto board = start_board;
        static std::mt19937 rng(std::random_device{}());

        for (int depth = 0; depth < max_depth; ++depth)
        {
            int circleSA = stones_in_SA(board, "circle", rows, cols, score_cols);
            int squareSA = stones_in_SA(board, "square", rows, cols, score_cols);

            if (circleSA >= win_count)
                return (root_player == "circle") ? 1.0 : 0.0;
            if (squareSA >= win_count)
                return (root_player == "square") ? 1.0 : 0.0;

            auto moves = generate_all_moves_internal(board, current_player, rows, cols, score_cols);
            if (moves.empty())
                break;

            // Biased random selection favoring strategic moves
            Move selected_move;
            std::vector<Move> strategic_moves;
            for (const auto &move : moves)
            {
                if (move.action == "move" || move.action == "push")
                {
                    // Check if move gets closer to SA
                    int dist_before = bfs_distance_to_SA(board, move.from[0], move.from[1], current_player, rows, cols, score_cols);
                    auto test_board = board;
                    apply_move(test_board, move);
                    int tx = (move.action == "move") ? move.to[0] : move.from[0];
                    int ty = (move.action == "move") ? move.to[1] : move.from[1];
                    int dist_after = bfs_distance_to_SA(test_board, tx, ty, current_player, rows, cols, score_cols);

                    if (dist_after < dist_before)
                        strategic_moves.push_back(move);
                }
                else if (move.action == "flip" && !move.orientation.empty())
                {
                    double value = evaluate_river_placement(board, move.from[0], move.from[1],
                                                            move.orientation, current_player,
                                                            rows, cols, score_cols);
                    if (value > 2.0)
                        strategic_moves.push_back(move);
                }
            }

            if (!strategic_moves.empty() && std::uniform_real_distribution<>(0, 1)(rng) < 0.6)
            {
                std::uniform_int_distribution<size_t> dist(0, strategic_moves.size() - 1);
                selected_move = strategic_moves[dist(rng)];
            }
            else
            {
                std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
                selected_move = moves[dist(rng)];
            }

            apply_move(board, selected_move);
            current_player = (current_player == "circle") ? "square" : "circle";
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

    Move mcts_search(const std::vector<std::vector<std::map<std::string, std::string>>> &board,
                     const std::string &player, int rows, int cols, const std::vector<int> &score_cols,
                     int win_count, std::chrono::milliseconds time_limit)
    {
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
                node = expand(node, rows, cols, score_cols);
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
    explicit StudentAgent(std::string side) : side(std::move(side)) {}

    std::vector<Move> generate_all_moves(const std::vector<std::vector<std::map<std::string, std::string>>> &board,
                                         int rows, int cols, const std::vector<int> &score_cols) const
    {
        return generate_all_moves_internal(board, side, rows, cols, score_cols);
    }

    Move choose(const std::vector<std::vector<std::map<std::string, std::string>>> &boardIn,
                int rows, int cols, const std::vector<int> &score_cols,
                float my_time, float /*opp_time*/)
    {
        int win_count = (int)score_cols.size();
        int time_ms = std::min(2000, std::max(100, (int)(my_time * 100)));

        return mcts_search(boardIn, side, rows, cols, score_cols, win_count, std::chrono::milliseconds(time_ms));
    }

private:
    std::string side;
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
        .def("generate_all_moves", &StudentAgent::generate_all_moves)
        .def("choose", &StudentAgent::choose);
}