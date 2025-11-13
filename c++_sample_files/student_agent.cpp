#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <map>
#include <random>
#include <array>
#include <algorithm>
#include <iostream>
#include <cstdint>
#include <chrono>
#include <unordered_map>

namespace py = pybind11;

struct Move
{
    std::string action;
    std::vector<int> from;      // [x,y]
    std::vector<int> to;        // [x,y]
    std::vector<int> pushed_to; // [x,y]
    std::string orientation;    // for stone->river flip
};

namespace
{
    inline int top_score_row() { return 2; }
    inline int bottom_score_row(int rows) { return rows - 3; }
    inline bool in_bounds(int x, int y, int rows, int cols) { return 0 <= x && x < cols && 0 <= y && y < rows; }

    struct CompactBoard
    {
        int rows{0}, cols{0};
        std::vector<uint8_t> owner;  // 0 none,1 circle,2 square
        std::vector<uint8_t> side;   // 0 none,1 stone,2 river
        std::vector<uint8_t> orient; // 0 horizontal,1 vertical (river only)
        inline int idx(int x, int y) const { return y * cols + x; }
        inline bool empty(int x, int y) const { return owner[idx(x, y)] == 0; }
        inline bool is_stone(int x, int y) const { return side[idx(x, y)] == 1; }
        inline bool is_river(int x, int y) const { return side[idx(x, y)] == 2; }
    };

    inline uint8_t owner_code(const std::string &s) { return s[0] == 'c' ? 1 : 2; }
    inline uint8_t side_code(const std::string &s) { return s[0] == 's' ? 1 : 2; }
    inline uint8_t orient_code(const std::string &s) { return s[0] == 'h' ? 0 : 1; }

    inline CompactBoard make_compact(const std::vector<std::vector<std::map<std::string, std::string>>> &board)
    {
        CompactBoard cb;
        cb.rows = (int)board.size();
        cb.cols = cb.rows ? (int)board[0].size() : 0;
        int N = cb.rows * cb.cols;
        cb.owner.assign(N, 0);
        cb.side.assign(N, 0);
        cb.orient.assign(N, 0);
        for (int y = 0; y < cb.rows; ++y)
        {
            for (int x = 0; x < cb.cols; ++x)
            {
                const auto &cell = board[y][x];
                if (cell.empty())
                    continue;
                int id = cb.idx(x, y);
                cb.owner[id] = owner_code(cell.at("owner"));
                cb.side[id] = side_code(cell.at("side"));
                if (cb.side[id] == 2)
                    cb.orient[id] = orient_code(cell.at("orientation"));
            }
        }
        return cb;
    }

    inline bool is_opponent_score_cell(int x, int y, const std::string &player, int rows, int cols, const std::vector<int> &score_cols)
    {
        if (player == "circle")
            return (y == bottom_score_row(rows)) && (std::find(score_cols.begin(), score_cols.end(), x) != score_cols.end());
        else
            return (y == top_score_row()) && (std::find(score_cols.begin(), score_cols.end(), x) != score_cols.end());
    }

    inline bool is_own_score_cell(int x, int y, const std::string &player, int rows, const std::vector<int> &score_cols)
    {
        int row = (player == "circle") ? top_score_row() : bottom_score_row(rows);
        return (y == row) && (std::find(score_cols.begin(), score_cols.end(), x) != score_cols.end());
    }

    // River flow destinations (mirrors Python engine).
    inline std::vector<std::pair<int, int>> river_flow_destinations(
        const std::vector<std::vector<std::map<std::string, std::string>>> &board,
        int rx, int ry, int sx, int sy, const std::string &player,
        int rows, int cols, const std::vector<int> &score_cols, bool river_push)
    {
        std::vector<std::pair<int, int>> dest;
        std::vector<uint8_t> visited(rows * cols, 0);
        std::vector<std::pair<int, int>> queue;
        queue.emplace_back(rx, ry);
        size_t qhead = 0;

        auto cell_at = [&](int x, int y) -> const std::map<std::string, std::string> *
        {
            if (!in_bounds(x, y, rows, cols))
                return nullptr;
            const auto &m = board[y][x];
            return m.empty() ? nullptr : &m;
        };

        while (qhead < queue.size())
        {
            auto [x, y] = queue[qhead++];
            if (!in_bounds(x, y, rows, cols))
                continue;
            int id = y * cols + x;
            if (visited[id])
                continue;
            visited[id] = 1;

            // Override starting cell for river_push with pushing river's piece (orientation source).
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
                    const auto *next = cell_at(nx, ny);
                    if (!next)
                    {
                        dest.emplace_back(nx, ny);
                        nx += dx;
                        ny += dy;
                        continue;
                    }
                    if (nx == sx && ny == sy)
                    { // pass origin
                        nx += dx;
                        ny += dy;
                        continue;
                    }
                    auto itS = next->find("side");
                    if (itS != next->end() && itS->second == "river")
                    {
                        queue.emplace_back(nx, ny);
                        break;
                    }
                    break;
                }
            }
        }
        // Dedup
        std::vector<std::pair<int, int>> out;
        out.reserve(dest.size());
        std::vector<uint8_t> seen(rows * cols, 0);
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
        std::vector<std::pair<int, int>> moves;                                  // unique landing squares
        std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> pushes; // ((to),(pushed_to))
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
        // moves & river flows
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
                    auto flow = river_flow_destinations(board, tx, ty, sx, sy, player, rows, cols, score_cols, false);
                    for (auto &d : flow)
                        t.moves.push_back(d);
                }
                else
                {
                    // push attempt
                    if (isStone)
                    {
                        int px = tx + dx, py = ty + dy;
                        if (in_bounds(px, py, rows, cols) &&
                            board[py][px].empty() &&
                            !is_opponent_score_cell(px, py, board[ty][tx].at("owner"), rows, cols, score_cols))
                        {
                            t.pushes.push_back({{tx, ty}, {px, py}});
                        }
                    }
                    else
                    {
                        // river push (only adjacent stone allowed)
                        if (target.at("side") == "stone")
                        {
                            std::string pushed_owner = target.at("owner");
                            auto flow = river_flow_destinations(board, tx, ty, sx, sy, pushed_owner, rows, cols, score_cols, true);
                            for (auto &d : flow)
                            {
                                if (!is_opponent_score_cell(d.first, d.second, pushed_owner, rows, cols, score_cols))
                                    t.pushes.push_back({{tx, ty}, d});
                            }
                        }
                    }
                }
            }
        }
        // dedup moves
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

    // ----- Evaluation/search helpers -----

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

    inline int reachable_in_one_to_SA(const std::vector<std::vector<std::map<std::string, std::string>>> &board,
                                      const std::string &me, int rows, int cols, const std::vector<int> &score_cols)
    {
        int sa_row = (me == "circle") ? top_score_row() : bottom_score_row(rows);
        std::vector<uint8_t> mark(rows * cols, 0);
        int cnt = 0;
        for (int y = 0; y < rows; ++y)
        {
            for (int x = 0; x < cols; ++x)
            {
                const auto &c = board[y][x];
                if (c.empty() || c.at("owner") != me)
                    continue;
                // 1-step into SA
                static const int D[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
                for (auto &d : D)
                {
                    int nx = x + d[0], ny = y + d[1];
                    if (!in_bounds(nx, ny, rows, cols))
                        continue;
                    if (ny == sa_row &&
                        std::find(score_cols.begin(), score_cols.end(), nx) != score_cols.end() &&
                        board[ny][nx].empty())
                    {
                        int id = ny * cols + nx;
                        if (!mark[id])
                        {
                            mark[id] = 1;
                            ++cnt;
                        }
                    }
                    // via river flow chain
                    if (!board[ny][nx].empty() && board[ny][nx].at("side") == "river")
                    {
                        auto dests = river_flow_destinations(board, nx, ny, x, y, me, rows, cols, score_cols, false);
                        for (auto &d2 : dests)
                        {
                            int tx = d2.first, ty = d2.second;
                            if (ty == sa_row &&
                                std::find(score_cols.begin(), score_cols.end(), tx) != score_cols.end())
                            {
                                int id = ty * cols + tx;
                                if (!mark[id])
                                {
                                    mark[id] = 1;
                                    ++cnt;
                                }
                            }
                        }
                    }
                }
            }
        }
        return cnt;
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
            const bool pusher_is_river = (!from.empty() && from.at("side") == "river");
            if (pusher_is_river)
            {
                // River push: river stays put; the adjacent stone at 'to' is moved to 'pushed_to'
                auto pushed = b[m.to[1]][m.to[0]];
                b[m.to[1]][m.to[0]].clear();
                b[m.pushed_to[1]][m.pushed_to[0]] = pushed;
                // Optional: if engine flips river to stone after push, simulate if needed:
                // from["side"] = "stone"; from.erase("orientation");
            }
            else
            {
                // Stone push: stone moves into 'to'; occupant moves to 'pushed_to'
                auto piece_from = from;
                from.clear();
                auto pushed = b[m.to[1]][m.to[0]];
                b[m.to[1]][m.to[0]] = piece_from;
                b[m.pushed_to[1]][m.pushed_to[0]] = pushed;
            }
        }
        else if (m.action == "flip")
        {
            auto &piece = from;
            if (piece.at("side") == "stone")
            {
                piece["side"] = "river";
                piece["orientation"] = m.orientation.empty() ? "horizontal" : m.orientation;
            }
            else
            {
                piece["side"] = "stone";
                piece.erase("orientation");
            }
        }
        else if (m.action == "rotate")
        {
            auto &piece = from;
            piece["orientation"] = (piece.at("orientation") == "horizontal") ? "vertical" : "horizontal";
        }
    }

    // Simple Zobrist + TT for pruning on Python-shaped board
    struct TTEntry
    {
        int depth;
        int score;
        int flag; // 0=exact, 1=lower, 2=upper
        Move best;
    };

    struct Zobrist
    {
        std::vector<uint64_t> table; // [y*cols+x][owner*side*orient bucket]
        int rows{0}, cols{0};
        std::mt19937_64 rng{1234567};
        void init(int r, int c)
        {
            rows = r;
            cols = c;
            table.resize((size_t)r * c * 12);
            for (auto &v : table)
                v = rng();
        }
        inline uint64_t key(int x, int y, int owner, int side, int orient) const
        {
            int bucket = owner * 6 + side * 2 + orient; // owner:0..2, side:0..2, orient:0..1
            size_t idx = (size_t)((y * cols + x) * 12 + bucket);
            return table[idx];
        }
    };

    inline uint64_t hash_board(const std::vector<std::vector<std::map<std::string, std::string>>> &board,
                               const Zobrist &zb)
    {
        uint64_t h = 0;
        int rows = (int)board.size(), cols = rows ? (int)board[0].size() : 0;
        for (int y = 0; y < rows; ++y)
        {
            for (int x = 0; x < cols; ++x)
            {
                const auto &c = board[y][x];
                if (c.empty())
                    continue;
                int owner = (c.at("owner")[0] == 'c') ? 1 : 2;
                int side = (c.at("side")[0] == 's') ? 1 : 2;
                int orient = (side == 2) ? ((c.at("orientation")[0] == 'h') ? 0 : 1) : 0;
                h ^= zb.key(x, y, owner, side, orient);
            }
        }
        return h;
    }

    inline int evaluate(const std::vector<std::vector<std::map<std::string, std::string>>> &board,
                        const std::string &me, int rows, int cols, const std::vector<int> &score_cols, int win_count)
    {
        std::string opp = (me == "circle") ? "square" : "circle";
        int mySA = stones_in_SA(board, me, rows, cols, score_cols);
        int opSA = stones_in_SA(board, opp, rows, cols, score_cols);
        if (mySA >= win_count)
            return 100000;
        if (opSA >= win_count)
            return -100000;

        int myR1 = reachable_in_one_to_SA(board, me, rows, cols, score_cols);
        int opR1 = reachable_in_one_to_SA(board, opp, rows, cols, score_cols);

        int score = 0;
        score += 5000 * (mySA - opSA);
        score += 300 * (myR1 - opR1);
        return score;
    }

} // namespace

class StudentAgent
{
public:
    explicit StudentAgent(std::string side) : side(std::move(side)), gen(rd()) {}

    std::vector<Move> generate_all_moves(const std::vector<std::vector<std::map<std::string, std::string>>> &board,
                                         int rows, int cols, const std::vector<int> &score_cols) const
    {
        std::vector<Move> out;
        out.reserve(512);
        for (int y = 0; y < rows; ++y)
        {
            for (int x = 0; x < cols; ++x)
            {
                const auto &cell = board[y][x];
                if (cell.empty() || cell.at("owner") != side)
                    continue;

                Targets tg = compute_valid_targets(board, x, y, side, rows, cols, score_cols);

                for (auto &mv : tg.moves)
                {
                    out.push_back({"move", {x, y}, {mv.first, mv.second}, {}, ""});
                }
                for (auto &pp : tg.pushes)
                {
                    out.push_back({"push", {x, y}, {pp.first.first, pp.first.second}, {pp.second.first, pp.second.second}, ""});
                }

                // flips / rotates
                if (cell.at("side") == "stone")
                {
                    out.push_back({"flip", {x, y}, {x, y}, {}, "horizontal"});
                    out.push_back({"flip", {x, y}, {x, y}, {}, "vertical"});
                }
                else
                {
                    out.push_back({"flip", {x, y}, {x, y}, {}, ""}); // river->stone
                    out.push_back({"rotate", {x, y}, {x, y}, {}, ""});
                }
            }
        }
        return out;
    }

    // Alpha-beta with TT (iterative deepening)
    Move choose(const std::vector<std::vector<std::map<std::string, std::string>>> &boardIn,
                int rows, int cols, const std::vector<int> &score_cols,
                float my_time, float /*opp_time*/)
    {
        int win_count = (int)score_cols.size();
        auto all = generate_all_moves(boardIn, rows, cols, score_cols);
        if (all.empty())
        {
            std::cout << "[StudentAgent " << side << "] no legal moves\n";
            return {"move", {0, 0}, {0, 0}, {}, ""};
        }

        // 1) Immediate win
        for (const auto &m : all)
        {
            auto tmp = boardIn;
            apply_move(tmp, m);
            if (stones_in_SA(tmp, side, rows, cols, score_cols) >= win_count)
            {
                log_choice(m);
                return m;
            }
        }

        // 2) Block opponent immediate win
        {
            std::string opp = (side == "circle") ? "square" : "circle";
            for (const auto &m : all)
            {
                auto tmp = boardIn;
                apply_move(tmp, m);

                StudentAgent oppAgent(opp);
                auto oppMoves = oppAgent.generate_all_moves(tmp, rows, cols, score_cols);
                bool oppWins = false;
                for (const auto &om : oppMoves)
                {
                    auto tmp2 = tmp;
                    apply_move(tmp2, om);
                    if (stones_in_SA(tmp2, opp, rows, cols, score_cols) >= win_count)
                    {
                        oppWins = true;
                        break;
                    }
                }
                if (!oppWins)
                {
                    log_choice(m, "block");
                    return m;
                }
            }
        }

        // 3) Iterative deepening alpha-beta with a tight time budget
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(std::max(60, (int)(my_time * 1000 * 0.05)));
        Zobrist zb;
        zb.init(rows, cols);
        std::unordered_map<uint64_t, TTEntry> TT;
        TT.reserve(1 << 16);

        Move best{}, cur{};
        int bestScore = -100000000, depth = 1;

        while (std::chrono::steady_clock::now() < deadline)
        {
            auto board = boardIn;
            int val = alpha_beta(board, side, rows, cols, score_cols, win_count, depth, -100000000, 100000000, true, deadline, zb, TT, cur);
            if (std::chrono::steady_clock::now() >= deadline)
                break;
            best = cur;
            bestScore = val;
            depth += 1;
        }

        if (best.action.empty())
        {
            // Fallback: prefer pushes, then moves
            std::stable_sort(all.begin(), all.end(), [](const Move &a, const Move &b)
                             {
                                 auto key = [&](const Move &m)
                                 { return m.action == "push" ? 2 : (m.action == "move" ? 1 : 0); };
                                 return key(a) > key(b); });
            best = all.front();
        }

        log_choice(best, "search");
        return best;
    }

private:
    std::string side;
    std::random_device rd;
    mutable std::mt19937 gen;

    // Logging
    void log_choice(const Move &mv, const char *tag = nullptr) const
    {
        std::cout << "[StudentAgent " << side << "] ";
        if (tag)
            std::cout << tag << " ";
        std::cout << "action=" << mv.action
                  << " from=(" << mv.from[0] << "," << mv.from[1] << ")"
                  << " to=(" << mv.to[0] << "," << mv.to[1] << ")";
        if (mv.action == "push" || !mv.pushed_to.empty())
        {
            std::cout << " pushed_to=(" << mv.pushed_to[0] << "," << mv.pushed_to[1] << ")";
        }
        if (mv.action == "flip" && !mv.orientation.empty())
        {
            std::cout << " orientation=" << mv.orientation;
        }
        std::cout << std::endl;
    }

    // Alpha-beta on Python-shaped board (free function made friend-like)
    static int alpha_beta(std::vector<std::vector<std::map<std::string, std::string>>> &board,
                          const std::string &rootSide, int rows, int cols, const std::vector<int> &score_cols,
                          int win_count, int depth, int alpha, int beta, bool maximizing,
                          std::chrono::steady_clock::time_point deadline,
                          Zobrist &zb, std::unordered_map<uint64_t, TTEntry> &TT,
                          Move &outBest)
    {
        if (std::chrono::steady_clock::now() >= deadline)
            return evaluate(board, rootSide, rows, cols, score_cols, win_count);

        std::string sideToMove = maximizing ? rootSide : ((rootSide == "circle") ? "square" : "circle");
        if (depth == 0)
            return evaluate(board, rootSide, rows, cols, score_cols, win_count);

        uint64_t h = hash_board(board, zb);
        auto ttIt = TT.find(h);
        if (ttIt != TT.end() && ttIt->second.depth >= depth)
        {
            if (ttIt->second.flag == 0)
            {
                outBest = ttIt->second.best;
                return ttIt->second.score;
            }
            if (ttIt->second.flag == 1)
                alpha = std::max(alpha, ttIt->second.score);
            else
                beta = std::min(beta, ttIt->second.score);
            if (alpha >= beta)
            {
                outBest = ttIt->second.best;
                return ttIt->second.score;
            }
        }

        StudentAgent agent(sideToMove);
        auto moves = agent.generate_all_moves(board, rows, cols, score_cols);
        if (moves.empty())
            return evaluate(board, rootSide, rows, cols, score_cols, win_count);

        // Order: pushes > moves > others
        std::stable_sort(moves.begin(), moves.end(), [](const Move &a, const Move &b)
                         {
                             auto key = [&](const Move &m)
                             { return m.action == "push" ? 2 : (m.action == "move" ? 1 : 0); };
                             return key(a) > key(b); });

        Move bestLocal{};
        int bestScore = maximizing ? -100000000 : 100000000;

        for (const auto &m : moves)
        {
            auto saved = board;
            apply_move(saved, m);

            // Immediate win check for sideToMove
            if (stones_in_SA(saved, sideToMove, rows, cols, score_cols) >= win_count)
            {
                outBest = m;
                int val = (sideToMove == rootSide) ? 100000 : -100000;
                TT[h] = {depth, val, 0, m};
                return val;
            }

            Move childBest{};
            int val = alpha_beta(saved, rootSide, rows, cols, score_cols, win_count, depth - 1, alpha, beta, !maximizing, deadline, zb, TT, childBest);

            if (maximizing)
            {
                if (val > bestScore)
                {
                    bestScore = val;
                    bestLocal = m;
                }
                alpha = std::max(alpha, bestScore);
            }
            else
            {
                if (val < bestScore)
                {
                    bestScore = val;
                    bestLocal = m;
                }
                beta = std::min(beta, bestScore);
            }
            if (alpha >= beta)
                break;
        }

        int flag = 0;
        if (bestScore <= alpha)
            flag = 2;
        else if (bestScore >= beta)
            flag = 1;
        TT[h] = {depth, bestScore, flag, bestLocal};
        outBest = bestLocal;
        return bestScore;
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
        .def("generate_all_moves", &StudentAgent::generate_all_moves)
        .def("choose", &StudentAgent::choose);
}