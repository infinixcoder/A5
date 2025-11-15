"""
River and Stones Game - AI Agent Framework

This module provides the base agent implementations and factory function.
The student implementation is in student_agent.py for better organization.

Game Rules Summary:
- Two players: Circle (red) and Square (blue)
- Pieces can be stones or rivers (horizontal/vertical)
- Goal: Get 4 of your stones into opponent's scoring area
- Actions: move, push, flip (stone↔river), rotate (river orientation)
- Rivers allow flow-based movement to distant locations
"""
import queue
import random
import copy
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod


# ==================== GAME UTILITIES ====================
# These functions help agents understand and manipulate the game state

def in_bounds(x: int, y: int, rows: int, cols: int) -> bool:
    """Check if coordinates are within board boundaries."""
    return 0 <= x < cols and 0 <= y < rows


def score_cols_for(cols: int) -> List[int]:
    """Get the column indices for scoring areas.

    Scale scoring area width with board size:
    - For 12 cols: 4 scoring cols (1/3)
    - For 14 cols: 5 scoring cols
    - For 16 cols: 6 scoring cols
    """
    if cols <= 12:
        w = 4
    elif cols <= 14:
        w = 5
    else:
        w = 6
    start = max(0, (cols - w) // 2)
    return list(range(start, start + w))


def get_win_count(cols: int) -> int:
    """Get the number of stones needed to win based on board size."""
    if cols <= 12:
        return 4
    elif cols <= 14:
        return 5
    else:
        return 6


def top_score_row() -> int:
    """Get the row index for Circle's scoring area."""
    return 2


def bottom_score_row(rows: int) -> int:
    """Get the row index for Square's scoring area."""
    return rows - 3


def is_opponent_score_cell(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    """Check if a cell is in the opponent's scoring area."""
    if player == "circle":
        return (y == bottom_score_row(rows)) and (x in score_cols)
    else:
        return (y == top_score_row()) and (x in score_cols)


def is_own_score_cell(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    """Check if a cell is in the player's own scoring area."""
    if player == "circle":
        return (y == top_score_row()) and (x in score_cols)
    else:
        return (y == bottom_score_row(rows)) and (x in score_cols)


def get_opponent(player: str) -> str:
    """Get the opponent player identifier."""
    return "square" if player == "circle" else "circle"


# ==================== RIVER FLOW SIMULATION ====================
# equivalent to get_river_flow_destinations from gameEngine
def agent_river_flow(board, rx, ry, sx, sy, player, rows, cols, score_cols, river_push=False):
    destinations = []
    visited = set()
    q = queue.Queue()
    q.put((rx, ry))

    while not q.empty():
        x, y = q.get()
        if (x, y) in visited or not in_bounds(x, y, rows, cols):
            continue
        visited.add((x, y))

        cell = board[sy][sx] if river_push and x == rx and y == ry else board[y][x]

        if not cell:
            if not is_opponent_score_cell(x, y, player, rows, cols, score_cols):
                destinations.append((x, y))
            continue

        if cell.side == "stone":
            continue

        dirs = [(1, 0), (-1, 0)] if cell.orientation == "horizontal" else [(0, 1), (0, -1)]

        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            while in_bounds(nx, ny, rows, cols):
                if is_opponent_score_cell(nx, ny, player, rows, cols, score_cols):
                    break

                next_cell = board[ny][nx]
                if not next_cell:
                    destinations.append((nx, ny))
                    nx += dx
                    ny += dy
                    continue

                if nx == sx and ny == sy:
                    nx += dx
                    ny += dy
                    continue

                if next_cell.side == "river":
                    q.put((nx, ny))
                break

    out = []
    seen = set()
    for d in destinations:
        if d not in seen:
            seen.add(d)
            out.append(d)

    return out


# ==================== MOVE VALIDATION AND GENERATION ====================
# equivalent to compute_valid_moves from gameEngine
def agent_compute_valid_moves(board, sx, sy, player, rows, cols, score_cols):
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    moves = set()
    pushes = []

    if not in_bounds(sx, sy, rows, cols):
        return {'moves': moves, 'pushes': pushes}

    p = board[sy][sx]
    if p is None or p.owner != player:
        return {'moves': moves, 'pushes': pushes}

    for dx, dy in dirs:
        tx, ty = sx + dx, sy + dy
        if not in_bounds(tx, ty, rows, cols):
            continue
        if is_opponent_score_cell(tx, ty, player, rows, cols, score_cols):
            continue

        target = board[ty][tx]

        if target is None:
            moves.add((tx, ty))
        elif target.side == "river":  # river
            flow = agent_river_flow(board, tx, ty, sx, sy, player, rows, cols, score_cols)
            for dest in flow:
                moves.add(dest)
        elif target.side == "stone":  # stone
            if p.side == "stone":
                px, py = tx + dx, ty + dy
                pushed_player = target.owner
                if (in_bounds(px, py, rows, cols) and
                        board[py][px] is None and
                        not is_opponent_score_cell(px, py, pushed_player, rows, cols, score_cols)):
                    pushes.append(((tx, ty), (px, py)))
            else:
                pushed_player = target.owner
                flow = agent_river_flow(board, tx, ty, sx, sy, pushed_player, rows, cols, score_cols, True)
                for dest in flow:
                    if not is_opponent_score_cell(dest[0],dest[1], pushed_player, rows, cols, score_cols):
                        pushes.append(((tx, ty), dest))

    return {'moves': moves, 'pushes': pushes}


# ==================== MOVE APPLICATION (FOR SIMULATION) ====================
# equivalent to validate_and_apply_move from gameEngine
def agent_apply_move(board, move, player, rows, cols, score_cols):
    if not isinstance(move, dict):
        return False, "move must be dict"
    action = move.get("action")
    if action == "move":
        fr = move.get("from")
        to = move.get("to")
        if not fr or not to:
            return False, "move needs from & to"
        fx, fy = int(fr[0]), int(fr[1])
        tx, ty = int(to[0]), int(to[1])
        if not in_bounds(fx, fy, rows, cols) or not in_bounds(tx, ty, rows, cols):
            return False, "oob"
        if is_opponent_score_cell(tx, ty, player, rows, cols, score_cols):
            return False, "can't go into opponent score"
        piece = board[fy][fx]
        if piece is None or piece.owner != player:
            return False, "invalid piece"
        if board[ty][tx] is None:
            valid_acts = agent_compute_valid_moves(board, fx, fy, player, rows, cols, score_cols)
            if (tx, ty) not in valid_acts['moves']:
                return False, "invalid move location"
            board[ty][tx] = piece
            board[fy][fx] = None
            return True, "moved"
        return False, "invalid move"
    elif action == "push":
        fr = move.get("from")
        to = move.get("to")
        pushed = move.get("pushed_to")
        if not fr or not to or not pushed:
            return False, "push needs from,to,pushed_to"
        fx, fy = int(fr[0]), int(fr[1])
        tx, ty = int(to[0]), int(to[1])
        px, py = int(pushed[0]), int(pushed[1])

        if not (in_bounds(fx, fy, rows, cols) and in_bounds(tx, ty, rows, cols) and in_bounds(px, py, rows, cols)):
            return False, "oob"
        pushed_player = board[ty][tx].owner if board[ty][tx] else None
        if is_opponent_score_cell(px, py, pushed_player, rows, cols, score_cols):
            return False, "push would enter opponent score cell"
        piece = board[fy][fx]
        if piece is None or piece.owner != player:
            return False, "invalid piece"
        if board[ty][tx] is None:
            return False, "to must be occupied"
        if board[py][px] is not None:
            return False, "pushed_to not empty"
        if piece.side == "river" and board[ty][tx].side == "river":
            return False, "rivers cannot push rivers"
        info = agent_compute_valid_moves(board, fx, fy, player, rows, cols, score_cols)
        valid_pairs = info['pushes']
        if ((tx, ty), (px, py)) not in valid_pairs:
            return False, "push pair invalid"
        board[py][px] = board[ty][tx]
        board[ty][tx] = board[fy][fx]
        board[fy][fx] = None
        mover = board[ty][tx]
        if mover.side == "river":
            mover.side = "stone"
            mover.orientation = None
        return True, "push applied"
    elif action == "flip":
        fr = move.get("from")
        if not fr:
            return False, "flip needs from"
        fx, fy = int(fr[0]), int(fr[1])
        if not in_bounds(fx, fy, rows, cols):
            return False, "oob"
        piece = board[fy][fx]
        if piece is None or piece.owner != player:
            return False, "invalid piece"
        if piece.side == "stone":
            ori = move.get("orientation")
            if ori not in ("horizontal", "vertical"):
                return False, "stone->river needs orientation"
            piece.side = "river"
            piece.orientation = ori
            return True, "flipped to river"
        else:
            piece.side = "stone"
            piece.orientation = None
            return True, "flipped to stone"
    elif action == "rotate":
        fr = move.get("from")
        if not fr:
            return False, "rotate needs from"
        fx, fy = int(fr[0]), int(fr[1])
        if not in_bounds(fx, fy, rows, cols):
            return False, "oob"
        piece = board[fy][fx]
        if piece is None or piece.owner != player:
            return False, "invalid"
        if piece.side == "stone":
            return False, "rotate only on river"
        piece.orientation = "vertical" if piece.orientation == "horizontal" else "horizontal"
        return True, "rotated"
    return False, "unknown action"


# ==================== BASE AGENT CLASS ====================

class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    Students should inherit from this class and implement the required methods.
    """

    def __init__(self, player: str):
        """
        Initialize agent.

        Args:
            player: Either "circle" or "square"
        """
        self.player = player
        self.opponent = get_opponent(player)

    @abstractmethod
    def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int], current_player_time: float,
               opponent_time: float) -> Optional[Dict[str, Any]]:
        """
        Choose the best move for the current board state.

        Args:
            board: 2D list representing the game board
            rows, cols: Board dimensions
            score_cols: List of column indices for scoring areas
            current_player_time : Remaining time for this player (in seconds)
            opponent_time : Remaining time for the opponent (in seconds)

        Returns:
            Dictionary representing the chosen move, or None if no moves available

        Move format examples:
            {"action": "move", "from": [x, y], "to": [x2, y2]}
            {"action": "push", "from": [x, y], "to": [x2, y2], "pushed_to": [x3, y3]}
            {"action": "flip", "from": [x, y], "orientation": "horizontal"}  # stone->river
            {"action": "flip", "from": [x, y]}  # river->stone
            {"action": "rotate", "from": [x, y]}  # rotate river
        """
        pass

    def generate_all_moves(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
        """
        Generate all legal moves for the current player.

        This is a helper method that students can use in their implementations.
        """
        moves = []
        for y in range(rows):
            for x in range(cols):
                p = board[y][x]
                if not p or p.owner != self.player:
                    continue

                valid_moves = agent_compute_valid_moves(board, x, y, self.player, rows, cols, score_cols)
                for move_coord in valid_moves['moves']:
                    moves.append({"action": "move", "from": [x, y], "to": list(move_coord)})

                for push_data in valid_moves['pushes']:
                    target_coord, dest_coord = push_data
                    moves.append({
                        "action": "push",
                        "from": [x, y],
                        "to": list(target_coord),
                        "pushed_to": list(dest_coord)
                    })

                if p.side == "stone":
                    moves.append({"action": "flip", "from": [x, y], "orientation": "horizontal"})
                    moves.append({"action": "flip", "from": [x, y], "orientation": "vertical"})
                else:
                    moves.append({"action": "flip", "from": [x, y], "orientation": None})
                    new_orient = "vertical" if p.orientation == "horizontal" else "horizontal"
                    moves.append({"action": "rotate", "from": [x, y], "orientation": new_orient})

        return moves

    def evaluate_board(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int]) -> float:
        """
        Evaluate the current board state from this player's perspective.
        Higher values indicate better positions for this player.

        This is a basic evaluation function that students can override.
        """
        score = 0.0
        top_row = top_score_row()
        bottom_row = bottom_score_row(rows)

        for y in range(rows):
            for x in range(cols):
                piece = board[y][x]
                if not piece:
                    continue

                if piece.owner == self.player and piece.side == "stone":
                    score += 1.0

                    # Bonus for stones in own scoring area
                    if is_own_score_cell(x, y, self.player, rows, cols, score_cols):
                        score += 10.0

                    # Small bonus for advancing toward opponent
                    if self.player == "circle":
                        score += (rows - y) * 0.1
                    else:
                        score += y * 0.1

                elif piece.owner == self.opponent and piece.side == "stone":
                    score -= 1.0

                    # Penalty if opponent has stones in their scoring area
                    if is_own_score_cell(x, y, self.opponent, rows, cols, score_cols):
                        score -= 10.0

        return score

    def simulate_move(self, board: List[List[Any]], move: Dict[str, Any], rows: int, cols: int, score_cols: List[int]) -> Tuple[bool, Any]:
        """
        Simulate a move on a copy of the board.

        Returns:
            (success: bool, new_board or error_message)
        """
        board_copy = copy.deepcopy(board)
        success, message = agent_apply_move(board_copy, move, self.player, rows, cols, score_cols)

        if success:
            return True, board_copy
        else:
            return False, message


# ==================== EXAMPLE AGENT IMPLEMENTATIONS ====================

class RandomAgent(BaseAgent):
    """
    Simple random agent that chooses moves randomly.
    This serves as a baseline and example implementation.
    """

    def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int], current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        moves = self.generate_all_moves(board, rows, cols, score_cols)
        if not moves:
            return None
        return random.choice(moves)


# ==================== STUDENT AGENT IMPORT ====================

try:
    from student_agent import StudentAgent
except ImportError:
    print("Warning: student_agent.py not found. Creating placeholder StudentAgent.")


    class StudentAgent(BaseAgent):
        """Placeholder StudentAgent - implement in student_agent.py"""

        def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int],
                   current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
            moves = self.generate_all_moves(board, rows, cols, score_cols)
            return random.choice(moves) if moves else None


# ==================== AGENT FACTORY ====================

def get_agent(player: str, strategy: str) -> BaseAgent:
    """
    Factory function to create agents based on strategy name.

    Args:
        player: "circle" or "square"
        strategy: Strategy name ("random", "student")

    Returns:
        Agent instance
    """
    print(strategy)
    strategy = strategy.lower()

    if strategy == "random":
        return RandomAgent(player)
    elif strategy == "student":
        return StudentAgent(player)
    elif strategy == "student_cpp":
        try:
            import student_agent_cpp as student_agent
            StudentAgentCpp = student_agent.StudentAgent
        except ImportError:
            StudentAgentCpp = None
        if StudentAgentCpp:
            return StudentAgentCpp(player)
        else:
            print("C++ StudentAgent not available. Falling back to Python StudentAgent.")
            return StudentAgent(player)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Available: random, student")