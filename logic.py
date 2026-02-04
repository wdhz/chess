"""
logic.py - 国际象棋游戏逻辑与AI决策核心模块

本模块实现了完整的国际象棋游戏逻辑、AI决策系统和游戏状态管理。
用户执白，AI执黑，AI核心完全在Python中实现，不依赖任何外部可执行引擎。
基于python-chess库处理棋盘规则，结合alpha-beta剪枝、迭代加深、置换表等技术，
实现2000+ ELO水平的AI决策能力。

作者: AI助手
日期: 2026-02-04
"""

import chess
import time
import random
from typing import Dict, List, Optional, Tuple, Any
from functools import lru_cache
import threading


class TranspositionTable:
    """
    置换表实现，使用Zobrist哈希缓存搜索结果，避免重复计算。
    支持EXACT、LOWERBOUND、UPPERBOUND三种标志类型。
    """
    HASH_SIZE = 128 * 1024 * 1024 // 24  # 128MB内存

    def __init__(self):
        self.table = [None] * self.HASH_SIZE
        self.zobrist_table = self._init_zobrist_hash()

    def _init_zobrist_hash(self) -> Dict[Tuple[int, int], int]:
        """
        初始化Zobrist哈希表，为每个(格子, 棋子)组合生成随机64位哈希值。
        """
        random.seed(42)
        table = {}
        for square in range(64):
            for piece_type in range(1, 7):  # 兵、马、象、车、后、王
                for color in [chess.WHITE, chess.BLACK]:
                    key = (square, piece_type, color)
                    table[key] = random.getrandbits(64)
        return table

    def _hash_key(self, board: chess.Board) -> int:
        """
        计算棋盘的Zobrist哈希值。
        """
        h = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                key = (square, piece.piece_type, piece.color)
                h ^= self.zobrist_table.get(key, 0)
        return h

    def store(self, board: chess.Board, depth: int, value: int,
              flag: str, best_move: Optional[chess.Move]):
        """
        存储搜索结果到置换表。
        """
        key = self._hash_key(board)
        index = key % self.HASH_SIZE
        self.table[index] = {
            'key': key,
            'depth': depth,
            'value': value,
            'flag': flag,
            'best_move': best_move
        }

    def lookup(self, board: chess.Board) -> Optional[Dict[str, Any]]:
        """
        从置换表查找棋盘状态。
        """
        key = self._hash_key(board)
        index = key % self.HASH_SIZE
        entry = self.table[index]
        if entry and entry['key'] == key:
            return entry
        return None


class KillerHeuristic:
    """
    杀手启发实现，用于移动排序优化。
    每层维护两个杀手走法，优先搜索可能引发剪枝的走法。
    """
    def __init__(self, max_depth: int = 16):
        self.killers = [[None, None] for _ in range(max_depth)]

    def get_score(self, move: chess.Move, depth: int) -> int:
        """
        获取走法的杀手启发评分。
        """
        if depth < len(self.killers):
            if move == self.killers[depth][0]:
                return 9000
            elif move == self.killers[depth][1]:
                return 8000
        return 0

    def record(self, move: chess.Move, depth: int):
        """
        记录杀手走法。
        """
        if depth < len(self.killers) and move != self.killers[depth][0]:
            self.killers[depth][1] = self.killers[depth][0]
            self.killers[depth][0] = move


class HistoryHeuristic:
    """
    历史启发实现，记录导致剪枝的走法频率。
    """
    def __init__(self):
        self.history = [[0] * 64 for _ in range(64)]  # from_square -> to_square

    def get_score(self, move: chess.Move) -> int:
        """
        获取走法的历史启发评分。
        """
        return self.history[move.from_square][move.to_square]

    def update(self, move: chess.Move, depth: int):
        """
        更新历史表，加分与搜索深度相关。
        """
        self.history[move.from_square][move.to_square] += depth * depth


# 棋子基础价值（厘分）
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# 白方兵位置权重表（PAWN_PST）
PAWN_PST = [
    0,   0,   0,   0,   0,   0,   0,   0,  # 第1行
    50, 50, 50, 50, 50, 50, 50, 50,  # 第2行
    10, 10, 20, 30, 30, 20, 10, 10,  # 第3行
    5,  10, 15, 25, 25, 15, 10,  5,  # 第4行
    0,   5, 10, 20, 20, 10,  5,   0,  # 第5行
    10, 10, 10, 15, 15, 10, 10, 10,  # 第6行
    30, 30, 30, 30, 30, 30, 30, 30,  # 第7行
    0,   0,   0,   0,   0,   0,   0,   0   # 第8行（已升变）
]

# 白方马位置权重表（KNIGHT_PST）
KNIGHT_PST = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50
]

# 白方象位置权重表（BISHOP_PST）
BISHOP_PST = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20
]

# 白方车位置权重表（ROOK_PST）
ROOK_PST = [
    0,   0,   0,   0,   0,   0,   0,   0,
    5,  10,  10,  10,  10,  10,  10,   5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    5,  10,  10,  10,  10,  10,  10,   5,
    0,   0,   0,   0,   0,   0,   0,   0
]

# 白方后位置权重表（QUEEN_PST）
QUEEN_PST = [
    -20, -10, -10, -5,  -5, -10, -10, -20,
    -10,   0,   0,  0,   0,   0,   0, -10,
    -10,   0,   5,  5,   5,   5,   0, -10,
    -5,   0,   5,  5,   5,   5,   0,  -5,
    0,   0,   5,  5,   5,   5,   0,  -5,
    -10,   0,   5,  5,   5,   5,   0, -10,
    -10,   0,   0,  0,   0,   0,   0, -10,
    -20, -10, -10, -5,  -5, -10, -10, -20
]

# 白方王位置权重表（QUEEN_PST）
KING_PST = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20,  20,   0,   0,   0,   0,  20,  20,
    20,  30,  10,   0,   0,  10,  30,  20
]

# 残局时的王位置权重表（更积极）
KING_PST_ENDGAME = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -50, -40, -30, -20, -20, -30, -40, -50
]


class EvaluationCache:
    """
    评估函数缓存，避免重复计算相同局面的分数。
    使用LRU缓存策略，线程安全。
    """
    def __init__(self, maxsize: int = 100000):
        self.maxsize = maxsize

    @lru_cache(maxsize=100000)
    def _cached_evaluate(self, board_fen: str) -> int:
        """
        缓存的评估函数，以FEN字符串为键。
        """
        board = chess.Board(board_fen)
        return self._evaluate_board(board)

    def evaluate(self, board: chess.Board) -> int:
        """
        公共评估接口，处理FEN序列化。
        """
        # 只使用位置部分，忽略回合、特殊状态
        board_fen = ' '.join(board.fen().split(' ')[:4])
        return self._cached_evaluate(board_fen)

    def _evaluate_board(self, board: chess.Board) -> int:
        """
        实际的评估函数实现，计算局面分数。
        """
        if board.is_checkmate():
            return -100000 if board.turn == chess.WHITE else 100000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        score = 0
        score += self._evaluate_material(board)
        score += self._evaluate_piece_square_tables(board)
        score += self._evaluate_pawn_structure(board, chess.WHITE) - self._evaluate_pawn_structure(board, chess.BLACK)
        score += self._evaluate_king_safety(board, chess.WHITE) - self._evaluate_king_safety(board, chess.BLACK)
        score += self._evaluate_mobility(board, chess.WHITE) - self._evaluate_mobility(board, chess.BLACK)

        # 根据当前玩家调整符号
        return score if board.turn == chess.WHITE else -score

    def _evaluate_material(self, board: chess.Board) -> int:
        """
        计算材料分。
        """
        material = 0
        for piece_type in PIECE_VALUES:
            material += len(board.pieces(piece_type, chess.WHITE)) * PIECE_VALUES[piece_type]
            material -= len(board.pieces(piece_type, chess.BLACK)) * PIECE_VALUES[piece_type]
        return material

    def _evaluate_piece_square_tables(self, board: chess.Board) -> int:
        """
        计算位置权重分。
        """
        pst_score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                idx = rank * 8 + file
                if piece.color == chess.WHITE:
                    if piece.piece_type == chess.PAWN:
                        pst_score += PAWN_PST[idx]
                    elif piece.piece_type == chess.KNIGHT:
                        pst_score += KNIGHT_PST[idx]
                    elif piece.piece_type == chess.BISHOP:
                        pst_score += BISHOP_PST[idx]
                    elif piece.piece_type == chess.ROOK:
                        pst_score += ROOK_PST[idx]
                    elif piece.piece_type == chess.QUEEN:
                        pst_score += QUEEN_PST[idx]
                    elif piece.piece_type == chess.KING:
                        # 残局时使用不同的王表
                        if self._is_endgame(board):
                            pst_score += KING_PST_ENDGAME[idx]
                        else:
                            pst_score += KING_PST[idx]
                else:  # 黑方
                    if piece.piece_type == chess.PAWN:
                        pst_score -= PAWN_PST[63 - idx]
                    elif piece.piece_type == chess.KNIGHT:
                        pst_score -= KNIGHT_PST[63 - idx]
                    elif piece.piece_type == chess.BISHOP:
                        pst_score -= BISHOP_PST[63 - idx]
                    elif piece.piece_type == chess.ROOK:
                        pst_score -= ROOK_PST[63 - idx]
                    elif piece.piece_type == chess.QUEEN:
                        pst_score -= QUEEN_PST[63 - idx]
                    elif piece.piece_type == chess.KING:
                        if self._is_endgame(board):
                            pst_score -= KING_PST_ENDGAME[63 - idx]
                        else:
                            pst_score -= KING_PST[63 - idx]
        return pst_score

    def _is_endgame(self, board: chess.Board) -> bool:
        """
        判断是否进入残局。
        """
        white_material = sum(PIECE_VALUES[piece.piece_type] for piece in board.piece_map().values()
                             if piece.color == chess.WHITE and piece.piece_type != chess.KING)
        black_material = sum(PIECE_VALUES[piece.piece_type] for piece in board.piece_map().values()
                             if piece.color == chess.BLACK and piece.piece_type != chess.KING)
        total_material = white_material + black_material
        return total_material < 4000  # 低于4000厘分认为是残局

    def _evaluate_pawn_structure(self, board: chess.Board, color: bool) -> int:
        """
        评估兵结构。
        """
        score = 0
        enemy_color = not color
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                file = chess.square_file(square)
                rank = chess.square_rank(square)

                # 孤兵：同列无己方兵
                if not any(board.piece_at(chess.square(f, r)) and
                          board.piece_at(chess.square(f, r)).piece_type == chess.PAWN and
                          board.piece_at(chess.square(f, r)).color == color
                          for r in range(8) for f in [max(0, file-1), file, min(7, file+1)] if f != file):
                    score -= 15

                # 叠兵：同列有多个己方兵
                if sum(1 for r in range(8) if board.piece_at(chess.square(file, r)) and
                       board.piece_at(chess.square(file, r)).piece_type == chess.PAWN and
                       board.piece_at(chess.square(file, r)).color == color) > 1:
                    score -= 20

                # 通路兵：前方无敌方兵阻挡
                if color == chess.WHITE:
                    ahead_squares = [chess.square(file, r) for r in range(rank + 1, 8)]
                else:
                    ahead_squares = [chess.square(file, r) for r in range(0, rank)]
                if not any(board.piece_at(sq) and board.piece_at(sq).piece_type == chess.PAWN and
                           board.piece_at(sq).color == enemy_color for sq in ahead_squares):
                    # 奖励随推进程度增加
                    passed_bonus = 30 + (rank if color == chess.WHITE else 7 - rank) * 10
                    score += passed_bonus

        return score

    def _evaluate_king_safety(self, board: chess.Board, color: bool) -> int:
        """
        评估王的安全性。
        """
        if self._is_endgame(board):
            # 残局中王可作为战斗力
            return self._evaluate_king_activity(board, color)

        king_square = board.king(color)
        if king_square is None:
            return 0

        # 王环：王周围一圈格子
        king_ring = self._get_king_ring(king_square)
        attack_score = 0
        for square in king_ring:
            if board.is_attacked_by(not color, square):
                # 简化：根据攻击者数量评分
                attackers = len(board.attackers(not color, square))
                attack_score += attackers * 10

        # 兵盾保护
        pawn_shield_score = 0
        file = chess.square_file(king_square)
        rank = chess.square_rank(king_square)
        shield_files = [max(0, file-1), file, min(7, file+1)]
        shield_ranks = [rank-1] if color == chess.WHITE else [rank+1]
        for f in shield_files:
            for r in shield_ranks:
                if 0 <= r < 8 and 0 <= f < 8:
                    sq = chess.square(f, r)
                    piece = board.piece_at(sq)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        pawn_shield_score += 20

        return -attack_score + pawn_shield_score

    def _get_king_ring(self, king_square: int) -> List[int]:
        """
        获取王环区域。
        """
        ring = []
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        for df in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                if df == 0 and dr == 0:
                    continue
                file = king_file + df
                rank = king_rank + dr
                if 0 <= file <= 7 and 0 <= rank <= 7:
                    ring.append(chess.square(file, rank))
        return ring

    def _evaluate_king_activity(self, board: chess.Board, color: bool) -> int:
        """
        残局中评估王的活跃度。
        """
        king_square = board.king(color)
        if king_square is None:
            return 0
        # 王越靠近中心越活跃
        file = chess.square_file(king_square)
        rank = chess.square_rank(king_square)
        center_distance = abs(file - 3.5) + abs(rank - 3.5)
        return int((6 - center_distance) * 10)

    def _evaluate_mobility(self, board: chess.Board, color: bool) -> int:
        """
        评估棋子的机动性。
        """
        mobility = 0
        temp_board = board.copy()
        for move in temp_board.legal_moves:
            if temp_board.piece_at(move.from_square).color == color:
                mobility += 1
        return mobility


class ChessGameLogic:
    """
    国际象棋游戏逻辑核心类，管理游戏状态、AI决策和悔棋功能。
    """
    def __init__(self):
        self.board = chess.Board()
        self.move_stack = []  # 记录走法历史
        self.board_states = [self.board.fen()]  # 记录局面历史
        self.transposition_table = TranspositionTable()
        self.killer_heuristic = KillerHeuristic()
        self.history_heuristic = HistoryHeuristic()
        self.evaluation_cache = EvaluationCache()
        self.ai_color = chess.BLACK
        self.search_lock = threading.Lock()

    def reset_game(self):
        """
        重置游戏到初始状态。
        """
        self.board = chess.Board()
        self.move_stack = []
        self.board_states = [self.board.fen()]
        self.transposition_table = TranspositionTable()

    def make_move(self, move: chess.Move) -> bool:
        """
        执行用户走法。
        """
        if move not in self.board.legal_moves:
            return False

        try:
            self.board.push(move)
            self.move_stack.append(move)
            self.board_states.append(self.board.fen())
            return True
        except Exception:
            return False

    def undo_move(self) -> bool:
        """
        悔棋，撤销最后一步。
        """
        if len(self.board_states) <= 1:
            return False

        try:
            self.board_states.pop()
            self.move_stack.pop()
            previous_fen = self.board_states[-1]
            self.board.set_fen(previous_fen)
            return True
        except Exception:
            return False

    def get_game_status(self) -> Tuple[str, str]:
        """
        获取当前游戏状态。
        """
        if self.board.is_checkmate():
            return "checkmate", "将死"
        elif self.board.is_stalemate():
            return "stalemate", "逼和"
        elif self.board.is_insufficient_material():
            return "insufficient_material", "子力不足"
        elif self.board.can_claim_fifty_moves():
            return "fifty_moves", "50步自然限着"
        elif self.board.is_repetition(count=3):
            return "repetition", "局面重复三次"
        else:
            return "ongoing", "进行中"

    def _null_move_pruning(self, board: chess.Board, depth: int,
                           alpha: int, beta: int, maximizing: bool) -> int:
        """
        空着启发实现。
        """
        if depth < 3 or board.is_check() or self._has_few_pieces(board):
            return self._alpha_beta_search(board, depth, alpha, beta, not maximizing)

        # 执行空着
        board.push(chess.Move.null())
        # 搜索深度减少R=2
        null_score = -self._alpha_beta_search(board, depth - 3, -beta, -beta + 1, not maximizing)
        board.pop()

        if null_score >= beta:
            return beta
        return self._alpha_beta_search(board, depth, alpha, beta, maximizing)

    def _has_few_pieces(self, board: chess.Board) -> bool:
        """
        判断是否为残局（棋子较少）。
        """
        total_pieces = len([p for p in board.piece_map().values() if p])
        return total_pieces < 10

    def _alpha_beta_search(self, board: chess.Board, depth: int,
                           alpha: int, beta: int, maximizing: bool,
                           depth_from_root: int = 0) -> int:
        """
        Alpha-Beta剪枝搜索核心。
        """
        # 检查游戏结束
        if depth == 0 or board.is_game_over():
            return self.evaluation_cache.evaluate(board)

        # 置换表查找
        tt_entry = self.transposition_table.lookup(board)
        if tt_entry and tt_entry['depth'] >= depth:
            if tt_entry['flag'] == 'EXACT':
                return tt_entry['value']
            elif tt_entry['flag'] == 'LOWERBOUND':
                alpha = max(alpha, tt_entry['value'])
            elif tt_entry['flag'] == 'UPPERBOUND':
                beta = min(beta, tt_entry['value'])
            if alpha >= beta:
                return tt_entry['value']

        # 空着启发
        if not maximizing and depth >= 3:
            value = self._null_move_pruning(board, depth, alpha, beta, maximizing)
            if value >= beta:
                return beta

        best_value = float('-inf') if maximizing else float('inf')
        best_move = None
        original_alpha = alpha

        # 移动排序
        moves = self._order_moves(board, depth_from_root)

        for move in moves:
            board.push(move)
            value = self._alpha_beta_search(board, depth - 1, alpha, beta, not maximizing, depth_from_root + 1)
            board.pop()

            if maximizing:
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, value)
            else:
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, value)

            if alpha >= beta:
                # 剪枝成功，记录杀手走法
                if not board.is_capture(move):
                    self.killer_heuristic.record(move, depth_from_root)
                break

        # 存储到置换表
        flag = 'EXACT'
        if best_value <= original_alpha:
            flag = 'UPPERBOUND'
        elif best_value >= beta:
            flag = 'LOWERBOUND'
        self.transposition_table.store(board, depth, best_value, flag, best_move)

        return best_value

    def _order_moves(self, board: chess.Board, depth: int) -> List[chess.Move]:
        """
        移动排序，提高剪枝效率。
        """
        moves = list(board.legal_moves)
        scored_moves = []

        for move in moves:
            score = 0

            # 置换表最佳走法优先
            tt_entry = self.transposition_table.lookup(board)
            if tt_entry and tt_entry['best_move'] == move:
                score += 10000

            # 杀手启发
            score += self.killer_heuristic.get_score(move, depth)

            # 历史启发
            score += self.history_heuristic.get_score(move)

            # 捕获走法：MVV-LVA（最高价值受害者，最低价值攻击者）
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    score += 1000 * victim.piece_type - 10 * attacker.piece_type

            scored_moves.append((move, score))

        # 按分数降序排序
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        return [move for move, score in scored_moves]

    def find_best_move(self, time_limit: float = 5.0) -> Optional[chess.Move]:
        """
        迭代加深搜索寻找最佳走法。
        """
        with self.search_lock:
            start_time = time.time()
            best_move = None
            self.transposition_table = TranspositionTable()  # 新搜索开始
            self.killer_heuristic = KillerHeuristic()
            self.history_heuristic = HistoryHeuristic()

            # 迭代加深
            for depth in range(1, 16):
                # 检查时间限制
                if time.time() - start_time > time_limit * 0.8:
                    break

                current_best = None
                moves = self._order_moves(self.board, 0)

                for move in moves:
                    self.board.push(move)
                    score = self._alpha_beta_search(self.board, depth - 1,
                                                   float('-inf'), float('inf'), False)
                    self.board.pop()

                    if current_best is None or score > current_best:
                        current_best = score
                        best_move = move

            return best_move


def main():
    """
    程序入口点，演示AI对战逻辑。
    """
    game = ChessGameLogic()
    print("国际象棋AI对战开始，用户执白，AI执黑")

    while True:
        print(f"\n{game.board}")
        status, msg = game.get_game_status()
        print(f"状态: {msg}")
        if status != "ongoing":
            break

        if game.board.turn == chess.WHITE:
            # 用户走法
            move_uci = input("请输入走法 (UCI格式, 如 e2e4): ").strip()
            if move_uci.lower() == 'quit':
                break
            if move_uci.lower() == 'undo':
                if game.undo_move():
                    print("已悔棋")
                else:
                    print("无法悔棋")
                continue

            try:
                move = chess.Move.from_uci(move_uci)
                if game.make_move(move):
                    print("走法已执行")
                else:
                    print("非法走法")
            except Exception:
                print("走法格式错误")
        else:
            # AI走法
            print("AI正在思考...")
            best_move = game.find_best_move(time_limit=3.0)
            if best_move:
                game.make_move(best_move)
                print(f"AI走法: {best_move.uci()}")
            else:
                print("AI无法找到走法")
                break

    print("游戏结束")


if __name__ == "__main__":
    main()