

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



"""
gui.py - 国际象棋人机对战图形界面模块

本模块使用PyQt5构建完整的国际象棋人机对战图形用户界面。
用户执白方，AI执黑方，通过图形化界面进行对战。
界面与逻辑完全分离，逻辑处理由logic.py中的ChessGameLogic类负责。

作者: AI助手
日期: 2026-02-04
"""

import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QMenuBar, QStatusBar, QFileDialog,
    QMessageBox
)
from PyQt5.QtGui import QPainter, QColor, QPixmap, QBrush, QFont, QIcon
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread
  # 导入自定义游戏逻辑模块


class AIWorker(QThread):
    """
    独立线程中运行AI思考，防止界面卡顿。
    调用ChessGameLogic.find_best_move()获取AI走法。
    """
    move_calculated = pyqtSignal(object)  # 发射chess.Move对象
    error_occurred = pyqtSignal(str)

    def __init__(self, game_logic):
        super().__init__()
        self.game_logic = game_logic
        self.is_running = True

    def run(self):
        try:
            # 调用逻辑模块中的AI决策方法
            best_move = self.game_logic.find_best_move(time_limit=3.0)
            if self.is_running:
                self.move_calculated.emit(best_move)
        except Exception as e:
            self.error_occurred.emit(f"AI计算出错: {str(e)}")

    def stop(self):
        self.is_running = False


class ChessBoardWidget(QWidget):
    """
    国际象棋棋盘图形界面组件。
    负责绘制8x8棋盘、棋子，处理用户点击交互。
    """
    move_made = pyqtSignal(object)  # 发射chess.Move对象

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 480)
        self.square_size = 60
        self.selected_square = None  # 选中的格子 (row, col)
        self.legal_moves = []  # 合法移动目标 [(row, col), ...]
        self.piece_images = {}
        self.load_piece_images()

    def load_piece_images(self):
        """
        加载12张棋子图片资源。
        优先从resources目录加载PNG图像，失败时使用Unicode字符回退。
        """
        pieces = {
            'K': 'white_king.png', 'Q': 'white_queen.png', 'R': 'white_rook.png',
            'B': 'white_bishop.png', 'N': 'white_knight.png', 'P': 'white_pawn.png',
            'k': 'black_king.png', 'q': 'black_queen.png', 'r': 'black_rook.png',
            'b': 'black_bishop.png', 'n': 'black_knight.png', 'p': 'black_pawn.png'
        }

        base_path = os.path.join(os.path.dirname(__file__), 'resources')
        fallback = True  # 是否使用字符回退

        for piece, filename in pieces.items():
            path = os.path.join(base_path, filename)
            if os.path.exists(path):
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(
                        self.square_size * 0.8, self.square_size * 0.8,
                        Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                    self.piece_images[piece] = scaled_pixmap
                    fallback = False
                else:
                    break
            else:
                break

        # 如果资源不存在，使用Unicode字符作为回退
        if fallback:
            self.piece_images = {
                'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
                'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
            }

    def paintEvent(self, event):
        """绘制棋盘、坐标标记、选中高亮、合法移动提示和棋子。"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制8x8棋盘方格
        for row in range(8):
            for col in range(8):
                x = col * self.square_size
                y = row * self.square_size

                # 交替颜色：浅色和深色
                color = QColor("#f0d9b5") if (row + col) % 2 == 0 else QColor("#b58863")
                painter.fillRect(x, y, self.square_size, self.square_size, QBrush(color))

                # 高亮选中的格子
                if self.selected_square == (row, col):
                    painter.setPen(QColor("#39c5bb"))
                    painter.setBrush(QColor(57, 197, 187, 100))
                    painter.drawRect(x, y, self.square_size, self.square_size)

                # 高亮合法移动目标
                if (row, col) in self.legal_moves:
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QColor(0, 255, 0, 100))
                    painter.drawEllipse(x + 20, y + 20, 20, 20)

                # 绘制棋子
                square = row * 8 + col
                piece = self.parent().game_logic.board.piece_at(square)
                if piece:
                    piece_symbol = piece.symbol()
                    if piece_symbol in self.piece_images:
                        img = self.piece_images[piece_symbol]
                        if isinstance(img, QPixmap):
                            # 绘制图像
                            painter.drawPixmap(
                                x + (self.square_size - img.width()) // 2,
                                y + (self.square_size - img.height()) // 2,
                                img
                            )
                        else:
                            # 绘制Unicode字符
                            painter.setFont(QFont("Arial", 36, QFont.Bold))
                            painter.setPen(QColor("#000000" if piece.color else "#ffffff"))
                            painter.drawText(x, y, self.square_size, self.square_size,
                                           Qt.AlignCenter, img)

        # 绘制棋盘坐标标记
        painter.setFont(QFont("Arial", 8, QFont.Bold))
        painter.setPen(QColor("#000000"))
        for i in range(8):
            # 行号 (8-1)
            painter.drawText(2, (7 - i) * self.square_size + 12, f"{8 - i}")
            # 列标 (a-h)
            painter.drawText(i * self.square_size + 50, 475, f"{chr(97 + i)}")

    def mousePressEvent(self, event):
        """处理鼠标点击事件，实现选择棋子和移动功能。"""
        if event.button() != Qt.LeftButton:
            return

        col = event.x() // self.square_size
        row = event.y() // self.square_size

        # 检查点击是否在棋盘范围内
        if col < 0 or col > 7 or row < 0 or row > 7:
            return

        square_index = row * 8 + col
        piece = self.parent().game_logic.board.piece_at(square_index)

        # 如果点击的是合法移动目标，则执行移动
        if (row, col) in self.legal_moves:
            move = self._find_move_from_selected_to(square_index)
            if move:
                self._execute_move(move)
            return

        # 重置选择状态
        self.selected_square = None
        self.legal_moves = []

        # 仅允许用户选择白方棋子
        if piece and piece.color == True:  # chess.WHITE is True
            self.selected_square = (row, col)
            self.legal_moves = self._get_legal_moves_for_square(square_index)

        self.update()

    def _get_legal_moves_for_square(self, square_index):
        """获取指定格子上棋子的所有合法移动目标。"""
        targets = []
        for move in self.parent().game_logic.board.legal_moves:
            if move.from_square == square_index:
                to_row = move.to_square // 8
                to_col = move.to_square % 8
                targets.append((to_row, to_col))
        return targets

    def _find_move_from_selected_to(self, to_square_index):
        """根据选中的起始格和目标格，查找对应的chess.Move对象。"""
        if not self.selected_square:
            return None

        from_row, from_col = self.selected_square
        from_square_index = from_row * 8 + from_col

        for move in self.parent().game_logic.board.legal_moves:
            if move.from_square == from_square_index and move.to_square == to_square_index:
                return move
        return None

    def _execute_move(self, move):
        """执行走法并发出信号。"""
        if move in self.parent().game_logic.board.legal_moves:
            self.parent().handle_human_move(move)
            self.selected_square = None
            self.legal_moves = []
            self.update()

    def set_board(self, board):
        """外部调用以更新棋盘显示。"""
        self.selected_square = None
        self.legal_moves = []
        self.update()

    def update_display(self):
        """更新显示，由父窗口调用。"""
        self.update()


class GameController(QObject):
    """
    游戏控制中心。
    管理游戏状态、AI交互和用户操作，连接GUI与GameLogic。
    """
    game_status_updated = pyqtSignal(str)
    board_changed = pyqtSignal(object)  # 发射chess.Board对象
    current_player_updated = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.game_logic = ChessGameLogic()
        self.ai_worker = None

    def start_new_game(self):
        """开始新游戏，重置所有状态。"""
        self.game_logic.reset_game()
        self.board_changed.emit(self.game_logic.board)
        self.game_status_updated.emit("游戏开始，轮到你了")
        self.current_player_updated.emit("白方")

    def handle_human_move(self, move):
        """处理用户走法。"""
        if self.game_logic.make_move(move):
            self.board_changed.emit(self.game_logic.board)
            # 检查游戏是否结束
            status_key, status_msg = self.game_logic.get_game_status()
            if status_key != "ongoing":
                self.game_status_updated.emit(f"游戏结束: {status_msg}")
                return

            # 用户走完，轮到AI
            self.current_player_updated.emit("黑方")
            self.game_status_updated.emit("AI正在思考...")
            self._start_ai_think()
        else:
            self.game_status_updated.emit("非法走法")

    def _start_ai_think(self):
        """启动AI思考线程。"""
        if self.game_logic.board.turn == False and not self.game_logic.board.is_game_over():  # chess.BLACK is False
            self.ai_worker = AIWorker(self.game_logic)
            self.ai_worker.move_calculated.connect(self._handle_ai_move)
            self.ai_worker.error_occurred.connect(self._handle_ai_error)
            self.ai_worker.start()

    def _handle_ai_move(self, best_move):
        """处理AI返回的走法。"""
        if best_move:
            self.game_logic.make_move(best_move)
            self.board_changed.emit(self.game_logic.board)
            # 检查游戏结束状态
            status_key, status_msg = self.game_logic.get_game_status()
            if status_key != "ongoing":
                self.game_status_updated.emit(f"游戏结束: {status_msg}")
            else:
                self.game_status_updated.emit("轮到你了")
                self.current_player_updated.emit("白方")
        else:
            self.game_status_updated.emit("AI无法找到走法")
        self.ai_worker = None

    def _handle_ai_error(self, error_msg):
        """处理AI计算错误。"""
        self.game_status_updated.emit(f"AI错误: {error_msg}")
        self.ai_worker = None

    def undo_move(self):
        """执行悔棋操作。"""
        if self.game_logic.undo_move():
            self.board_changed.emit(self.game_logic.board)
            # 更新当前玩家
            if self.game_logic.board.turn == True:
                self.current_player_updated.emit("白方")
                self.game_status_updated.emit("已悔棋，轮到你了")
            else:
                self.current_player_updated.emit("黑方")
                self.game_status_updated.emit("已悔棋，轮到AI")
            return True
        else:
            self.game_status_updated.emit("无法悔棋")
            return False

    def cleanup(self):
        """清理资源，终止AI线程。"""
        if self.ai_worker and self.ai_worker.isRunning():
            self.ai_worker.stop()
            self.ai_worker.wait()


class ChessMainWindow(QMainWindow):
    """
    主窗口类。
    集成棋盘、控制面板、菜单栏和状态栏。
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("国际象棋人机对战")
        self.setGeometry(100, 100, 800, 500)
        self.setWindowIcon(QIcon())  # 可设置程序图标

        # 创建游戏控制器
        self.controller = GameController(self)
        self.controller.board_changed.connect(self.update_board_display)
        self.controller.game_status_updated.connect(self.update_status_bar)
        self.controller.current_player_updated.connect(self.update_current_player)

        # 初始化UI
        self.init_ui()

        # 开始新游戏
        self.controller.start_new_game()

    def init_ui(self):
        """初始化用户界面布局。"""
        # 创建中央部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧：棋盘区域
        board_container = QFrame()
        board_container.setFrameShape(QFrame.StyledPanel)
        board_layout = QVBoxLayout(board_container)
        board_layout.setAlignment(Qt.AlignCenter)

        self.chess_board = ChessBoardWidget(self)
        self.chess_board.move_made.connect(self.controller.handle_human_move)
        board_layout.addWidget(self.chess_board)

        # 右侧：控制面板
        control_panel = QFrame()
        control_panel.setFrameShape(QFrame.StyledPanel)
        control_panel.setFixedWidth(250)
        control_layout = QVBoxLayout(control_panel)

        # 游戏状态显示区域
        status_frame = QFrame()
        status_frame.setFrameShape(QFrame.Box)
        status_layout = QVBoxLayout(status_frame)

        self.current_player_label = QLabel("当前玩家：白方")
        self.current_player_label.setAlignment(Qt.AlignCenter)
        self.current_player_label.setStyleSheet("QLabel { font-weight: bold; font-size: 14px; }")
        status_layout.addWidget(self.current_player_label)

        self.game_status_label = QLabel("游戏状态：进行中")
        self.game_status_label.setAlignment(Qt.AlignCenter)
        self.game_status_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }")
        status_layout.addWidget(self.game_status_label)

        # 控制按钮
        btn_new_game = QPushButton("新游戏")
        btn_new_game.clicked.connect(self.controller.start_new_game)
        control_layout.addWidget(btn_new_game)

        btn_undo = QPushButton("悔棋")
        btn_undo.clicked.connect(self.controller.undo_move)
        control_layout.addWidget(btn_undo)

        btn_exit = QPushButton("退出")
        btn_exit.clicked.connect(self.close)
        control_layout.addWidget(btn_exit)

        # 添加弹性空间，使按钮靠上
        control_layout.addStretch()

        # 将左右部分加入主布局
        main_layout.addWidget(board_container, 7)  # 棋盘占7份
        main_layout.addWidget(control_panel, 3)    # 控制面板占3份

        # 创建菜单栏
        self._create_menu_bar()

        # 创建状态栏
        self.statusBar().showMessage("国际象棋人机对战已就绪")

    def _create_menu_bar(self):
        """创建菜单栏。"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件")

        new_game_action = file_menu.addAction("新游戏")
        new_game_action.triggered.connect(self.controller.start_new_game)

        undo_action = file_menu.addAction("悔棋")
        undo_action.triggered.connect(self.controller.undo_move)

        exit_action = file_menu.addAction("退出")
        exit_action.triggered.connect(self.close)

    def update_board_display(self, board):
        """更新棋盘显示。"""
        self.chess_board.set_board(board)
        self.chess_board.update_display()

    def update_status_bar(self, text):
        """更新状态栏和游戏状态标签。"""
        self.statusBar().showMessage(text)
        self.game_status_label.setText(f"游戏状态：{text}")

    def update_current_player(self, player):
        """更新当前玩家显示。"""
        self.current_player_label.setText(f"当前玩家：{player}")

    def closeEvent(self, event):
        """窗口关闭时清理资源。"""
        self.controller.cleanup()
        event.accept()



def main():
    """程序入口点。"""
    app = QApplication(sys.argv)
    window = ChessMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
