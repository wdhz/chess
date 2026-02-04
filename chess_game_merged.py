import sys
import chess
import chess.engine
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGridLayout
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QFont, QPixmap, QIcon
import random

class ChessBoard(QWidget):
    move_made = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.board = chess.Board()
        self.selected_square = None
        self.square_size = 60
        self.setFixedSize(self.square_size * 8, self.square_size * 8)
        self.piece_images = self.load_piece_images()
        
    def load_piece_images(self):
        # 创建简单的棋子表示（使用字母）
        pieces = {}
        piece_symbols = {
            'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
            'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
        }
        font = QFont('Arial', 24)
        for piece, symbol in piece_symbols.items():
            pixmap = QPixmap(self.square_size, self.square_size)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setFont(font)
            painter.drawText(pixmap.rect(), Qt.AlignCenter, symbol)
            pieces[piece] = pixmap
        return pieces
    
    def paintEvent(self, event):
        painter = QPainter(self)
        for row in range(8):
            for col in range(8):
                square_color = QColor(240, 217, 181) if (row + col) % 2 == 0 else QColor(181, 136, 99)
                painter.fillRect(col * self.square_size, row * self.square_size, 
                                self.square_size, self.square_size, square_color)
                
                square = chess.square(col, 7 - row)
                piece = self.board.piece_at(square)
                if piece:
                    piece_key = piece.symbol()
                    if piece_key in self.piece_images:
                        painter.drawPixmap(col * self.square_size, row * self.square_size, 
                                          self.square_size, self.square_size, 
                                          self.piece_images[piece_key])
        
        # 高亮选中格子
        if self.selected_square is not None:
            col = chess.square_file(self.selected_square)
            row = 7 - chess.square_rank(self.selected_square)
            painter.setPen(QColor(255, 0, 0))
            painter.drawRect(col * self.square_size, row * self.square_size, 
                            self.square_size, self.square_size, 3)
    
    def mousePressEvent(self, event):
        col = event.x() // self.square_size
        row = event.y() // self.square_size
        square = chess.square(col, 7 - row)
        
        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == chess.WHITE:  # 假设白方是玩家
                self.selected_square = square
                self.update()
        else:
            # 尝试移动棋子
            from_square = self.selected_square
            to_square = square
            move = chess.Move(from_square, to_square)
            
            # 检查是否是合法的吃子或移动
            if move in self.board.legal_moves:
                self.board.push(move)
                self.move_made.emit(move.uci())
            elif chess.Move(from_square, to_square, promotion=chess.QUEEN) in self.board.legal_moves:
                # 自动升变为皇后
                move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
                self.board.push(move)
                self.move_made.emit(move.uci())
            
            self.selected_square = None
            self.update()
    
    def set_board_position(self, board):
        self.board = board.copy()
        self.selected_square = None
        self.update()

class ChessEngine:
    def __init__(self):
        self.depth = 4  # AI搜索深度
        
    def evaluate_board(self, board):
        """评估棋盘局面"""
        if board.is_checkmate():
            if board.turn:
                return -9999  # 白方被将死
            else:
                return 9999   # 黑方被将死
        if board.is_stalemate() or board.is_insufficient_material():
            return 0  # 和棋
        
        # 计算棋子价值
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value
        
        # 添加位置价值（简化版）
        pawn_table = [
             0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
             5,  5, 10, 25, 25, 10,  5,  5,
             0,  0,  0, 20, 20,  0,  0,  0,
             5, -5,-10,  0,  0,-10, -5,  5,
             5, 10, 10,-20,-20, 10, 10,  5,
             0,  0,  0,  0,  0,  0,  0,  0
        ]
        
        knight_table = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ]
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                if piece.piece_type == chess.PAWN:
                    if piece.color == chess.WHITE:
                        score += pawn_table[square]
                    else:
                        score -= pawn_table[chess.square_mirror(square)]
                elif piece.piece_type == chess.KNIGHT:
                    if piece.color == chess.WHITE:
                        score += knight_table[square]
                    else:
                        score -= knight_table[chess.square_mirror(square)]
        
        return score if board.turn == chess.WHITE else -score
    
    def minimax(self, board, depth, alpha, beta, maximizing_player):
        """使用alpha-beta剪枝的minimax算法"""
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta剪枝
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta剪枝
            return min_eval
    
    def get_best_move(self, board):
        """获取最佳走法"""
        best_move = None
        best_value = float('-inf') if board.turn == chess.WHITE else float('inf')
        
        for move in board.legal_moves:
            board.push(move)
            board_value = self.minimax(board, self.depth - 1, float('-inf'), float('inf'), 
                                      board.turn == chess.BLACK)
            board.pop()
            
            if board.turn == chess.WHITE:
                if board_value > best_value:
                    best_value = board_value
                    best_move = move
            else:
                if board_value < best_value:
                    best_value = board_value
                    best_move = move
        
        return best_move

class ChessGameWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("国际象棋AI")
        self.setGeometry(100, 100, 800, 700)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # 棋盘
        self.chess_board = ChessBoard()
        self.layout.addWidget(self.chess_board)
        
        # 控制按钮
        self.control_layout = QHBoxLayout()
        
        self.new_game_btn = QPushButton("新游戏")
        self.new_game_btn.clicked.connect(self.new_game)
        self.control_layout.addWidget(self.new_game_btn)
        
        self.undo_btn = QPushButton("悔棋")
        self.undo_btn.clicked.connect(self.undo_move)
        self.control_layout.addWidget(self.undo_btn)
        
        self.status_label = QLabel("轮到白方走棋")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.control_layout.addWidget(self.status_label)
        
        self.layout.addLayout(self.control_layout)
        
        # 连接信号
        self.chess_board.move_made.connect(self.on_player_move)
        
        # 初始化游戏
        self.engine = ChessEngine()
        self.new_game()
        
    def new_game(self):
        self.chess_board.set_board_position(chess.Board())
        self.update_status()
        
    def undo_move(self):
        if len(self.chess_board.board.move_stack) >= 1:
            self.chess_board.board.pop()
            if len(self.chess_board.board.move_stack) >= 1:
                self.chess_board.board.pop()
            self.chess_board.update()
            self.update_status()
    
    def on_player_move(self, uci_move):
        self.update_status()
        # AI自动走棋
        QTimer.singleShot(500, self.ai_move)
    
    def ai_move(self):
        if not self.chess_board.board.is_game_over():
            ai_move = self.engine.get_best_move(self.chess_board.board)
            if ai_move:
                self.chess_board.board.push(ai_move)
                self.chess_board.update()
                self.update_status()
    
    def update_status(self):
        if self.chess_board.board.is_checkmate():
            winner = "白方" if self.chess_board.board.turn == chess.BLACK else "黑方"
            self.status_label.setText(f"将死！{winner}获胜！")
        elif self.chess_board.board.is_stalemate():
            self.status_label.setText("和棋（逼和）")
        elif self.chess_board.board.is_insufficient_material():
            self.status_label.setText("和棋（子力不足）")
        elif self.chess_board.board.is_check():
            self.status_label.setText("将军！")
        else:
            turn = "白方" if self.chess_board.board.turn == chess.WHITE else "黑方"
            self.status_label.setText(f"轮到{turn}走棋")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChessGameWindow()
    window.show()
    sys.exit(app.exec_())
