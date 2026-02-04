import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QMessageBox,
                             QDesktopWidget)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QFont, QColor
import chess
from logic import ChessEngine  # 导入你的逻辑层


class ChessApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("国际象棋 - 感受绝望")
        self.setGeometry(100, 100, 800, 800)

        # 游戏核心
        self.board = chess.Board()
        self.engine = ChessEngine(depth=4)  # 调整深度以匹配2000-2500 ELO

        # UI 状态
        self.selected_square = None
        self.is_user_turn = True
        self.move_made_by_ui = False

        # 设置中央部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 棋盘区域
        self.board_widget = BoardWidget(self)
        main_layout.addWidget(self.board_widget)

        # 控制按钮
        control_layout = QHBoxLayout()
        
        self.new_game_button = QPushButton("新游戏")
        self.undo_button = QPushButton("悔棋")
        self.quit_button = QPushButton("退出")

        self.new_game_button.clicked.connect(self.new_game)
        self.undo_button.clicked.connect(self.undo_move)
        self.quit_button.clicked.connect(self.close)

        control_layout.addWidget(self.new_game_button)
        control_layout.addWidget(self.undo_button)
        control_layout.addWidget(self.quit_button)

        main_layout.addLayout(control_layout)

        # 状态栏
        self.game_status_label = QLabel("游戏状态：等待开始")
        self.statusBar().addWidget(self.game_status_label)

        self.update_status_bar("你的回合 (白方)")
        self.center_on_screen()

    def center_on_screen(self):
        """将窗口居中显示"""
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def new_game(self):
        """开始新游戏"""
        self.board.reset()
        self.selected_square = None
        self.is_user_turn = True
        self.board_widget.update()  # 刷新棋盘显示
        self.update_status_bar("你的回合 (白方)")

    def undo_move(self):
        """悔棋一步"""
        if len(self.board.move_stack) >= 1:
            self.board.pop()
            if len(self.board.move_stack) >= 1:
                self.board.pop()
            self.is_user_turn = True
            self.board_widget.update()
            self.update_status_bar("你的回合 (白方)")

    def update_status_bar(self, text):
        """
        更新状态栏，并处理对象可能被删除的情况。
        """
        try:
            # 在更新前检查 QLabel 对象是否仍然有效
            # 如果窗口被关闭，QLabel 可能已被销毁，访问它会抛出 RuntimeError
            if self.game_status_label is not None and isinstance(self.game_status_label, QLabel):
                self.game_status_label.setText(f"游戏状态：{text}")
            else:
                # 如果对象无效，则不再执行任何操作
                pass
        except RuntimeError:
            # 捕获 'wrapped C/C++ object ... deleted' 错误
            # 这表示UI对象已经不存在，应停止所有相关操作
            print("警告: 尝试更新已销毁的UI组件，操作已取消。")

    def make_engine_move(self):
        """让AI执行一步棋"""
        if self.board.is_game_over():
            result = self.board.result()
            status_map = {
                "1-0": "白方获胜!",
                "0-1": "黑方获胜!",
                "1/2-1/2": "平局!"
            }
            final_status = status_map.get(result, f"游戏结束: {result}")
            self.update_status_bar(final_status)
            return

        if not self.is_user_turn:
            move = self.engine.get_best_move(self.board)
            if move:
                self.board.push(move)
                self.move_made_by_ui = True  # 标记移动由UI流程产生
                self.board_widget.update()
                self.update_status_bar("你的回合 (白方)")
                self.is_user_turn = True
                
                # 检查AI走棋后游戏是否结束
                if self.board.is_game_over():
                    result = self.board.result()
                    status_map = {
                        "1-0": "白方获胜! (这不可能！)",
                        "0-1": "黑方获胜! (绝望感+1)",
                        "1/2-1/2": "平局! (你居然能和我打平？)"
                    }
                    final_status = status_map.get(result, f"游戏结束: {result}")
                    # 使用 QTimer.singleShot 来延迟更新状态栏，避免在AI思考时卡顿
                    # 同时增加对窗口是否存在的保护
                    QTimer.singleShot(100, lambda: self.update_status_bar_if_exists(final_status))

    def update_status_bar_if_exists(self, text):
        """
        一个安全的更新方法，供 QTimer.singleShot 调用。
        """
        if not self.isHidden():  # 检查窗口是否还存在（未被隐藏/销毁）
            self.update_status_bar(text)


class BoardWidget(QWidget):
    square_size = 80

    def __init__(self, parent):
        super().__init__()
        self.parent_app = parent
        self.setFixedSize(8 * self.square_size, 8 * self.square_size)
        self.piece_images = self.load_piece_images()

    def load_piece_images(self):
        """加载棋子图像"""
        pieces = ['P', 'N', 'B', 'R', 'Q', 'K']
        colors = ['w', 'b']
        images = {}
        for color in colors:
            for piece in pieces:
                filename = f"assets/{color}{piece.lower()}.png"
                pixmap = QPixmap(filename)
                if pixmap.isNull():
                    print(f"警告: 无法加载图像 {filename}")
                    continue
                scaled_pixmap = pixmap.scaled(self.square_size, self.square_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                images[color + piece.lower()] = scaled_pixmap
        return images

    def paintEvent(self, event):
        """绘制棋盘和棋子"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        for row in range(8):
            for col in range(8):
                x = col * self.square_size
                y = (7 - row) * self.square_size  # 翻转Y轴使a1在左下角

                square_color = (row + col) % 2
                if square_color == 0:
                    painter.fillRect(x, y, self.square_size, self.square_size, QColor(240, 217, 181))  # 浅色格子
                else:
                    painter.fillRect(x, y, self.square_size, self.square_size, QColor(181, 136, 99))   # 深色格子

                if self.parent_app.selected_square is not None:
                    selected_row, selected_col = divmod(self.parent_app.selected_square, 8)
                    if row == selected_row and col == selected_col:
                        painter.fillRect(x, y, self.square_size, self.square_size, QColor(150, 200, 255, 150))  # 高亮选中格

                square_index = chess.square(col, 7 - row)
                piece = self.parent_app.board.piece_at(square_index)
                if piece:
                    piece_key = f"{piece.color and 'w' or 'b'}{piece.symbol().lower()}"
                    if piece_key in self.piece_images:
                        painter.drawPixmap(x, y, self.piece_images[piece_key])

    def mousePressEvent(self, event):
        """处理鼠标点击事件"""
        if not self.parent_app.is_user_turn or not self.isEnabled():
            return

        col = event.x() // self.square_size
        row = 7 - (event.y() // self.square_size)  # 翻转Y轴
        square_index = chess.square(col, row)

        if event.button() == Qt.LeftButton:
            if self.parent_app.selected_square is None:
                piece = self.parent_app.board.piece_at(square_index)
                if piece and piece.color == chess.WHITE:
                    self.parent_app.selected_square = square_index
                    self.update()
            else:
                from_square = self.parent_app.selected_square
                to_square = square_index
                
                if from_square == to_square:
                    # 取消选择
                    self.parent_app.selected_square = None
                    self.update()
                    return

                move = chess.Move(from_square, to_square)

                # 检查是否为吃子或升变
                if self.parent_app.board.piece_at(to_square) or self.parent_app.board.piece_at(from_square).piece_type == chess.PAWN:
                    # 检查是否为合法的吃子或兵的移动，以确定是否需要升变提示
                    if move in self.parent_app.board.legal_moves:
                        # 如果是兵到底线，需要升变
                        if self.parent_app.board.piece_at(from_square).piece_type == chess.PAWN and (to_square // 8 == 0 or to_square // 8 == 7):
                            promotion = self.get_promotion_piece()
                            if promotion:
                                move = chess.Move(from_square, to_square, promotion=promotion)
                        self.attempt_move(move)
                else:
                    # 快速移动，先尝试移动
                    self.attempt_move(move)

                self.parent_app.selected_square = None
                self.update()

    def attempt_move(self, move):
        """尝试执行移动"""
        if move in self.parent_app.board.legal_moves:
            self.parent_app.board.push(move)
            self.parent_app.move_made_by_ui = True
            self.update()
            self.parent_app.update_status_bar("AI 思考中...")
            self.parent_app.is_user_turn = False
            
            # 使用 QTimer.singleShot 延迟AI走棋，防止界面冻结
            # 并增加对窗口是否存在的保护
            if not self.parent_app.isHidden():
                QTimer.singleShot(100, self.parent_app.make_engine_move)
        else:
            # 检查是否为王车易位
            uci_move_str = move.uci()
            if uci_move_str in ["e1g1", "e1c1", "e8g8", "e8c8"]:
                castling_move = None
                if uci_move_str == "e1g1" and self.parent_app.board.is_kingside_castling(chess.Move.from_uci(uci_move_str)):
                    castling_move = chess.Move.from_uci("e1g1")
                elif uci_move_str == "e1c1" and self.parent_app.board.is_queenside_castling(chess.Move.from_uci(uci_move_str)):
                    castling_move = chess.Move.from_uci("e1c1")
                elif uci_move_str == "e8g8" and self.parent_app.board.is_kingside_castling(chess.Move.from_uci(uci_move_str)):
                    castling_move = chess.Move.from_uci("e8g8")
                elif uci_move_str == "e8c8" and self.parent_app.board.is_queenside_castling(chess.Move.from_uci(uci_move_str)):
                    castling_move = chess.Move.from_uci("e8c8")
                
                if castling_move and castling_move in self.parent_app.board.legal_moves:
                    self.parent_app.board.push(castling_move)
                    self.parent_app.move_made_by_ui = True
                    self.update()
                    self.parent_app.update_status_bar("AI 思考中...")
                    self.parent_app.is_user_turn = False
                    if not self.parent_app.isHidden():
                        QTimer.singleShot(100, self.parent_app.make_engine_move)
            else:
                # 移动非法，取消选择
                pass

    def get_promotion_piece(self):
        """获取升变棋子类型"""
        # 简化处理，直接返回QUEEN
        # 在更复杂的实现中，这里可以弹出一个对话框让用户选择
        return chess.QUEEN


def main():
    app = QApplication(sys.argv)
    window = ChessApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()