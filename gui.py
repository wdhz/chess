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
from logic import ChessGameLogic  # 导入自定义游戏逻辑模块


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