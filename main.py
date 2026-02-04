"""
main.py - 国际象棋人机对战程序启动脚本

本脚本作为程序的入口点，负责初始化PyQt应用程序并启动主窗口。
用户执白方，AI执黑方，通过图形界面进行完整规则的国际象棋对战。
支持新游戏、悔棋、退出等控制功能，AI能力达到2000-2500 ELO水平。

作者: AI助手
日期: 2026-02-04
"""

import sys
from gui import ChessMainWindow  # 导入图形界面主窗口类
from PyQt5.QtWidgets import QApplication


def main():
    """
    主函数：启动国际象棋人机对战程序。
    
    初始化QApplication，创建并显示主窗口，启动事件循环。
    程序退出时正确释放资源。
    """
    # 创建Qt应用程序实例
    app = QApplication(sys.argv)
    
    # 设置应用程序名称
    app.setApplicationName("国际象棋人机对战")
    
    # 创建主窗口实例
    window = ChessMainWindow()
    
    # 显示主窗口
    window.show()
    
    # 启动事件循环，并在窗口关闭时安全退出
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
