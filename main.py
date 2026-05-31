"""
火柴人击剑格斗 - Bloody Bastards 风格
强化学习 + PyMunk物理引擎 + Pygame渲染

用法:
  python main.py          # 显示菜单选择
  python main.py 1        # 直接进入训练
  python main.py 2        # 直接进入人机对战
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    print("""
    ╔══════════════════════════════════════════════════╗
    ║       ⚔️ 火柴人击剑格斗 - RL 强化学习 ⚔️        ║
    ║     Bloody Bastards 风格物理引擎格斗游戏         ║
    ║        PyMunk + PyTorch + Pygame               ║
    ╚══════════════════════════════════════════════════╝
    """)


def show_menu():
    """显示交互菜单"""
    print("=" * 50)
    print("  ⚔️  请选择模式:")
    print("  " + "─" * 46)
    print("   1️⃣   训练        (多局AI自我对抗, 宫格视图)")
    print("   2️⃣   人机对战    (你 vs AI, 鼠标+键盘)")
    print("  " + "─" * 46)
    print("  输入 1 或 2 后回车: ", end="", flush=True)
    try:
        choice = input().strip()
        return choice
    except (EOFError, KeyboardInterrupt):
        return ""


def find_best_model():
    """自动寻找最佳模型"""
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.exists(models_dir):
        return ''
    files = [f for f in os.listdir(models_dir)
             if f.startswith('agent_p2') and f.endswith('.pth')]
    if not files:
        return ''
    preferred = [f for f in files if 'episode' in f or 'final' in f]
    return os.path.join(models_dir, (preferred or files)[0])


def mode_train():
    """多宫格训练模式"""
    print("\n🔥 启动多宫格训练 (同时监控多局AI训练)...")
    print("  提示: 按 ESC 或关闭窗口可停止并保存模型\n")
    from multi_viewer import start_multi_training
    start_multi_training(grid_rows=3, grid_cols=3)


def mode_play():
    """人机对战模式"""
    print("\n🎮 启动人机对战模式...")
    print("  鼠标拖拽挥剑 | WASD移动 | Space重置 | ESC退出\n")
    from play import human_vs_ai
    human_vs_ai(model_path=find_best_model())


def main():
    print_banner()

    if len(sys.argv) > 1 and sys.argv[1] in ("1", "2"):
        choice = sys.argv[1]
    else:
        choice = show_menu()

    if choice == "1":
        mode_train()
    elif choice == "2":
        mode_play()
    else:
        print("\n❌ 无效选择, 默认进入人机对战\n")
        mode_play()


if __name__ == '__main__':
    main()
