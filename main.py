"""
火柴人击剑格斗游戏 - 主入口
基于Bloody Bastards风格，使用PyMunk物理引擎 + PyTorch DQN强化学习

操作说明:
  WASD 移动 | 鼠标左键 挥剑 | 鼠标右键 举盾

参考论文:
  - Creating Pro-Level AI for a Real-Time Fighting Game Using DRL (2019)
  - Diversity-based DRL for Fighting Game AI (2022)
  - MAAIP: Multi-Agent Adversarial Interaction Priors (2023)
"""
import sys
import os
import game_config as cfg


def print_banner():
    """打印游戏标题"""
    banner = """
    ╔══════════════════════════════════════════════════╗
    ║         ⚔ 火柴人击剑格斗 ⚔                       ║
    ║     Bloody Bastards 风格 · PyMunk物理引擎        ║
    ║     PyTorch DQN 强化学习 · 自对弈训练             ║
    ╚══════════════════════════════════════════════════╝

    参考论文:
      [1] Pro-Level Fighting Game AI - DRL (2019)
      [2] Diversity-based DRL for Fighting Game AI (2022)
      [3] MAAIP - Multi-Agent Adversarial Interaction Priors (2023)

    """
    print(banner)


def print_help():
    """打印帮助信息"""
    help_text = """
    可用命令:

      python main.py --mode play         人机对战（默认）
      python main.py --mode train        无界面训练
      python main.py --mode train_vs     带界面训练
      python main.py --mode eval         评估AI
      python main.py --mode ai_vs_ai     AI互相对战观看

    参数:
      --episodes N   训练回合数 (默认: 500)
      --model PATH   AI模型路径 (默认: models/agent_p1_final.pth)
      --no-render    训练时不显示画面

    示例:
      python main.py --mode train --episodes 1000
      python main.py --mode play --model models/agent_p1_best.pth
      python main.py --mode ai_vs_ai --model models/agent_p1_final.pth
    """
    print(help_text)


def run_mode(mode, model='models/agent_p1_final.pth', model_p2=None, episodes=500, render=True):
    """运行指定模式"""
    if mode == 'play':
        from play import HumanVsAI
        game = HumanVsAI(model_path=model)
        game.run()
    elif mode == 'train':
        from train import Trainer, MultiTrainer
        # 从菜单中选择训练 = 9宫格模式, 命令行 --mode train 可选渲染
        trainer = MultiTrainer(total_episodes=episodes) if render else Trainer(render=render, total_episodes=episodes)
        trainer.train()
    elif mode == 'ai_vs_ai':
        from play import AIVsAI
        game = AIVsAI(model_path_p1=model, model_path_p2=model_p2)
        game.run()
    elif mode == 'exit':
        print("感谢游玩!")
        sys.exit(0)


def main():
    """主入口 - 先显示菜单，再进入对应模式"""
    import argparse

    parser = argparse.ArgumentParser(
        description='火柴人击剑格斗 - Bloody Bastards风格',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--mode', type=str, default=None,
                       choices=['play', 'train', 'train_vs', 'eval', 'ai_vs_ai'],
                       help='直接进入模式（跳过菜单）')
    parser.add_argument('--episodes', type=int, default=500, help='训练回合数')
    parser.add_argument('--model', type=str, default='models/agent_p1_final.pth', help='AI模型路径')
    parser.add_argument('--model-p2', type=str, default=None, help='玩家2AI模型路径')
    parser.add_argument('--no-render', action='store_true', help='训练时不显示画面')

    args = parser.parse_args()

    # 如果命令行指定了模式，直接运行
    if args.mode:
        print_banner()
        mode = args.mode
        if mode == 'train_vs':
            mode = 'train'
            args.no_render = False
        run_mode(mode, args.model, args.model_p2, args.episodes, not args.no_render)
        return

    # 否则显示菜单
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT))
    pygame.display.set_caption('⚔ 火柴人击剑格斗')

    from menu import Menu
    menu = Menu(screen)
    mode = menu.run()

    if mode == 'exit':
        pygame.quit()
        sys.exit(0)

    # 运行选择的模式
    if mode == 'train':
        run_mode('train', episodes=args.episodes, render=True)
    else:
        run_mode(mode)


if __name__ == '__main__':
    main()
