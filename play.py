"""
人机对战模式 / AI vs AI 观看模式
玩家使用 WASD 移动，鼠标左键挥剑，鼠标右键举盾
"""
import os
import sys
import pygame

import game_config as cfg
from fencing_game import FencingGame
from dqn_agent import DQNAgent


class HumanVsAI:
    """人机对战模式"""

    def __init__(self, model_path=None, device='cpu'):
        self.game = FencingGame(render=True)

        # 创建AI智能体（玩家2为AI）- 自动加载或创建
        self.ai_agent = DQNAgent.load_or_create(
            model_path or 'models/agent_p1_final.pth',
            player_id=2, device=device
        )
        self.ai_agent.training = False

        # 游戏状态
        self.running = True
        self.paused = False
        self.frame_count = 0  # AI决策帧计数器
        self.game.create_players()

    def run(self):
        """运行主循环"""
        print(f"\n{'='*50}")
        print("  火柴人击剑格斗 - 人机对战")
        print(f"{'='*50}")
        print("  操作说明:")
        print("  - WASD: 移动")
        print("  - 鼠标左键: 挥剑")
        print("  - 鼠标右键: 举盾")
        print("  - 鼠标移动: 瞄准")
        print("  - R: 重新开始")
        print("  - ESC: 退出")
        print(f"{'='*50}\n")

        while self.running:
            # 处理事件
            events = self.game.handle_events()
            for event in events:
                if event == 'quit':
                    self.running = False
                    return
                elif event == 'reset':
                    self.game.reset()  # reset内部已调用create_players

            if self.paused:
                continue

            if not self.game.game_over:
                # 获取人类玩家输入
                human_action = self.game.get_human_input(player_id=1)

                # AI每3帧决策一次(60/3=20次/秒)
                if self.frame_count % 3 == 0:
                    state = self.game.get_state_for_agent(2)
                    if state is not None:
                        self._ai_action_idx = self.ai_agent.select_action(state, eval_mode=True)
                    else:
                        self._ai_action_idx = 0
                ai_action = self.ai_agent.get_action_tuple(self._ai_action_idx)
                self.frame_count += 1

                # 执行一步(不强制翻转朝向, 角色固定初始朝向)
                self.game.step(human_action, ai_action)

            # 渲染
            self.game.render_frame()

        self.game.close()


class AIVsAI:
    """AI vs AI 观看模式"""

    def __init__(self, model_path_p1=None, model_path_p2=None, device='cpu'):
        self.game = FencingGame(render=True)

        # 创建两个AI - 使用自动加载/创建
        self.agent1 = DQNAgent.load_or_create(
            model_path_p1 or 'models/agent_p1_final.pth',
            player_id=1, device=device
        )
        self.agent2 = DQNAgent.load_or_create(
            model_path_p2 or 'models/agent_p2_final.pth',
            player_id=2, device=device
        )
        self.agent1.training = False
        self.agent2.training = False

        self.running = True
        self.frame_count = 0
        self._ai1_idx = 0
        self._ai2_idx = 0
        self.game.create_players()

    def run(self):
        """运行AI对战"""
        print(f"\n{'='*50}")
        print("  AI vs AI 对战模式")
        print(f"{'='*50}")
        print("  按 ESC 退出 | 按 R 重新开始")
        print(f"{'='*50}\n")

        while self.running:
            events = self.game.handle_events()
            for event in events:
                if event == 'quit':
                    self.running = False
                    return
                elif event == 'reset':
                    self.game.reset()

            if not self.game.game_over:
                # 获取状态(AI每3帧决策)
                if self.frame_count % 3 == 0:
                    state_1 = self.game.get_state_for_agent(1)
                    state_2 = self.game.get_state_for_agent(2)
                    if state_1 is not None and state_2 is not None:
                        self._ai1_idx = self.agent1.select_action(state_1, eval_mode=True)
                        self._ai2_idx = self.agent2.select_action(state_2, eval_mode=True)

                action_1 = self.agent1.get_action_tuple(self._ai1_idx)
                action_2 = self.agent2.get_action_tuple(self._ai2_idx)
                self.frame_count += 1

                # 执行一步(AI不翻转朝向)

            self.game.render_frame()

        self.game.close()


def main():
    """入口"""
    import argparse

    parser = argparse.ArgumentParser(description='火柴人击剑格斗 - 对战模式')
    parser.add_argument('--mode', type=str, default='human_vs_ai',
                       choices=['human_vs_ai', 'ai_vs_ai'],
                       help='对战模式')
    parser.add_argument('--model', type=str, default='models/agent_p1_final.pth',
                       help='AI模型路径')
    parser.add_argument('--model-p2', type=str, default=None,
                       help='玩家2AI模型路径 (AI vs AI模式)')
    args = parser.parse_args()

    if args.mode == 'human_vs_ai':
        game = HumanVsAI(model_path=args.model)
        game.run()
    elif args.mode == 'ai_vs_ai':
        game = AIVsAI(model_path_p1=args.model, model_path_p2=args.model_p2)
        game.run()


if __name__ == '__main__':
    main()
