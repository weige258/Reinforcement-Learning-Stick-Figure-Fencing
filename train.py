"""
强化学习训练脚本 - 训练DQN智能体玩火柴人击剑格斗
支持自我对抗训练
"""
import os
import sys
import time
import numpy as np
from collections import deque

import torch

from game_config import *
from fencing_game import FencingGame
from dqn_agent import DQNAgent


class Trainer:
    """训练管理器"""
    
    def __init__(self, render=False):
        self.game = FencingGame(render=render)
        self.state_dim = RL['state_dim']
        self.action_dim = RL['action_dim']
        
        # 创建两个智能体 (自我对抗)
        self.agent_p1 = DQNAgent(self.state_dim, self.action_dim)
        self.agent_p2 = DQNAgent(self.state_dim, self.action_dim)
        
        # 训练统计
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.p1_wins = 0
        self.p2_wins = 0
        self.draws = 0
        
        # 模型保存路径
        self.model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(self.model_dir, exist_ok=True)
    
    def train_episode(self, episode):
        """训练一个episode"""
        state = self.game.reset()  # P1视角
        episode_reward_p1 = 0
        episode_reward_p2 = 0
        episode_loss = 0
        loss_count = 0
        
        for step in range(RL['max_steps_per_episode']):
            # 各自用自己视角的状态选动作
            action1 = self.agent_p1.select_action(state)  # state是P1视角
            
            # 获取P2视角的状态
            state_p2 = self.game._get_state(perspective=2)
            action2 = self.agent_p2.select_action(state_p2)
            
            # 执行动作 (step返回各自视角的next_state)
            next_state, reward1, next_state_p2, reward2, done, info = self.game.step(action1, action2)
            
            # 存储经验 (各自用自己的视角)
            self.agent_p1.store_transition(state, action1, next_state, reward1)
            self.agent_p2.store_transition(state_p2, action2, next_state_p2, reward2)
            
            # 优化
            loss1 = self.agent_p1.optimize_model()
            loss2 = self.agent_p2.optimize_model()
            
            if loss1 > 0:
                episode_loss += loss1
                loss_count += 1
            if loss2 > 0:
                episode_loss += loss2
                loss_count += 1
            
            # 每N步软更新一次目标网络 (减少开销)
            if step % RL['target_update'] == 0:
                self.agent_p1.soft_update_target()
                self.agent_p2.soft_update_target()
            
            # 累积奖励
            episode_reward_p1 += reward1
            episode_reward_p2 += reward2
            
            state = next_state  # P1视角
            
            # 渲染 + 检测窗口关闭
            if self.game.render:
                cont = self.game.render_frame()
                if not cont:  # 用户点了窗口X
                    print("\n🛑 窗口关闭, 正在保存模型...")
                    raise KeyboardInterrupt()
            
            if done:
                break
        
        # 统计
        self.episode_rewards.append(episode_reward_p1 + episode_reward_p2)
        self.episode_lengths.append(step + 1)
        
        if info['winner'] == self.game.player1:
            self.p1_wins += 1
        elif info['winner'] == self.game.player2:
            self.p2_wins += 1
        else:
            self.draws += 1
        
        avg_loss = episode_loss / max(loss_count, 1)
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
        
        return {
            'episode': episode,
            'reward_p1': episode_reward_p1,
            'reward_p2': episode_reward_p2,
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'avg_loss': avg_loss,
            'p1_health': info['health1'],
            'p2_health': info['health2'],
            'winner': 'P1' if info['winner'] == self.game.player1 else 'P2' if info['winner'] == self.game.player2 else 'Draw',
            'steps': step + 1,
            'p1_wins': self.p1_wins,
            'p2_wins': self.p2_wins,
            'draws': self.draws,
        }
    
    def train(self, num_episodes=None):
        """执行训练 - 支持无限循环训练"""
        infinite = (num_episodes is None or num_episodes <= 0)
        if infinite:
            num_episodes = float('inf')
            print_str = "∞ (无限循环, Ctrl+C 停止)"
        else:
            print_str = str(num_episodes)
        
        print("=" * 80)
        print("火柴人击剑格斗 RL 训练开始 (自我对抗)")
        print(f"状态维度: {self.state_dim}, 动作维度: {self.action_dim}")
        print(f"设备: {self.agent_p1.device}")
        print(f"训练轮数: {print_str}")
        print("=" * 80)
        
        start_time = time.time()
        best_avg_reward = -float('inf')
        episode = 0
        
        try:
            while True:
                episode += 1
                if not infinite and episode > num_episodes:
                    break
                
                stats = self.train_episode(episode)
                
                # 打印进度
                if episode % 10 == 0 or episode == 1:
                    elapsed = time.time() - start_time
                    ep_str = f"{stats['episode']}/∞" if infinite else f"{stats['episode']}/{num_episodes}"
                    print(f"\nEpisode {ep_str} | "
                          f"Time: {elapsed:.1f}s | "
                          f"Steps: {stats['steps']} | "
                          f"Winner: {stats['winner']}")
                    print(f"  P1 Health: {stats['p1_health']:.0f} | "
                          f"P2 Health: {stats['p2_health']:.0f} | "
                          f"Avg Reward: {stats['avg_reward']:.2f}")
                    print(f"  Avg Loss: {stats['avg_loss']:.4f} | "
                          f"Avg Length: {stats['avg_length']:.1f}")
                    print(f"  Wins - P1: {self.p1_wins} | P2: {self.p2_wins} | "
                          f"Draws: {self.draws}")
                    
                    # 记录损失
                    self.agent_p1.loss_history.append(stats['avg_loss'])
                
                # 保存最佳模型 (仅当优于上次)
                if stats['avg_reward'] > best_avg_reward and episode > 100:
                    best_avg_reward = stats['avg_reward']
                    self.save_models()
        
        except KeyboardInterrupt:
            print(f"\n\n🛑 训练被用户中断 (第{episode}轮)")
        
        # 结束时仅保存一次
        self.save_models()
        
        total_time = time.time() - start_time
        print(f"\n{'=' * 80}")
        print(f"训练完成! 总时间: {total_time:.1f}s, 共{episode}轮")
        print(f"P1 胜: {self.p1_wins}, P2 胜: {self.p2_wins}, 平局: {self.draws}")
        print(f"胜率: P1={self.p1_wins/max(episode,1)*100:.1f}%, "
              f"P2={self.p2_wins/max(episode,1)*100:.1f}%")
        print(f"{'=' * 80}")
    
    def save_models(self):
        """保存模型 (只保留1份, 覆盖旧文件)"""
        self.agent_p1.save(os.path.join(self.model_dir, "agent_p1_best.pth"))
        self.agent_p2.save(os.path.join(self.model_dir, "agent_p2_best.pth"))
    
    def evaluate(self, num_episodes=10):
        """评估训练好的智能体"""
        import pygame  # 确保pygame已导入
        
        self.agent_p1.policy_net.eval()
        self.agent_p2.policy_net.eval()
        
        wins = 0
        total_health_diff = 0
        
        for ep in range(num_episodes):
            state = self.game.reset()
            ep_reward = 0
            
            for step in range(RL['max_steps_per_episode']):
                action1 = self.agent_p1.select_action(state, eval_mode=True)
                state_p2 = self.game._get_state(perspective=2)
                action2 = self.agent_p2.select_action(state_p2, eval_mode=True)
                
                next_state, reward1, next_state_p2, reward2, done, info = self.game.step(action1, action2)
                ep_reward += reward1
                state = next_state
                
                if self.game.render:
                    cont = self.game.render_frame()
                    if not cont:
                        break
                    pygame.time.wait(10)
                
                if done:
                    break
            
            if info['winner'] == self.game.player1:
                wins += 1
            total_health_diff += (info['health1'] - info['health2'])
        
        print(f"评估结果 ({num_episodes} 局):")
        print(f"  智能体1胜率: {wins/num_episodes*100:.1f}%")
        print(f"  平均血量差: {total_health_diff/num_episodes:.1f}")
        
        return wins / num_episodes


def main():
    """主训练入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='火柴人击剑格斗 RL 训练')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'play', 'train_vs'],
                       help='运行模式')
    parser.add_argument('--episodes', type=int, default=RL['max_episodes'],
                       help='训练轮数')
    parser.add_argument('--render', action='store_true',
                       help='显示渲染窗口')
    parser.add_argument('--load', type=str, default='',
                       help='加载模型路径')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        trainer = Trainer(render=args.render)
        if args.load:
            trainer.agent_p1.load(args.load)
        trainer.train(num_episodes=args.episodes)
    
    elif args.mode == 'eval':
        trainer = Trainer(render=args.render)
        if args.load:
            trainer.agent_p1.load(args.load)
        trainer.evaluate(num_episodes=20)
    
    elif args.mode == 'play':
        # 人机对战
        from play import human_vs_ai
        human_vs_ai(model_path=args.load)
    
    elif args.mode == 'train_vs':
        # 训练时显示画面
        trainer = Trainer(render=True)
        trainer.train(num_episodes=args.episodes)


if __name__ == '__main__':
    # 需要导入pygame用于evaluate
    import pygame
    main()
