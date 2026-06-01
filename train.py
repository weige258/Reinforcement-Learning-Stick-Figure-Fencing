"""
训练脚本 - 自我对抗训练DQN智能体
使用自对弈课程学习（参考Pro-Level Fighting Game AI论文方法）
"""
import time
import os
import sys
import math
import signal

import numpy as np
import pygame
import torch

import game_config as cfg
from fencing_game import FencingGame
from dqn_agent import DQNAgent, SelfPlayManager


class Trainer:
    """训练管理器"""

    def __init__(self, render=False, total_episodes=1000, model_dir='models'):
        self.render = render
        self.total_episodes = total_episodes
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, 'agent_p1_final.pth')

        # 创建游戏环境
        self.game = FencingGame(render=render)

        # 创建自对弈管理器 - 自动加载已有模型
        self.self_play = SelfPlayManager()
        self.self_play.agent1 = DQNAgent.load_or_create(
            self.model_path, player_id=1
        )
        # agent2 始终新建（对手不需要保存）
        self.self_play.agent2 = DQNAgent(player_id=2)

        # 训练统计
        self.best_reward = float('-inf')
        self.loss_history = []
        self.reward_history = []
        self._interrupted = False

    def _signal_handler(self, sig, frame):
        """处理Ctrl+C中断信号"""
        self._interrupted = True
        print("\n\n检测到中断信号, 正在保存模型...")

    def _save_latest(self):
        """保存最新模型(仅保留best和final两个)"""
        self.self_play.agent1.save(self.model_path)

    def train(self):
        """主训练循环"""
        # 注册SIGINT处理器(Ctrl+C)
        old_handler = signal.signal(signal.SIGINT, self._signal_handler)

        print(f"{'='*60}")
        print(f"  火柴人击剑格斗 - DQN训练")
        print(f"{'='*60}")
        print(f"\n训练设备: {self.self_play.agent1.device}")
        print(f"训练回合数: {self.total_episodes}")
        print(f"已有经验: {len(self.self_play.agent1.memory)}条")
        print(f"{'='*60}\n")

        try:
            for episode in range(1, self.total_episodes + 1):
                if self._interrupted:
                    print("\n训练中断, 保存模型...")
                    break

                # 更新课程
                self.self_play.update_curriculum(episode)

                state = self.game.reset()
                if state is None:
                    continue

                state_p1 = state['player1']
                state_p2 = state['player2']

                episode_reward_p1 = 0
                episode_reward_p2 = 0
                step = 0
                done = False
                max_steps = 1200
                stuck_frames = 0
                STUCK_THRESHOLD = 120  # 2秒无变化视为卡住
                # 课程学习: 根据阶段调整对手强度
                stage = self.self_play.curriculum_stage
                agent2_eps = [0.7, 0.5, 0.3, 0.1][min(stage, 3)]
                self.self_play.agent2.epsilon = agent2_eps

                while not done and step < max_steps:
                    if self._interrupted:
                        break

                    # 渲染模式下定期轮询事件, 防止窗口卡死
                    if self.render and step % 10 == 0:
                        for e in pygame.event.get():
                            if e.type == pygame.QUIT:
                                self._interrupted = True
                                done = True
                                break
                            elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                                self._interrupted = True
                                done = True
                                break

                    action_idx_1 = self.self_play.agent1.select_action(state_p1)
                    action_idx_2 = self.self_play.agent2.select_action(state_p2)

                    action_1 = self.self_play.agent1.get_action_tuple(action_idx_1)
                    action_2 = self.self_play.agent2.get_action_tuple(action_idx_2)

                    # 执行动作
                    reward = self.game.step(action_1, action_2)

                    # 检查玩家是否都倒在地上(躯干Y接近地面)
                    if self.game.player1 and self.game.player2:
                        p1_y = self.game.player1.get_position().y
                        p2_y = self.game.player2.get_position().y
                        if p1_y > cfg.GROUND_Y - 80 and p2_y > cfg.GROUND_Y - 80:
                            stuck_frames += 1
                            if stuck_frames > STUCK_THRESHOLD:
                                done = True
                                break
                        else:
                            stuck_frames = 0

                    # 获取新状态
                    next_state_p1 = self.game.get_state_for_agent(1)
                    next_state_p2 = self.game.get_state_for_agent(2)

                    if next_state_p1 is None or next_state_p2 is None:
                        break

                    reward_p1 = reward['player1']
                    reward_p2 = reward['player2']

                    # 存储经验(只存agent1的, agent2不需要训练)
                    done_flag = 1.0 if self.game.game_over else 0.0
                    self.self_play.agent1.memory.push(
                        state_p1, action_idx_1, reward_p1, next_state_p1, done_flag
                    )

                    # 学习(只训练agent1)
                    if step % cfg.TRAIN_INTERVAL == 0:
                        loss1 = self.self_play.agent1.learn()
                        if loss1 is not None:
                            if len(self.loss_history) < 1000:
                                self.loss_history.append(loss1)

                    # 更新状态
                    state_p1 = next_state_p1
                    state_p2 = next_state_p2
                    episode_reward_p1 += reward_p1
                    episode_reward_p2 += reward_p2
                    step += 1

                    # 如果渲染，显示游戏
                    if self.render and step % 3 == 0:
                        self.game.render_frame()

            # 回合结束统计
            self.self_play.episode_rewards[1].append(episode_reward_p1)
            self.self_play.episode_rewards[2].append(episode_reward_p2)
            self.self_play.episode_lengths.append(step)

            if self.game.winner == 1:
                self.self_play.win_counts[1] += 1
            elif self.game.winner == 2:
                self.self_play.win_counts[2] += 1

            self.reward_history.append(episode_reward_p1)

            # 保存最佳模型
            if episode_reward_p1 > self.best_reward:
                self.best_reward = episode_reward_p1
                self.self_play.agent1.save(
                    os.path.join(self.model_dir, 'agent_p1_best.pth')
                )

            # 打印进度
            if episode % 10 == 0:
                stats = self.self_play.get_stats()
                print(
                    f"回合 {episode}/{self.total_episodes} | "
                    f"奖励: {episode_reward_p1:6.1f} | "
                    f"胜率: {self.self_play.win_counts[1] / max(1, sum(self.self_play.win_counts.values())) * 100:.1f}% | "
                    f"ε: {self.self_play.agent1.epsilon:.3f} | "
                    f"步数: {step} | "
                    f"课阶: {self.self_play.curriculum_stage+1}"
                )

            # 保存最新模型
            if episode == self.total_episodes or self._interrupted:
                self._save_latest()
                print(f"\n{'='*60}")
                print(f"  训练{'完成' if not self._interrupted else '中断'}!")
                print(f"  总回合: {len(self.self_play.episode_rewards[1])}")
                wr = self.self_play.win_counts[1] / max(1, sum(self.self_play.win_counts.values())) * 100
                print(f"  胜率: {wr:.1f}%")
                print(f"  最佳奖励: {self.best_reward:.1f}")
                if self.self_play.episode_lengths:
                    print(f"  平均回合长度: {np.mean(self.self_play.episode_lengths[-100:]):.0f} 步")
                print(f"{'='*60}")

            return self.self_play
        finally:
            self._save_latest()
            signal.signal(signal.SIGINT, old_handler)

class MultiTrainer:
    """4x4宫格并行训练 - 16个环境, CUDA加速"""

    def __init__(self, total_episodes=1000, model_dir='models'):
        self.total_episodes = total_episodes
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, 'agent_p1_final.pth')

        # CUDA检测
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 4x4 = 16个并行环境
        self.grid_size = 4
        self.num_envs = 16
        self.envs = [FencingGame(render=False) for _ in range(self.num_envs)]
        for env in self.envs:
            env.create_players()

        # 共享DQN智能体(CUDA)
        self.agent = DQNAgent.load_or_create(self.model_path, player_id=1, device=self.device)
        self.agent.training = True
        # 对手用随机策略(CPU即可)
        self.opponents = [DQNAgent(player_id=2, device='cpu') for _ in range(self.num_envs)]

        # 模型参数统计
        total_params = sum(p.numel() for p in self.agent.policy_net.parameters())
        print(f"\n[模型] {total_params:,} 参数 | 设备: {self.device.upper()}")
        print(f"[训练] 4x4={self.num_envs}环境 | {self.total_episodes}回合\n")

        # 训练统计
        self.best_reward = float('-inf')
        self._interrupted = False
        self._screen = None
        self._font = None

    def _init_display(self):
        pygame.init()
        self._screen = pygame.display.set_mode((cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT))
        pygame.display.set_caption(f"16宫格训练 - {self.device.upper()} - DQN")
        self._font = pygame.font.Font(None, 14)
        self._cell_w = cfg.WINDOW_WIDTH // 4
        self._cell_h = cfg.WINDOW_HEIGHT // 4

    def _signal_handler(self, sig, frame):
        self._interrupted = True

    def _render_grid(self, episode, step, total_reward):
        if not self._screen: return
        self._screen.fill((15, 15, 25))

        for idx, env in enumerate(self.envs):
            row, col = idx // 4, idx % 4
            cx, cy = col * self._cell_w, row * self._cell_h
            self._draw_mini_game(env, cx, cy, self._cell_w, self._cell_h, idx)

        # 顶部信息栏
        avg_r = total_reward / max(1, self.num_envs)
        info = f"Ep {episode}/{self.total_episodes} | Reward {avg_r:.1f} | eps {self.agent.epsilon:.3f} | {self.device.upper()} | 16x"
        txt = self._font.render(info, True, (180, 220, 180))
        self._screen.blit(txt, (8, 3))

        # 网格线
        for i in range(1, 4):
            pygame.draw.line(self._screen, (50, 50, 70), (i*self._cell_w, 0), (i*self._cell_w, cfg.WINDOW_HEIGHT), 1)
            pygame.draw.line(self._screen, (50, 50, 70), (0, i*self._cell_h), (cfg.WINDOW_WIDTH, i*self._cell_h), 1)

        pygame.display.flip()
        # 网格线
        for i in range(1, 3):
            pygame.draw.line(self._screen, (60, 60, 80), (i*self._cell_w, 0), (i*self._cell_w, cfg.WINDOW_HEIGHT), 1)
            pygame.draw.line(self._screen, (60, 60, 80), (0, i*self._cell_h), (cfg.WINDOW_WIDTH, i*self._cell_h), 1)

        pygame.display.flip()

    def _draw_mini_game(self, env, cx, cy, w, h, idx):
        """绘制单个小环境"""
        # 背景
        pygame.draw.rect(self._screen, (40, 40, 50), (cx, cy, w, h))
        # 地面
        gy = int(cy + h * 0.82)
        pygame.draw.line(self._screen, (80, 70, 60), (cx, gy), (cx+w, gy), 2)

        # 绘制玩家
        for p, color in [(env.player1, cfg.PLAYER1_COLOR), (env.player2, cfg.PLAYER2_COLOR)]:
            if not p: continue
            # 缩放坐标
            sx = cx + p.get_position().x * w / cfg.WINDOW_WIDTH
            sy = cy + p.get_position().y * h / cfg.WINDOW_HEIGHT
            # 躯干
            try:
                ta = p.bodies['torso'].angle
                tw = cfg.TORSO_WIDTH * w / cfg.WINDOW_WIDTH
                th = cfg.TORSO_HEIGHT * h / cfg.WINDOW_HEIGHT
                rect = pygame.Rect(sx-tw/2, sy-th/2, tw, th)
                points = [
                    (rect.centerx - tw/2*math.cos(ta) + th/2*math.sin(ta),
                     rect.centery - tw/2*math.sin(ta) - th/2*math.cos(ta)),
                ]
                pygame.draw.rect(self._screen, color, rect, 0, 2)
            except: pass
            # 头
            try:
                hx = cx + p.bodies['head'].position.x * w / cfg.WINDOW_WIDTH
                hy = cy + p.bodies['head'].position.y * h / cfg.WINDOW_HEIGHT
                hr = cfg.HEAD_RADIUS * min(w, h) / 700
                pygame.draw.circle(self._screen, cfg.HEAD_COLOR, (int(hx), int(hy)), max(2, int(hr)))
            except: pass
            # 剑
            if p.sword_body:
                sx2 = cx + p.get_sword_tip_position().x * w / cfg.WINDOW_WIDTH
                sy2 = cy + p.get_sword_tip_position().y * h / cfg.WINDOW_HEIGHT
                pygame.draw.line(self._screen, cfg.SWORD_COLOR, (int(sx), int(sy)), (int(sx2), int(sy2)), max(1, int(cfg.SWORD_WIDTH*0.3)))

        # 血条
        hp_pct1 = env.player1.health / cfg.MAX_HEALTH if env.player1 else 0
        hp_pct2 = env.player2.health / cfg.MAX_HEALTH if env.player2 else 0
        bar_w = w * 0.3
        pygame.draw.rect(self._screen, (60,0,0), (cx+5, cy+5, bar_w, 4))
        pygame.draw.rect(self._screen, (0,200,0), (cx+5, cy+5, bar_w*hp_pct1, 4))
        pygame.draw.rect(self._screen, (60,0,0), (cx+w-bar_w-5, cy+5, bar_w, 4))
        pygame.draw.rect(self._screen, (200,0,0), (cx+w-bar_w-5, cy+5, bar_w*hp_pct2, 4))
        # 编号
        t = self._font.render(str(idx+1), True, (150,150,150))
        self._screen.blit(t, (cx+w-20, cy+h-18))

    def train(self):
        old_handler = signal.signal(signal.SIGINT, self._signal_handler)
        self._init_display()

        print(f"16宫格并行训练 ({self.num_envs}环境) {self.device.upper()}")
        try:
            for ep in range(1, self.total_episodes + 1):
                if self._interrupted: break

                # 重置所有环境
                for env in self.envs:
                    env.reset()

                total_reward = 0
                step = 0
                for step in range(300):
                    if self._interrupted: break

                    for idx, env in enumerate(self.envs):
                        if env.game_over: continue
                        # Agent1选择动作
                        s1 = env.get_state_for_agent(1)
                        a1 = self.agent.select_action(s1) if s1 else 0
                        # 对手随机
                        a2 = self.opponents[idx].select_action(env.get_state_for_agent(2) or [0]*24) if env.get_state_for_agent(2) else 0
                        # 执行
                        reward = env.step(
                            self.agent.get_action_tuple(a1),
                            self.opponents[idx].get_action_tuple(a2)
                        )
                        # 存储经验
                        ns1 = env.get_state_for_agent(1)
                        if ns1:
                            self.agent.memory.push(s1, a1, reward['player1'], ns1, float(env.game_over))
                            total_reward += reward['player1']

                    # 学习
                    if step % cfg.TRAIN_INTERVAL == 0:
                        loss = self.agent.learn()

                    # 渲染
                    if step % 5 == 0:
                        self._render_grid(ep, step, total_reward)
                        # 处理关闭事件
                        for e in pygame.event.get():
                            if e.type == pygame.QUIT:
                                self._interrupted = True
                                break

                # 保存
                if ep % 50 == 0 or ep == self.total_episodes:
                    self.agent.save(self.model_path)
                    print(f"  Ep {ep}: saved model, avg reward {total_reward/max(1,self.num_envs):.1f}")

        finally:
            self.agent.save(self.model_path)
            pygame.quit()
            signal.signal(signal.SIGINT, old_handler)


def main():
    """训练入口"""
    import argparse

    parser = argparse.ArgumentParser(description='火柴人击剑格斗 - DQN训练')
    parser.add_argument('--episodes', type=int, default=500, help='训练回合数')
    parser.add_argument('--render', action='store_true', help='单窗口渲染')
    parser.add_argument('--multi', action='store_true', help='9宫格并行训练')
    parser.add_argument('--model-dir', type=str, default='models', help='模型保存目录')
    args = parser.parse_args()

    if args.multi:
        trainer = MultiTrainer(total_episodes=args.episodes, model_dir=args.model_dir)
        trainer.train()
    else:
        trainer = Trainer(render=args.render, total_episodes=args.episodes, model_dir=args.model_dir)
        trainer.train()


if __name__ == '__main__':
    main()
