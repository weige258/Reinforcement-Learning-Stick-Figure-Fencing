"""
训练脚本 - 自我对抗训练DQN智能体
使用自对弈课程学习（参考Pro-Level Fighting Game AI论文方法）
"""
import random as _rnd
import time
import os
import sys
import math
import signal

import numpy as np
import pygame
import torch
from pymunk import Vec2d

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
        # agent2 定期同步agent1权重(真正的自对弈)
        self.self_play.agent2 = DQNAgent(player_id=2)
        self._sync_opponent()

        # 训练统计
        self.best_reward = float('-inf')
        self.loss_history = []
        self.reward_history = []
        self._interrupted = False
        self._sync_counter = 0

    def _sync_opponent(self):
        """自对弈核心: 将agent1权重复制给agent2, 制造真正的对手"""
        self.self_play.agent2.policy_net.load_state_dict(
            self.self_play.agent1.policy_net.state_dict())
        self.self_play.agent2.target_net.load_state_dict(
            self.self_play.agent1.target_net.state_dict())
        self._sync_counter += 1
        print(f"  [自对弈] agent2已同步agent1权重 (第{self._sync_counter}次)")

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
                # 每50回合同步agent2权重(真正的自对弈)
                if episode % 50 == 0 and episode > 0:
                    self._sync_opponent()

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
                                self._save_latest()
                                print(f"\n[ESC] 训练中断, 模型已保存: {self.model_path}")
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

class MultiTrainer9:
    """3×3=9环境并行训练器（取代旧的4×4 MultiTrainer）

    优化:
    1. 批量张量推理: 9个状态合成(9,24)tensor, 单次前向传播
    2. 经验共享: 全部环境共用一个ReplayMemory, 9倍经验/步
    3. 真自对弈: opponent定期同步agent1权重
    4. 9×累积学习: 每步收集9条经验后学习
    """

    def __init__(self, total_episodes=1000, model_dir='models'):
        self.total_episodes = total_episodes
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, 'agent_p1_final.pth')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 3×3 = 9个并行环境, 各环境中两个玩家独立随机位置
        self.grid_size = 3
        self.num_envs = 9
        self.envs = [FencingGame(render=False) for _ in range(self.num_envs)]
        for env in self.envs:
            env.create_players()
        self._randomize_all_positions()  # 玩家各自独立随机

        # 共享DQN智能体(CUDA加速)
        self.agent = DQNAgent.load_or_create(self.model_path, player_id=1, device=self.device)
        self.agent.training = True

        # 对手池(自对弈: 定期同步)
        self.opponents = [DQNAgent(player_id=2, device=self.device) for _ in range(self.num_envs)]
        self._sync_counter = 0
        self._sync_opponent()

        # 重置全部环境时同时随机化位置
        self._reset_all = self._reset_all_and_randomize

        # 训练状态
        self.best_reward = float('-inf')
        self._interrupted = False
        self._sync_counter = 0
        self._screen = None
        self._font = None

    def _randomize_all_positions(self):
        """随机化全部环境的玩家初始位置(玩家各自独立)"""
        for idx, env in enumerate(self.envs):
            if not env.player1 or not env.player2:
                continue
            # player1 在左侧随机: 100~350
            p1x = _rnd.randint(80, 350)
            p1y = _rnd.randint(cfg.GROUND_Y - 120, cfg.GROUND_Y - 60)
            env.player1.bodies['torso'].position = Vec2d(p1x, p1y)
            # player2 在右侧随机: 850~1100
            p2x = _rnd.randint(850, 1120)
            p2y = _rnd.randint(cfg.GROUND_Y - 120, cfg.GROUND_Y - 60)
            env.player2.bodies['torso'].position = Vec2d(p2x, p2y)

    def _reset_all_and_randomize(self):
        """重置全部环境 + 随机化位置"""
        for env in self.envs:
            env.reset()
        self._randomize_all_positions()

    def _init_display(self):
        """初始化训练窗口(从菜单切换过来, 直接resize窗口)"""
        w = min(cfg.WINDOW_WIDTH, 900)
        h = min(cfg.WINDOW_HEIGHT, 700)
        self._screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(f"3x3并行训练 - {self.device.upper()} - {self.num_envs}env")
        self._font = pygame.font.Font(None, 13)
        self._cell_w = w // 3
        self._cell_h = h // 3
        # 立即显示一帧, 避免黑屏
        self._screen.fill((15, 15, 25))
        pygame.display.flip()

    def _signal_handler(self, sig, frame):
        self._interrupted = True

    def _sync_opponent(self):
        """自对弈: 同步agent1权重到所有对手"""
        state_dict = self.agent.policy_net.state_dict()
        for opp in self.opponents:
            opp.policy_net.load_state_dict(state_dict)
        self._sync_counter += 1
        print(f"  [自对弈] 已同步9个对手权重 (第{self._sync_counter}次)")

    def train(self):
        """主训练循环 (9环境并行)"""
        old_handler = signal.signal(signal.SIGINT, self._signal_handler)
        self._init_display()

        try:
            for ep in range(1, self.total_episodes + 1):
                if self._interrupted: break

                # 自对弈同步
                if ep % 50 == 0:
                    self._sync_opponent()

                # 重置全部环境(含随机化位置)
                self._reset_all()

                total_reward = 0
                step = 0
                active_mask = [True] * self.num_envs

                for step in range(300):
                    if self._interrupted: break

                    # ==== 批量收集状态 ====
                    states = []
                    active_indices = []
                    for idx in range(self.num_envs):
                        if not active_mask[idx] or self.envs[idx].game_over:
                            continue
                        s = self.envs[idx].get_state_for_agent(1)
                        if s is not None:
                            states.append(s)
                            active_indices.append(idx)
                        else:
                            active_mask[idx] = False

                    if not active_indices:
                        break

                    # ==== 批量推理: 9状态→1张量→1前向→9动作 ====
                    state_tensor = torch.FloatTensor(np.array(states)).to(self.device)
                    with torch.no_grad():
                        q_vals = self.agent.policy_net(state_tensor)
                        actions_idx = q_vals.max(1)[1].cpu().numpy()

                    # 对手批量推理(或随机)
                    opp_states = []
                    for idx in active_indices:
                        s2 = self.envs[idx].get_state_for_agent(2)
                        opp_states.append(s2 if s2 is not None else [0.0]*cfg.STATE_DIM)
                    opp_tensor = torch.FloatTensor(np.array(opp_states)).to(self.device)
                    with torch.no_grad():
                        opp_q = self.opponents[0].policy_net(opp_tensor)
                        opp_actions = opp_q.max(1)[1].cpu().numpy()

                    # ==== 执行动作 + 收集经验 ====
                    for i, idx in enumerate(active_indices):
                        env = self.envs[idx]
                        a1 = self.agent.get_action_tuple(int(actions_idx[i]))
                        a2 = self.opponents[idx].get_action_tuple(int(opp_actions[i]))
                        reward = env.step(a1, a2)

                        ns1 = env.get_state_for_agent(1)
                        if ns1 is not None:
                            self.agent.memory.push(
                                states[i], int(actions_idx[i]),
                                reward['player1'], ns1,
                                float(env.game_over)
                            )
                            total_reward += reward['player1']

                        if env.game_over:
                            active_mask[idx] = False

                    # ==== 学习(每次9条经验后) ====
                    if step % max(1, cfg.TRAIN_INTERVAL) == 0:
                        loss = self.agent.learn()

                    # ==== 渲染 ====
                    if step % 3 == 0:
                        self._render_grid(ep, step, total_reward)
                        for e in pygame.event.get():
                            if e.type == pygame.QUIT:
                                self._interrupted = True; break
                            elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                                self._interrupted = True  # ESC: 保存并返回菜单
                                self.agent.save(self.model_path)
                                print(f"\n[ESC] 训练中断, 模型已保存: {self.model_path}")

                # 回合统计
                avg_r = total_reward / max(1, self.num_envs)
                if avg_r > self.best_reward:
                    self.best_reward = avg_r

                if ep % 10 == 0 or ep == self.total_episodes:
                    self.agent.save(self.model_path)
                    print(
                        f"Ep {ep:4d}/{self.total_episodes} | "
                        f"奖励 {avg_r:+7.1f} | "
                        f"ε {self.agent.epsilon:.3f} | "
                        f"记忆 {len(self.agent.memory):5d} | "
                        f"步 {step:3d}"
                    )

        finally:
            self.agent.save(self.model_path)
            print(f"\n训练结束! 模型已保存: {self.model_path}")
            pygame.quit()
            signal.signal(signal.SIGINT, old_handler)

    def _render_grid(self, episode, step, total_reward):
        """3×3宫格渲染"""
        if not self._screen:
            return
        w, h = self._screen.get_size()
        self._screen.fill((15, 15, 25))
        cw, ch = w // 3, h // 3

        for idx, env in enumerate(self.envs):
            row, col = idx // 3, idx % 3
            cx, cy = col * cw, row * ch
            self._draw_mini(env, cx, cy, cw, ch, idx + 1)

        avg_r = total_reward / max(1, self.num_envs)
        info = (
            f"Ep {episode} | R {avg_r:.1f} | ε {self.agent.epsilon:.3f} | "
            f"buf {len(self.agent.memory)} | {self.device.upper()}"
        )
        txt = self._font.render(info, True, (180, 220, 180))
        self._screen.blit(txt, (6, 2))

        for i in range(1, 3):
            pygame.draw.line(self._screen, (50, 50, 70), (i * cw, 0), (i * cw, h), 1)
            pygame.draw.line(self._screen, (50, 50, 70), (0, i * ch), (w, i * ch), 1)
        pygame.display.flip()

    def _draw_mini(self, env, cx, cy, cw, ch, label):
        """绘制单个小环境 (完整火柴人+武器)"""
        pygame.draw.rect(self._screen, (40, 40, 50), (cx, cy, cw, ch))
        gy = int(cy + ch * 0.82)
        pygame.draw.line(self._screen, (80, 70, 60), (cx, gy), (cx + cw, gy), 2)

        sx = cw / cfg.WINDOW_WIDTH
        sy = ch / cfg.WINDOW_HEIGHT

        def _wp(bp):
            return (cx + bp.x * sx, cy + bp.y * sy)

        def _ds(p, col):
            """绘制单个人物 (包含内部_dw闭包, 正确捕获p)"""
            if not p: return
            try:
                tx, ty = _wp(p.get_position())
                tw = cfg.TORSO_WIDTH * sx; th = cfg.TORSO_HEIGHT * sy
                r = pygame.Rect(tx - tw / 2, ty - th / 2, tw, th)
                pygame.draw.rect(self._screen, col, r, 0, max(1, int(2 * sx)))
                hx, hy = _wp(p.bodies['head'].position)
                hr = max(3, cfg.HEAD_RADIUS * sx)
                pygame.draw.circle(self._screen, cfg.HEAD_COLOR, (int(hx), int(hy)), int(hr))

                # 内部闭包: 正确捕获p
                def _dw(ba, bb, w, col):
                    if ba in p.bodies and bb in p.bodies:
                        x1, y1 = _wp(p.bodies[ba].position)
                        x2, y2 = _wp(p.bodies[bb].position)
                        pygame.draw.line(self._screen, col,
                            (int(x1), int(y1)), (int(x2), int(y2)), max(1, int(w * sx)))

                # 腿
                for sd in ['left', 'right']:
                    _dw('torso', f'{sd}_upper_leg', 3, col)
                    _dw(f'{sd}_upper_leg', f'{sd}_lower_leg', 3, col)

                # 右臂(持剑)
                if 'right_upper_arm' in p.bodies:
                    rx, ry = _wp(p.bodies['right_upper_arm'].position)
                    rsx, rsy = tx + 12 * sx, ty - 16 * sy
                    pygame.draw.line(self._screen, col, (int(rsx), int(rsy)),
                        (int(rx), int(ry)), max(1, int(3 * sx)))
                if 'right_lower_arm' in p.bodies and 'right_upper_arm' in p.bodies:
                    _dw('right_upper_arm', 'right_lower_arm', 2, col)

                # 左臂(持盾)
                if 'left_upper_arm' in p.bodies:
                    lx, ly = _wp(p.bodies['left_upper_arm'].position)
                    lsx, lsy = tx - 12 * sx, ty - 16 * sy
                    pygame.draw.line(self._screen, col, (int(lsx), int(lsy)),
                        (int(lx), int(ly)), max(1, int(3 * sx)))
                if 'left_lower_arm' in p.bodies and 'left_upper_arm' in p.bodies:
                    _dw('left_upper_arm', 'left_lower_arm', 2, col)

                # 剑
                if p.sword_body and 'right_lower_arm' in p.bodies:
                    wx, wy = _wp(p.bodies['right_lower_arm'].position)
                    tx2, ty2 = _wp(p.get_sword_tip_position())
                    pygame.draw.line(self._screen, cfg.SWORD_COLOR,
                        (int(wx), int(wy)), (int(tx2), int(ty2)), max(1, int(2 * sx)))

                # 盾牌
                if p.shield_body:
                    shx, shy = _wp(p.shield_body.position)
                    sw = cfg.SHIELD_WIDTH * sx; sh = cfg.SHIELD_HEIGHT * sy
                    sr = pygame.Rect(shx - sw / 2, shy - sh / 2, sw, sh)
                    pygame.draw.rect(self._screen, cfg.SHIELD_COLOR, sr, 0, max(1, int(1 * sx)))
            except Exception as e:
                if str(e):  # 只打印一次, 不刷屏
                    pass

        _ds(env.player1, cfg.PLAYER1_COLOR)
        _ds(env.player2, cfg.PLAYER2_COLOR)

        hp1 = env.player1.health / cfg.MAX_HEALTH if env.player1 and hasattr(env.player1, 'health') else 0
        hp2 = env.player2.health / cfg.MAX_HEALTH if env.player2 and hasattr(env.player2, 'health') else 0
        bw = cw * 0.35
        pygame.draw.rect(self._screen, (60, 0, 0), (cx + 4, cy + 4, bw, 3))
        pygame.draw.rect(self._screen, (0, 200, 0), (cx + 4, cy + 4, bw * hp1, 3))
        pygame.draw.rect(self._screen, (60, 0, 0), (cx + cw - bw - 4, cy + 4, bw, 3))
        pygame.draw.rect(self._screen, (200, 0, 0), (cx + cw - bw - 4, cy + 4, bw * hp2, 3))
        t = self._font.render(str(label), True, (150, 150, 150))
        self._screen.blit(t, (cx + 4, cy + ch - 18))


def main():
    """训练入口"""
    import argparse

    parser = argparse.ArgumentParser(description='火柴人击剑格斗 - DQN训练')
    parser.add_argument('--episodes', type=int, default=500, help='训练回合数')
    parser.add_argument('--render', action='store_true', help='单窗口渲染')
    parser.add_argument('--multi', action='store_true', help='3×3=9环境并行训练(推荐)')
    parser.add_argument('--model-dir', type=str, default='models', help='模型保存目录')
    args = parser.parse_args()

    if args.multi:
        trainer = MultiTrainer9(total_episodes=args.episodes, model_dir=args.model_dir)
        trainer.train()
    else:
        trainer = Trainer(render=args.render, total_episodes=args.episodes, model_dir=args.model_dir)
        trainer.train()


if __name__ == '__main__':
    main()
