"""
DQN (Deep Q-Network) 智能体 - 使用PyTorch实现
参考: "Creating Pro-Level AI for a Real-Time Fighting Game Using Deep Reinforcement Learning" (2019)
      "Diversity-based Deep Reinforcement Learning for Fighting Game AI" (2022)
"""
import random
import math
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import game_config as cfg


class DQN(nn.Module):
    """深度Q网络 - 用于格斗游戏的决策网络"""

    def __init__(self, state_dim=cfg.STATE_DIM, action_dim=cfg.ACTION_DIM):
        super(DQN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, cfg.HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Linear(cfg.HIDDEN_DIM_1, cfg.HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Linear(cfg.HIDDEN_DIM_2, action_dim)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重（使用Kaiming初始化）"""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """前向传播"""
        return self.net(x)


class ReplayMemory:
    """经验回放缓冲区"""

    def __init__(self, capacity=cfg.MEMORY_SIZE):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """随机采样一批经验"""
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)).unsqueeze(1),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones))
        )

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    """DQN智能体 - 用于训练和对战"""

    def __init__(self, state_dim=cfg.STATE_DIM, action_dim=cfg.ACTION_DIM,
                 player_id=1, device='cpu'):
        self.player_id = player_id
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 主网络和目标网络
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)

        # 经验回放
        self.memory = ReplayMemory()

        # 训练参数
        self.epsilon = cfg.EPSILON_START
        self.steps_done = 0
        self.training = True

        # 多样性参数（参考 Diversity-based DRL 论文）
        self.style_vector = None  # 可选的风格向量
        self.diversity_reward_weight = 0.0

    def select_action(self, state, eval_mode=False):
        """选择动作 - 使用epsilon-greedy策略

        参考: Pro-Level Fighting Game AI 的探索策略
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # epsilon-greedy
        if not eval_mode and self.training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            q_values = self.policy_net(state)
            return q_values.max(1)[1].item()

    def get_action_tuple(self, action_idx):
        """将动作索引转换为游戏动作元组"""
        if action_idx < 0 or action_idx >= len(cfg.ACTIONS):
            action_idx = 0
        return cfg.ACTIONS[action_idx]

    def learn(self):
        """从经验回放中学习"""
        if len(self.memory) < cfg.BATCH_SIZE:
            return None

        # 采样
        states, actions, rewards, next_states, dones = self.memory.sample(cfg.BATCH_SIZE)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 奖励裁剪到合理范围, 防止梯度爆炸
        rewards = torch.clamp(rewards, -10.0, 10.0)

        # 计算当前Q值
        current_q = self.policy_net(states).gather(1, actions)

        # 计算目标Q值（使用Double DQN）
        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + cfg.GAMMA * next_q * (1 - dones.unsqueeze(1))
            # 目标Q值裁剪, 防止Q值发散
            target_q = torch.clamp(target_q, -100.0, 100.0)

        # 计算损失
        loss = nn.SmoothL1Loss()(current_q, target_q)

        # NaN检测: 如果损失是NaN则跳过此步
        if torch.isnan(loss):
            print("  [WARN] NaN loss detected, skipping step")
            self.steps_done += 1
            self.epsilon = max(cfg.EPSILON_END, self.epsilon * cfg.EPSILON_DECAY)
            return 0.0

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)

        # 梯度NaN检测
        valid_grad = True
        for p in self.policy_net.parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                valid_grad = False
                break
        if not valid_grad:
            print("  [WARN] NaN gradient detected, skipping step")
            self.optimizer.zero_grad()
            self.steps_done += 1
            return 0.0

        self.optimizer.step()

        # epsilon按步衰减(温和)
        self.steps_done += 1
        self.epsilon = max(cfg.EPSILON_END, self.epsilon * cfg.EPSILON_DECAY)

        # 更新目标网络
        if self.steps_done % cfg.TARGET_UPDATE_INTERVAL == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.scheduler.step()

        return loss.item()

    def save(self, filepath):
        """保存模型（含配置参数校验）"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }, filepath)
        print(f"模型已保存: {filepath}")

    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', cfg.EPSILON_END)
        self.steps_done = checkpoint.get('steps_done', 0)
        self.policy_net.train()
        print(f"模型已加载: {filepath}")

    @classmethod
    def load_or_create(cls, filepath, state_dim=None, action_dim=None, player_id=1, device='cpu'):
        """加载模型，如果不存在或参数不匹配则创建新模型"""
        import os
        state_dim = state_dim or cfg.STATE_DIM
        action_dim = action_dim or cfg.ACTION_DIM

        if os.path.exists(filepath):
            try:
                # 先加载检查配置是否匹配
                temp = torch.load(filepath, map_location='cpu')
                sd = temp.get('state_dim', -1)
                ad = temp.get('action_dim', -1)
                if sd == state_dim and ad == action_dim:
                    agent = cls(state_dim, action_dim, player_id, device)
                    agent.policy_net.load_state_dict(temp['policy_net_state_dict'])
                    agent.target_net.load_state_dict(temp['target_net_state_dict'])
                    agent.optimizer.load_state_dict(temp['optimizer_state_dict'])
                    agent.epsilon = temp.get('epsilon', cfg.EPSILON_END)
                    agent.steps_done = temp.get('steps_done', 0)
                    agent.policy_net.train()
                    print(f"模型已加载: {filepath}")
                    return agent
                else:
                    print(f"模型参数不匹配(期望{sd}x{ad}, 当前{state_dim}x{action_dim}), 创建新模型")
            except Exception as e:
                print(f"模型加载失败({e}), 创建新模型")
        else:
            print(f"未找到模型: {filepath}, 创建新模型")
        return cls(state_dim, action_dim, player_id, device)

    def set_style(self, style_vector):
        """设置风格向量（用于多样性训练）"""
        self.style_vector = style_vector

    def get_q_values(self, state):
        """获取所有动作的Q值（用于可视化/分析）"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.policy_net(state).cpu().numpy()[0]


class SelfPlayManager:
    """自对弈管理器 - 管理两个智能体的训练

    参考 Pro-Level Fighting Game AI 的自对弈课程学习方法
    """

    def __init__(self, device='cpu'):
        self.agent1 = DQNAgent(player_id=1, device=device)
        self.agent2 = DQNAgent(player_id=2, device=device)

        # 训练统计
        self.episode_rewards = {1: [], 2: []}
        self.episode_lengths = []
        self.win_counts = {1: 0, 2: 0}

        # 课程学习参数
        self.curriculum_stage = 0
        # 不同阶段的配置
        self.curriculum_stages = [
            {'agent2_epsilon': 0.5, 'desc': "简单对手"},
            {'agent2_epsilon': 0.3, 'desc': "中等对手"},
            {'agent2_epsilon': 0.1, 'desc': "强对手"},
            {'agent2_epsilon': 0.05, 'desc': "专家对手"},
        ]

    def update_curriculum(self, episode):
        """更新课程阶段"""
        stage = min(episode // 200, len(self.curriculum_stages) - 1)
        if stage > self.curriculum_stage:
            self.curriculum_stage = stage
            print(f"\n=== 进入课程阶段 {stage+1}: {self.curriculum_stages[stage]['desc']} ===")

    def get_agent2_epsilon(self):
        """获取对手的epsilon值（基于课程）"""
        stage = self.curriculum_stages[self.curriculum_stage]
        return stage['agent2_epsilon']

    def get_stats(self):
        """获取训练统计"""
        recent_rewards_1 = self.episode_rewards[1][-100:] if self.episode_rewards[1] else [0]
        recent_rewards_2 = self.episode_rewards[2][-100:] if self.episode_rewards[2] else [0]

        return {
            'agent1_avg_reward': sum(recent_rewards_1) / len(recent_rewards_1),
            'agent2_avg_reward': sum(recent_rewards_2) / len(recent_rewards_2),
            'win_rate_1': self.win_counts[1] / max(1, sum(self.win_counts.values())),
            'total_episodes': sum(self.win_counts.values()),
            'epsilon_1': self.agent1.epsilon,
            'epsilon_2': self.agent2.epsilon,
            'curriculum_stage': self.curriculum_stage,
        }
