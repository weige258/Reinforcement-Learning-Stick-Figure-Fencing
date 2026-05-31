"""
深度Q学习(DQN)智能体 - 用于火柴人击剑格斗
基于 PyTorch 实现，包含经验回放和目标网络
"""
import random
import math
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from game_config import RL

# 经验回放单元
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    """经验回放缓冲区"""
    
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        """保存一个转移"""
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        """随机采样一批"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """深度Q网络"""
    
    def __init__(self, n_observations, n_actions, hidden_dim=256):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, n_actions)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, state_dim, action_dim, device=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 设备
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else
                "cpu"
            )
        else:
            self.device = device
        
        # print(f"DQN Agent using device: {self.device}")  # 静默初始化
        
        # Q网络
        self.policy_net = DQN(state_dim, action_dim, RL['hidden_dim']).to(self.device)
        self.target_net = DQN(state_dim, action_dim, RL['hidden_dim']).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 优化器
        self.optimizer = optim.AdamW(self.policy_net.parameters(), 
                                     lr=RL['learning_rate'], amsgrad=True)
        
        # 经验回放
        self.memory = ReplayMemory(RL['memory_size'])
        
        # 训练步数
        self.steps_done = 0
        
        # 损失记录
        self.loss_history = []
    
    def select_action(self, state, eval_mode=False):
        """选择动作 (epsilon-greedy)"""
        if eval_mode:
            # 评估模式 - 纯贪心
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, 
                                           device=self.device).unsqueeze(0)
                return self.policy_net(state_tensor).max(1).indices.view(1, 1).item()
        
        # 训练模式 - epsilon-greedy
        sample = random.random()
        eps_threshold = RL['epsilon_end'] + (RL['epsilon_start'] - RL['epsilon_end']) * \
                        math.exp(-1. * self.steps_done / RL['epsilon_decay'])
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, 
                                           device=self.device).unsqueeze(0)
                return self.policy_net(state_tensor).max(1).indices.view(1, 1).item()
        else:
            return random.randrange(self.action_dim)
    
    def optimize_model(self):
        """优化模型 - 单步优化"""
        if len(self.memory) < RL['batch_size']:
            return 0
        
        transitions = self.memory.sample(RL['batch_size'])
        batch = Transition(*zip(*transitions))
        
        # 计算非终止状态的掩码
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device, dtype=torch.bool
        )
        non_final_next_states = torch.tensor(
            [s for s in batch.next_state if s is not None],
            dtype=torch.float32, device=self.device
        )
        
        # 拼接批数据
        state_batch = torch.tensor(batch.state, dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        
        # 计算 Q(s_t, a_t)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # 计算 V(s_{t+1}) = max_a Q(s_{t+1}, a)
        next_state_values = torch.zeros(RL['batch_size'], device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        
        # 计算期望 Q 值
        expected_state_action_values = (next_state_values * RL['gamma']) + reward_batch
        
        # Huber 损失
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪 (典型值5)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 5)
        self.optimizer.step()
        
        return loss.item()
    
    def soft_update_target(self):
        """软更新目标网络"""
        tau = RL['tau']
        target_dict = self.target_net.state_dict()
        policy_dict = self.policy_net.state_dict()
        for key in policy_dict:
            target_dict[key] = policy_dict[key] * tau + target_dict[key] * (1 - tau)
        self.target_net.load_state_dict(target_dict)
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'loss_history': self.loss_history,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint.get('steps_done', 0)
        self.loss_history = checkpoint.get('loss_history', [])
        print(f"Model loaded from {filepath}")
    
    def store_transition(self, state, action, next_state, reward):
        """存储经验"""
        self.memory.push(state, action, next_state, reward)
