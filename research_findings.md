# 网络调研结果

## Bloody Bastards (该死的混蛋) 游戏机制分析

### 核心玩法
- **物理驱动战斗**: 基于 ragdoll（布娃娃）物理引擎的实时对战游戏
- **武器系统**: 多种武器（剑、锤、斧、矛、盾牌等），每种武器有不同重量、长度、伤害
- **肢体控制**: 玩家通过鼠标拖拽控制角色的手臂/武器挥动方向
- **部位伤害**: 身体各部位独立血量，头部、躯干、四肢可被砍断
- ** ragdoll 物理**: 受到攻击后角色会根据物理碰撞产生真实反应
- **战斗策略**: 需要控制武器挥动轨迹、格挡、闪避

### 关键游戏特性
1. **鼠标控制**: 拖动鼠标控制武器挥动方向和力度
2. **物理碰撞**: 武器与对手身体的碰撞基于真实物理
3. **装备系统**: 不同武器有不同属性，可更换装备
4. **血量系统**: 身体各部位独立血量
5. ** ragdoll 死亡**: 死亡时角色变成布娃娃物理

## 最新强化学习研究

### 1. MAAIP (2023)
- **论文**: Multi-Agent Adversarial Interaction Priors for imitation from fighting demonstrations for physics-based characters
- **作者**: Mohamed Younes et al. (SCA'23)
- **核心**: 多智能体生成对抗模仿学习，用于物理角色的格斗动作模仿
- **方法**: 使用两个非结构化数据集（单人动作 + 双人交互）训练控制策略
- **应用**: 拳击和全身体术格斗风格
- **链接**: https://arxiv.org/abs/2311.02502

### 2. Diversity-based DRL for Fighting Game AI (2022)
- **论文**: Diversity-based Deep Reinforcement Learning Towards Multidimensional Difficulty for Fighting Game AI
- **作者**: Emily Halina, Matthew Guzdial
- **核心**: 基于多样性的深度强化学习，生成不同策略的AI对手
- **方法**: 在奖励函数中加入多样性奖励，训练多个不同风格的智能体
- **链接**: https://arxiv.org/abs/2211.02759

### 3. Pro-Level Fighting Game AI (2019)
- **论文**: Creating Pro-Level AI for a Real-Time Fighting Game Using Deep Reinforcement Learning
- **作者**: Inseok Oh et al.
- **核心**: 在商业格斗游戏"Blade & Soul"中战胜职业选手（胜率62%）
- **方法**: 
  - 自对弈课程学习 (Self-play curriculum)
  - 数据跳跃技术 (Data skipping)
  - 奖励塑形 (Reward shaping) 创造不同风格的智能体
- **链接**: https://arxiv.org/abs/1904.03821

### 4. 其他相关研究
- **NVIDIA Physics-Based Character Control**: Isaac Gym 环境中的物理角色控制
- **DeepMimic**: 物理角色的动作模仿学习 (OpenAI)
- **ASE (Adversarial Skill Embeddings)**: 学习多样化物理角色技能

## 技术方案选择

本项目采用:
1. **PyMunk** - 2D物理引擎（类似Bloody Bastards的2D简化版）
2. **PyTorch DQN** - 深度Q网络（参考Pro-Level Fighting Game AI的自对弈方法）
3. **Pygame** - 游戏渲染和输入处理
4. **自我对抗训练** - 智能体与自己下棋对弈（Self-play）
5. **Epsilon-greedy** - 探索与利用平衡
