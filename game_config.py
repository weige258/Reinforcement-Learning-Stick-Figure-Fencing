"""
游戏配置文件 - 基于Bloody Bastards风格的火柴人击剑格斗游戏
"""

# ============ 窗口设置 ============
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 700
FPS = 60
BACKGROUND_COLOR = (50, 50, 60)
GROUND_COLOR = (80, 70, 60)
SKY_COLOR = (60, 65, 75)

# ============ 物理世界设置 ============
GRAVITY = (0.0, 600.0)  # 重力
GROUND_Y = 620
GROUND_THICKNESS = 30
DAMPING = 0.99  # 空间阻尼

# ============ 火柴人身体参数(双段四肢) ============
HEAD_RADIUS = 14
TORSO_WIDTH = 28
TORSO_HEIGHT = 50
UPPER_ARM_LENGTH = 30
LOWER_ARM_LENGTH = 30
UPPER_LEG_LENGTH = 35
LOWER_LEG_LENGTH = 35

# 身体部位质量
HEAD_MASS = 1.5
TORSO_MASS = 4.0
UPPER_ARM_MASS = 1.2
LOWER_ARM_MASS = 1.0
UPPER_LEG_MASS = 1.8
LOWER_LEG_MASS = 1.5

# ============ 武器参数 ============
SWORD_LENGTH = 55
SWORD_WIDTH = 5
SWORD_MASS = 2.0
SWORD_BASE_DAMAGE = 18    # 基础伤害, ×刀速/50
SWORD_MAX_DAMAGE = 100    # 单次最大伤害

SHIELD_WIDTH = 30
SHIELD_HEIGHT = 45
SHIELD_MASS = 3.0
SHIELD_BLOCK_DAMAGE_REDUCTION = 0.7

# ============ 战斗参数 ============
MAX_HEALTH = 100
HEAD_DAMAGE_MULTIPLIER = 2.0
TORSO_DAMAGE_MULTIPLIER = 1.0
LIMB_DAMAGE_MULTIPLIER = 0.6
HIT_FORCE_MULTIPLIER = 200.0
ATTACK_COOLDOWN = 0.3  # 攻击冷却时间（秒）
BLOCK_COOLDOWN = 0.1

# 攻击速度阈值 - 挥动速度达到此值才造成伤害
ATTACK_SPEED_THRESHOLD = 30.0   # 降低阈值，使攻击更容易命中
ATTACK_TORQUE = 200000.0         # 攻击旋转力矩
ATTACK_IMPULSE = 500.0           # 攻击脉冲力

# ============ 颜色配置 ============
PLAYER1_COLOR = (100, 180, 255)  # 蓝色系
PLAYER2_COLOR = (255, 100, 100)  # 红色系
SWORD_COLOR = (200, 200, 210)
SHIELD_COLOR = (160, 120, 80)
HEAD_COLOR = (255, 220, 180)
BLOOD_COLOR = (180, 30, 30)

# ============ RL训练参数 ============
STATE_DIM = 24  # 状态空间维度
ACTION_DIM = 9  # 动作空间维度

# 动作映射: [move_x, move_y, jump, attack, block]
# 简化: 去掉冗余的上下左右组合, 保留核心动作
ACTIONS = [
    (0, 0, 0, 0, 0),  # 0: 不动
    (-1, 0, 0, 0, 0), # 1: 左移
    (1, 0, 0, 0, 0),  # 2: 右移
    (0, 0, 1, 0, 0),  # 3: 跳跃
    (0, 0, 0, 1, 0),  # 4: 挥剑
    (0, 0, 0, 0, 1),  # 5: 举盾
    (1, 0, 0, 1, 0),  # 6: 右移+挥剑
    (-1, 0, 0, 1, 0), # 7: 左移+挥剑
    (0, 0, 0, 1, 1),  # 8: 挥剑+举盾
]
ACTION_DIM = len(ACTIONS)  # 自动计算

# DQN超参数
LEARNING_RATE = 0.0003
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.998  # 500回合后: 0.998^500=0.37, 更平缓衰减
BATCH_SIZE = 64
MEMORY_SIZE = 100000
TARGET_UPDATE_INTERVAL = 100
TRAIN_INTERVAL = 4

# 网络结构
HIDDEN_DIM_1 = 256
HIDDEN_DIM_2 = 256

# 奖励设计
# 奖励设计(统一缩放到[-1,1]区间)
REWARD_MOVE_TOWARD = 0.02       # 向对手移动
REWARD_HIT = 0.5               # 击中对手
REWARD_HIT_HEAD = 1.0          # 击中头部(已用HEAD_DAMAGE_MULTIPLIER)
REWARD_BLOCK = 0.2             # 成功格挡
REWARD_DAMAGE_TAKEN = 0.5      # 受到伤害惩罚(正数, 代码中用abs)
REWARD_DEATH = -1.0            # 死亡
REWARD_KILL = 1.0              # 击杀对手
REWARD_STEP_PENALTY = -0.001   # 每步惩罚
REWARD_SWORD_SPEED = 0.02      # 剑挥动速度奖励
