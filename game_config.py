"""
游戏配置文件 - 火柴人击剑格斗 (Bloody Bastards 风格)
"""
import pymunk

# 屏幕设置
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 700
FPS = 60
GROUND_Y = SCREEN_HEIGHT - 50

# 物理世界设置
GRAVITY = (0, 900)
PHYSICS_STEPS = 3  # 每帧物理子步数 (训练时3步够用)

# 火柴人尺寸
STICKMAN = {
    'head_radius': 15,
    'torso_width': 30,
    'torso_height': 50,
    'upper_arm_length': 25,
    'forearm_length': 25,
    'hand_radius': 6,
    'thigh_length': 28,
    'shin_length': 28,
    'foot_size': 8,
    'shoulder_width': 20,
}

# 剑
SWORD_LENGTH = 55
SWORD_WIDTH = 5
SWORD_HANDLE = 10

# 物理参数
BODY_PART_MASS = {
    'head': 3.0,
    'torso': 8.0,
    'upper_arm': 2.5,
    'forearm': 2.0,
    'hand': 1.0,
    'thigh': 3.5,
    'shin': 2.5,
}

# 关节角度限制 (弧度)
JOINT_LIMITS = {
    'shoulder': (-2.8, 2.8),
    'elbow': (-2.5, 0.0),
    'hip': (-1.5, 1.5),
    'knee': (-2.3, 0.0),
    'neck': (-0.5, 0.5),
}

# 颜色 (R, G, B)
COLORS = {
    'player1_body': (70, 130, 180),    # 钢蓝
    'player1_head': (100, 160, 210),
    'player1_sword': (192, 192, 192),
    'player1_armor': (60, 100, 140),
    'player2_body': (180, 70, 70),     # 暗红
    'player2_head': (210, 100, 100),
    'player2_sword': (218, 165, 32),
    'player2_armor': (140, 50, 50),
    'background': (30, 30, 40),
    'ground': (50, 50, 60),
    'health_bar': (50, 200, 50),
    'health_bg': (80, 30, 30),
    'ui_text': (255, 255, 255),
}

# 战斗参数
FIGHT = {
    'max_health': 100,
    'sword_hit_damage': 12,
    'sword_hit_velocity_threshold': 100,
    'hit_impulse_multiplier': 0.3,
    'knockback_force': 600,
    'stun_duration': 0.3,  # 秒
    'block_damage_reduction': 0.7,
    'attack_cooldown': 0.3,
    'move_speed': 250,
    'dash_speed': 400,
}

# 碰撞类型
COLLISION_TYPES = {
    'sword': 1,
    'body': 2,
    'head': 3,
    'ground': 4,
}

# RL 训练参数
RL = {
    'state_dim': 29,
    'action_dim': 7,
    'hidden_dim': 128,
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'batch_size': 128,
    'memory_size': 50000,
    'epsilon_start': 0.9,
    'epsilon_end': 0.05,
    'epsilon_decay': 5000,
    'tau': 0.005,
    'target_update': 100,
    'max_episodes': 2000,
    'max_steps_per_episode': 500,
}

# 动作定义
# 动作索引-功能映射 (7个离散动作)
NUM_ACTIONS = 7

# 动作映射
ACTION_MAP = {
    0: {'attack': 0, 'block': False, 'move': 0},         # idle
    1: {'attack': 1, 'block': False, 'move': 0},         # 高右斩
    2: {'attack': -1, 'block': False, 'move': 0},        # 高左斩
    3: {'attack': 0, 'block': True, 'move': 0},          # 格挡
    4: {'attack': 0, 'block': False, 'move': 1},         # 前进
    5: {'attack': 0, 'block': False, 'move': -1},        # 后退
    6: {'attack': 2, 'block': False, 'move': 0},         # 下斩
}
