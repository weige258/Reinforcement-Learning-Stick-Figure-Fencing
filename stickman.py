"""
火柴人角色 - 使用 PyMunk 物理引擎实现关节式身体
类似 Bloody Bastards 的 ragdoll 物理风格
"""
import math
import pymunk
from pymunk import Vec2d
from game_config import *

class StickMan:
    """基于PyMunk物理的火柴人角色"""
    
    def __init__(self, space, x, y, facing_right=True, color_scheme='player1'):
        self.space = space
        self.facing = 1 if facing_right else -1
        color_key = 'player1' if color_scheme == 'player1' else 'player2'
        self.body_color = COLORS[f'{color_key}_body']
        self.head_color = COLORS[f'{color_key}_head']
        self.sword_color = COLORS[f'{color_key}_sword']
        self.armor_color = COLORS[f'{color_key}_armor']
        
        # 战斗状态
        self.health = FIGHT['max_health']
        self.max_health = FIGHT['max_health']
        self.attack_cooldown = 0
        self.blocking = False
        self.stun_timer = 0
        self.alive = True
        self.damage_dealt_this_step = 0  # 本步造成的伤害 (用于奖励)
        
        # 存储所有身体部件
        self.bodies = {}      # 物理body
        self.shapes = []      # 所有shape
        self.joints = []      # 所有关节
        self.part_positions = {}  # 记录相对位置
        
        # 碰撞组 - 同组不碰撞 (防止自碰撞)
        self.collision_group = id(self)
        self._move_direction = 0
        
        self._build_body(x, y)
        self._build_joints()
        self._create_sword(x, y)
        
    def _apply_collision_filter(self, shape):
        """应用碰撞过滤 - 防止自碰撞"""
        shape.filter = pymunk.ShapeFilter(group=self.collision_group)
        return shape
    
    def _build_body(self, x, y):
        """构建火柴人身体"""
        s = STICKMAN
        cx, cy = x, y - 20  # 稍微抬高
        
        # === 躯干 (Torso) ===
        torso_body = pymunk.Body(BODY_PART_MASS['torso'], 
                                 pymunk.moment_for_box(BODY_PART_MASS['torso'], 
                                                       (s['torso_width'], s['torso_height'])))
        torso_body.position = cx, cy - s['torso_height']/2
        torso_shape = pymunk.Poly.create_box(torso_body, (s['torso_width'], s['torso_height']))
        torso_shape.elasticity = 0.01
        torso_shape.friction = 0.5
        torso_shape.collision_type = COLLISION_TYPES['body']
        self.bodies['torso'] = torso_body
        self.shapes.append(torso_shape)
        
        # === 头部 (Head) ===
        head_body = pymunk.Body(BODY_PART_MASS['head'],
                                pymunk.moment_for_circle(BODY_PART_MASS['head'], 0, s['head_radius']))
        head_body.position = cx, cy - s['torso_height'] - s['head_radius']
        head_shape = pymunk.Circle(head_body, s['head_radius'])
        head_shape.elasticity = 0.01
        head_shape.friction = 0.5
        head_shape.collision_type = COLLISION_TYPES['head']
        self.bodies['head'] = head_body
        self.shapes.append(head_shape)
        
        # === 右大臂 (Right Upper Arm) ===
        uarm_r_body = pymunk.Body(BODY_PART_MASS['upper_arm'],
                                  pymunk.moment_for_box(BODY_PART_MASS['upper_arm'], 
                                                        (s['upper_arm_length'], 8)))
        shoulder_r_x = cx + s['shoulder_width']/2
        shoulder_r_y = cy - s['torso_height'] * 0.8
        uarm_r_body.position = shoulder_r_x + s['upper_arm_length']/2, shoulder_r_y
        uarm_r_shape = pymunk.Poly.create_box(uarm_r_body, (s['upper_arm_length'], 8))
        uarm_r_shape.elasticity = 0.01
        uarm_r_shape.friction = 0.5
        uarm_r_shape.collision_type = COLLISION_TYPES['body']
        self.bodies['upper_arm_r'] = uarm_r_body
        self.shapes.append(uarm_r_shape)
        
        # === 右前臂 (Right Forearm) ===
        forearm_r_body = pymunk.Body(BODY_PART_MASS['forearm'],
                                     pymunk.moment_for_box(BODY_PART_MASS['forearm'],
                                                           (s['forearm_length'], 7)))
        forearm_r_body.position = shoulder_r_x + s['upper_arm_length'] + s['forearm_length']/2, shoulder_r_y
        forearm_r_shape = pymunk.Poly.create_box(forearm_r_body, (s['forearm_length'], 7))
        forearm_r_shape.elasticity = 0.01
        forearm_r_shape.friction = 0.5
        forearm_r_shape.collision_type = COLLISION_TYPES['body']
        self.bodies['forearm_r'] = forearm_r_body
        self.shapes.append(forearm_r_shape)
        
        # === 右手 (Right Hand) ===
        hand_r_body = pymunk.Body(BODY_PART_MASS['hand'],
                                  pymunk.moment_for_circle(BODY_PART_MASS['hand'], 0, s['hand_radius']))
        hand_r_body.position = shoulder_r_x + s['upper_arm_length'] + s['forearm_length'] + s['hand_radius'], shoulder_r_y
        hand_r_shape = pymunk.Circle(hand_r_body, s['hand_radius'])
        hand_r_shape.elasticity = 0.01
        hand_r_shape.friction = 0.5
        hand_r_shape.collision_type = COLLISION_TYPES['body']
        self.bodies['hand_r'] = hand_r_body
        self.shapes.append(hand_r_shape)
        
        # === 左大臂 (Left Upper Arm) ===
        uarm_l_body = pymunk.Body(BODY_PART_MASS['upper_arm'],
                                  pymunk.moment_for_box(BODY_PART_MASS['upper_arm'],
                                                        (s['upper_arm_length'], 8)))
        shoulder_l_x = cx - s['shoulder_width']/2
        shoulder_l_y = cy - s['torso_height'] * 0.8
        uarm_l_body.position = shoulder_l_x - s['upper_arm_length']/2, shoulder_l_y
        uarm_l_shape = pymunk.Poly.create_box(uarm_l_body, (s['upper_arm_length'], 8))
        uarm_l_shape.elasticity = 0.01
        uarm_l_shape.friction = 0.5
        uarm_l_shape.collision_type = COLLISION_TYPES['body']
        self.bodies['upper_arm_l'] = uarm_l_body
        self.shapes.append(uarm_l_shape)
        
        # === 左前臂 (Left Forearm) ===
        forearm_l_body = pymunk.Body(BODY_PART_MASS['forearm'],
                                     pymunk.moment_for_box(BODY_PART_MASS['forearm'],
                                                           (s['forearm_length'], 7)))
        forearm_l_body.position = shoulder_l_x - s['upper_arm_length'] - s['forearm_length']/2, shoulder_l_y
        forearm_l_shape = pymunk.Poly.create_box(forearm_l_body, (s['forearm_length'], 7))
        forearm_l_shape.elasticity = 0.01
        forearm_l_shape.friction = 0.5
        forearm_l_shape.collision_type = COLLISION_TYPES['body']
        self.bodies['forearm_l'] = forearm_l_body
        self.shapes.append(forearm_l_shape)
        
        # === 左手 (Left Hand) ===
        hand_l_body = pymunk.Body(BODY_PART_MASS['hand'],
                                  pymunk.moment_for_circle(BODY_PART_MASS['hand'], 0, s['hand_radius']))
        hand_l_body.position = shoulder_l_x - s['upper_arm_length'] - s['forearm_length'] - s['hand_radius'], shoulder_l_y
        hand_l_shape = pymunk.Circle(hand_l_body, s['hand_radius'])
        hand_l_shape.elasticity = 0.01
        hand_l_shape.friction = 0.5
        hand_l_shape.collision_type = COLLISION_TYPES['body']
        self.bodies['hand_l'] = hand_l_body
        self.shapes.append(hand_l_shape)
        
        # === 右大腿 (Right Thigh) ===
        thigh_r_body = pymunk.Body(BODY_PART_MASS['thigh'],
                                   pymunk.moment_for_box(BODY_PART_MASS['thigh'],
                                                         (10, s['thigh_length'])))
        hip_r_x = cx + 8
        hip_r_y = cy
        thigh_r_body.position = hip_r_x, hip_r_y + s['thigh_length']/2
        thigh_r_shape = pymunk.Poly.create_box(thigh_r_body, (10, s['thigh_length']))
        thigh_r_shape.elasticity = 0.1
        thigh_r_shape.friction = 0.3
        thigh_r_shape.collision_type = COLLISION_TYPES['body']
        self.bodies['thigh_r'] = thigh_r_body
        self.shapes.append(thigh_r_shape)
        
        # === 右小腿 (Right Shin) ===
        shin_r_body = pymunk.Body(BODY_PART_MASS['shin'],
                                  pymunk.moment_for_box(BODY_PART_MASS['shin'],
                                                        (9, s['shin_length'])))
        shin_r_body.position = hip_r_x, hip_r_y + s['thigh_length'] + s['shin_length']/2
        shin_r_shape = pymunk.Poly.create_box(shin_r_body, (9, s['shin_length']))
        shin_r_shape.elasticity = 0.01
        shin_r_shape.friction = 0.5
        shin_r_shape.collision_type = COLLISION_TYPES['body']
        self.bodies['shin_r'] = shin_r_body
        self.shapes.append(shin_r_shape)
        
        # === 左大腿 (Left Thigh) ===
        thigh_l_body = pymunk.Body(BODY_PART_MASS['thigh'],
                                   pymunk.moment_for_box(BODY_PART_MASS['thigh'],
                                                         (10, s['thigh_length'])))
        hip_l_x = cx - 8
        hip_l_y = cy
        thigh_l_body.position = hip_l_x, hip_l_y + s['thigh_length']/2
        thigh_l_shape = pymunk.Poly.create_box(thigh_l_body, (10, s['thigh_length']))
        thigh_l_shape.elasticity = 0.01
        thigh_l_shape.friction = 0.3
        thigh_l_shape.collision_type = COLLISION_TYPES['body']
        self.bodies['thigh_l'] = thigh_l_body
        self.shapes.append(thigh_l_shape)
        
        # === 左小腿 (Left Shin) ===
        shin_l_body = pymunk.Body(BODY_PART_MASS['shin'],
                                  pymunk.moment_for_box(BODY_PART_MASS['shin'],
                                                        (9, s['shin_length'])))
        shin_l_body.position = hip_l_x, hip_l_y + s['thigh_length'] + s['shin_length']/2
        shin_l_shape = pymunk.Poly.create_box(shin_l_body, (9, s['shin_length']))
        shin_l_shape.elasticity = 0.01
        shin_l_shape.friction = 0.3
        shin_l_shape.collision_type = COLLISION_TYPES['body']
        self.bodies['shin_l'] = shin_l_body
        self.shapes.append(shin_l_shape)
        
        # 对所有身体部件应用移动速度函数 - 使整体移动
        for name, body in self.bodies.items():
            body.velocity_func = self._apply_move_velocity
        
        # 应用碰撞过滤到所有形状 - 防止自碰撞
        for shape in self.shapes:
            self._apply_collision_filter(shape)
        
        # 将所有部件加入空间
        for body in self.bodies.values():
            self.space.add(body)
        for shape in self.shapes:
            self.space.add(shape)
    
    def _build_joints(self):
        """构建关节连接"""
        s = STICKMAN
        cx = self.bodies['torso'].position.x
        cy = self.bodies['torso'].position.y + s['torso_height']/2
        
        # 颈部: 头部->躯干
        neck_pos = (cx, cy - s['torso_height'])
        j = pymunk.PinJoint(self.bodies['head'], self.bodies['torso'], 
                           (0, s['head_radius']), (0, -s['torso_height']/2))
        self.space.add(j)
        self.joints.append(j)
        
        # 肩关节: 大臂->躯干 (使用RotaryLimitJoint限制角度)
        shoulder_r_pos = (cx + s['shoulder_width']/2, cy - s['torso_height'] * 0.8)
        
        # 对于右上臂,用PivotJoint固定到肩膀
        j = pymunk.PivotJoint(self.bodies['torso'], self.bodies['upper_arm_r'],
                             (s['shoulder_width']/2, -s['torso_height'] * 0.8 + s['torso_height']/2),
                             (-s['upper_arm_length']/2, 0))
        self.space.add(j)
        self.joints.append(j)
        
        # 右肘关节
        j = pymunk.PivotJoint(self.bodies['upper_arm_r'], self.bodies['forearm_r'],
                             (s['upper_arm_length']/2, 0), (-s['forearm_length']/2, 0))
        self.space.add(j)
        self.joints.append(j)
        
        # 右腕关节
        j = pymunk.PivotJoint(self.bodies['forearm_r'], self.bodies['hand_r'],
                             (s['forearm_length']/2, 0), (-s['hand_radius'], 0))
        self.space.add(j)
        self.joints.append(j)
        
        # 左肩关节
        shoulder_l_pos = (cx - s['shoulder_width']/2, cy - s['torso_height'] * 0.8)
        j = pymunk.PivotJoint(self.bodies['torso'], self.bodies['upper_arm_l'],
                             (-s['shoulder_width']/2, -s['torso_height'] * 0.8 + s['torso_height']/2),
                             (s['upper_arm_length']/2, 0))
        self.space.add(j)
        self.joints.append(j)
        
        # 左肘关节
        j = pymunk.PivotJoint(self.bodies['upper_arm_l'], self.bodies['forearm_l'],
                             (-s['upper_arm_length']/2, 0), (s['forearm_length']/2, 0))
        self.space.add(j)
        self.joints.append(j)
        
        # 左腕关节
        j = pymunk.PivotJoint(self.bodies['forearm_l'], self.bodies['hand_l'],
                             (-s['forearm_length']/2, 0), (s['hand_radius'], 0))
        self.space.add(j)
        self.joints.append(j)
        
        # 右髋关节
        j = pymunk.PivotJoint(self.bodies['torso'], self.bodies['thigh_r'],
                             (8, s['torso_height']/2 - 5), (0, -s['thigh_length']/2))
        self.space.add(j)
        self.joints.append(j)
        
        # 右膝关节
        j = pymunk.PivotJoint(self.bodies['thigh_r'], self.bodies['shin_r'],
                             (0, s['thigh_length']/2), (0, -s['shin_length']/2))
        self.space.add(j)
        self.joints.append(j)
        
        # 左髋关节
        j = pymunk.PivotJoint(self.bodies['torso'], self.bodies['thigh_l'],
                             (-8, s['torso_height']/2 - 5), (0, -s['thigh_length']/2))
        self.space.add(j)
        self.joints.append(j)
        
        # 左膝关节
        j = pymunk.PivotJoint(self.bodies['thigh_l'], self.bodies['shin_l'],
                             (0, s['thigh_length']/2), (0, -s['shin_length']/2))
        self.space.add(j)
        self.joints.append(j)
        
        # 使用GearJoint或SimpleMotor来让关节有弹性
        # 给右臂添加 RotaryLimitJoint 限制摆动范围
        # 由于RotaryLimitJoint需要两个body都有角度,
        # 我们用DampedRotarySpring让右臂保持自然下垂
        j = pymunk.DampedRotarySpring(self.bodies['torso'], self.bodies['upper_arm_r'],
                                      0, 1000, 50)  # 自然位置0度
        self.space.add(j)
        self.joints.append(j)
        
        j = pymunk.DampedRotarySpring(self.bodies['upper_arm_r'], self.bodies['forearm_r'],
                                      0, 800, 40)
        self.space.add(j)
        self.joints.append(j)
        
        # 左臂保持弯曲(盾牌手)
        j = pymunk.DampedRotarySpring(self.bodies['torso'], self.bodies['upper_arm_l'],
                                      -0.5, 1000, 50)
        self.space.add(j)
        self.joints.append(j)
        
        j = pymunk.DampedRotarySpring(self.bodies['upper_arm_l'], self.bodies['forearm_l'],
                                      -0.8, 800, 40)
        self.space.add(j)
        self.joints.append(j)
        
        # 腿的弹性 - 大幅增强让火柴人能站稳
        j = pymunk.DampedRotarySpring(self.bodies['torso'], self.bodies['thigh_r'],
                                      0.1, 30000, 200)
        self.space.add(j)
        self.joints.append(j)
        
        j = pymunk.DampedRotarySpring(self.bodies['thigh_r'], self.bodies['shin_r'],
                                      0, 25000, 180)
        self.space.add(j)
        self.joints.append(j)
        
        j = pymunk.DampedRotarySpring(self.bodies['torso'], self.bodies['thigh_l'],
                                      -0.1, 30000, 200)
        self.space.add(j)
        self.joints.append(j)
        
        j = pymunk.DampedRotarySpring(self.bodies['thigh_l'], self.bodies['shin_l'],
                                      0, 25000, 180)
        self.space.add(j)
        self.joints.append(j)
        
        # 给右臂添加 RotaryLimitJoint 限制范围
        j = pymunk.RotaryLimitJoint(self.bodies['torso'], self.bodies['upper_arm_r'],
                                    -2.5, 2.5)
        self.space.add(j)
        self.joints.append(j)
        
        j = pymunk.RotaryLimitJoint(self.bodies['upper_arm_r'], self.bodies['forearm_r'],
                                    -2.5, 0.5)
        self.space.add(j)
        self.joints.append(j)
    
    def _create_sword(self, x, y):
        """创建剑 - 连接在右手上"""
        s = STICKMAN
        cx = self.bodies['torso'].position.x
        cy = self.bodies['torso'].position.y + s['torso_height']/2
        shoulder_r_x = cx + s['shoulder_width']/2
        
        # 剑身: 一个薄的矩形
        sword_mass = 3.0
        sword_moment = pymunk.moment_for_box(sword_mass, (SWORD_LENGTH, SWORD_WIDTH))
        self.sword_body = pymunk.Body(sword_mass, sword_moment)
        # 剑的位置在右手延伸方向
        hand_pos = self.bodies['hand_r'].position
        self.sword_body.position = hand_pos.x + SWORD_LENGTH/2, hand_pos.y
        self.sword_body.angle = self.facing * (-0.5)  # 剑尖朝上
        
        # 剑的碰撞形状
        sword_verts = [
            (-SWORD_HANDLE, -SWORD_WIDTH/2),
            (SWORD_LENGTH - SWORD_HANDLE, -SWORD_WIDTH/2),
            (SWORD_LENGTH - SWORD_HANDLE, SWORD_WIDTH/2),
            (-SWORD_HANDLE, SWORD_WIDTH/2),
        ]
        self.sword_shape = pymunk.Poly(self.sword_body, sword_verts)
        self.sword_shape.elasticity = 0.01
        self.sword_shape.friction = 0.5
        self.sword_shape.collision_type = COLLISION_TYPES['sword']
        
        # 剑的碰撞过滤
        self._apply_collision_filter(self.sword_shape)
        
        self.space.add(self.sword_body, self.sword_shape)
        
        # 剑柄连接到右手 - 用PivotJoint
        j = pymunk.PivotJoint(self.bodies['hand_r'], self.sword_body,
                             (0, 0), (-SWORD_HANDLE, 0))
        self.space.add(j)
        self.joints.append(j)
        
        # 用DampedRotarySpring让剑自然下垂
        j = pymunk.DampedRotarySpring(self.bodies['hand_r'], self.sword_body,
                                      0, 2000, 30)
        self.space.add(j)
        self.joints.append(j)
    
    def get_sword_tip_position(self):
        """获取剑尖位置"""
        local_tip = Vec2d(SWORD_LENGTH - SWORD_HANDLE, 0)
        return self.sword_body.local_to_world(local_tip)
    
    def get_sword_base_position(self):
        """获取剑柄位置"""
        local_base = Vec2d(-SWORD_HANDLE, 0)
        return self.sword_body.local_to_world(local_base)
    
    def apply_attack(self, attack_type):
        """执行攻击 - 对右臂施加力来挥剑 (用于RL动作)"""
        if self.attack_cooldown > 0 or self.stun_timer > 0:
            return
        
        f = self.facing
        
        if attack_type == 1:  # 高右斩 (从上往下劈)
            force = 5000
            self.bodies['upper_arm_r'].apply_impulse_at_local_point(
                (0, -force * 0.5), (0, 0))
            self.bodies['forearm_r'].apply_impulse_at_local_point(
                (0, -force * 0.3), (0, 0))
        elif attack_type == -1:  # 横斩
            force = 4000
            self.bodies['upper_arm_r'].apply_impulse_at_local_point(
                (force * 0.3 * f, 0), (0, 0))
            self.bodies['forearm_r'].angular_velocity = 5 * f
        elif attack_type == 2:  # 下斩 (从下往上挑)
            force = 3000
            self.bodies['upper_arm_r'].apply_impulse_at_local_point(
                (0, force * 0.3), (0, 0))
            self.bodies['forearm_r'].angular_velocity = -8 * f
        
        self.attack_cooldown = FIGHT['attack_cooldown']
    
    def apply_mouse_swing(self, mouse_world_x, mouse_world_y, mouse_dx, mouse_dy, mouse_down):
        """鼠标控制挥剑 - 鼠标位置控制手臂指向, 拖动速度决定力度
           mouse_world_x/y: 鼠标在游戏世界中的位置
           mouse_dx/dy: 鼠标帧间移动距离 (速度)
           mouse_down: 鼠标是否按下
        """
        if self.stun_timer > 0:
            return
        
        s = STICKMAN
        # 计算肩膀位置 (世界坐标)
        shoulder = self.bodies['torso'].local_to_world(
            (self.facing * s['shoulder_width']/2, -s['torso_height'] * 0.8 + s['torso_height']/2))
        
        # 计算从肩膀到鼠标的向量
        dx = mouse_world_x - shoulder.x
        dy = mouse_world_y - shoulder.y
        
        # 计算目标角度
        target_angle = math.atan2(dy, dx) - math.pi/2  # 转换为关节角度
        
        # 鼠标移动速度 = 挥剑力度
        mouse_speed = math.sqrt(mouse_dx**2 + mouse_dy**2)
        
        if mouse_down and mouse_speed > 5:
            # 鼠标按下拖动 = 挥剑
            # 角度差
            current_angle = self.bodies['upper_arm_r'].angle
            angle_diff = target_angle - current_angle
            
            # 鼠标方向决定挥砍方向
            swing_force = min(mouse_speed * 5, 8000)
            
            # 对肩膀施加角力
            self.bodies['upper_arm_r'].apply_impulse_at_local_point(
                (0, -swing_force * math.sin(angle_diff)), (0, 0))
            self.bodies['upper_arm_r'].angular_velocity += angle_diff * 3
            
            # 前臂跟随
            forearm_angle = self.bodies['forearm_r'].angle
            f_diff = (target_angle * 0.7) - forearm_angle
            self.bodies['forearm_r'].angular_velocity += f_diff * 2
            
            self.attack_cooldown = 0  # 鼠标控制无冷却
        elif mouse_down:
            # 鼠标按下但没大幅移动 = 格挡/指向
            current_angle = self.bodies['upper_arm_r'].angle
            angle_diff = target_angle - current_angle
            self.bodies['upper_arm_r'].angular_velocity += angle_diff * 5
            self.bodies['forearm_r'].angular_velocity += (target_angle * 0.6 - self.bodies['forearm_r'].angle) * 3
            self.blocking = True
        else:
            self.blocking = False
    
    def apply_block(self, blocking):
        """执行格挡 - 抬起右臂"""
        self.blocking = blocking
        if blocking:
            # 抬起右臂做格挡姿势
            self.bodies['upper_arm_r'].angular_velocity = -3 * self.facing
            self.bodies['forearm_r'].angular_velocity = -2 * self.facing
    
    def move(self, direction):
        """移动: direction = -1 (左/后退), 0 (停止), 1 (右/前进)"""
        if self.stun_timer > 0:
            return
        
        if direction != 0:
            delta = FIGHT['move_speed'] * direction * self.facing / FPS  # per frame
            for name, body in self.bodies.items():
                body.position = (body.position.x + delta, body.position.y)
                # 让腿稍微抬高模拟走路
                if name in ['thigh_r', 'shin_r']:
                    body.position = (body.position.x, body.position.y - 2)
                elif name in ['thigh_l', 'shin_l']:
                    body.position = (body.position.x, body.position.y - 1)
            # 剑也跟着移动
            if hasattr(self, 'sword_body'):
                self.sword_body.position = (self.sword_body.position.x + delta, self.sword_body.position.y)
    
    def _apply_move_velocity(self, body, gravity, damping, dt):
        """自定义速度函数 - 移动由move()直接处理, 同时稳定躯干姿态"""
        # 先应用正常重力
        pymunk.Body.update_velocity(body, gravity, damping, dt)
        # 躯干稳定: 抑制水平旋转速度, 让躯干保持竖直
        if body == self.bodies.get('torso'):
            # 强阻尼抑制旋转
            body.angular_velocity *= 0.7
            # 如果倾斜过大, 施加回复力矩
            if abs(body.angle) > 0.15:
                body.angular_velocity -= body.angle * 10
    
    def take_damage(self, damage, impulse=None):
        """受到伤害"""
        actual_damage = damage
        if self.blocking:
            actual_damage = damage * (1 - FIGHT['block_damage_reduction'])
        
        self.health -= actual_damage
        self.stun_timer = FIGHT['stun_duration']
        
        # 击退效果
        if impulse:
            self.bodies['torso'].apply_impulse_at_world_point(impulse, self.bodies['torso'].position)
        
        if self.health <= 0:
            self.health = 0
            self.alive = False
    
    def deal_damage(self, amount):
        """记录本步造成的伤害"""
        self.damage_dealt_this_step += amount
    
    def get_center_x(self):
        """获取角色的X中心位置"""
        return self.bodies['torso'].position.x
    
    def get_center_y(self):
        return self.bodies['torso'].position.y
    
    def get_sword_velocity(self):
        """获取剑尖速度 (用于计算伤害)"""
        tip = self.get_sword_tip_position()
        return self.sword_body.velocity_at_world_point(tip).length
    
    def update(self, dt):
        self.damage_dealt_this_step = 0  # 每帧重置伤害计数
        """每帧更新"""
        if self.attack_cooldown > 0:
            self.attack_cooldown -= dt
        if self.stun_timer > 0:
            self.stun_timer -= dt
    
    def reset_position(self, x, y):
        """重置位置"""
        s = STICKMAN
        cx, cy = x, y - 20
        
        positions = {
            'torso': (cx, cy - s['torso_height']/2),
            'head': (cx, cy - s['torso_height'] - s['head_radius']),
            'upper_arm_r': (cx + s['shoulder_width']/2 + s['upper_arm_length']/2, 
                           cy - s['torso_height'] * 0.8),
            'forearm_r': (cx + s['shoulder_width']/2 + s['upper_arm_length'] + s['forearm_length']/2,
                         cy - s['torso_height'] * 0.8),
            'hand_r': (cx + s['shoulder_width']/2 + s['upper_arm_length'] + s['forearm_length'] + s['hand_radius'],
                      cy - s['torso_height'] * 0.8),
            'upper_arm_l': (cx - s['shoulder_width']/2 - s['upper_arm_length']/2,
                           cy - s['torso_height'] * 0.8),
            'forearm_l': (cx - s['shoulder_width']/2 - s['upper_arm_length'] - s['forearm_length']/2,
                         cy - s['torso_height'] * 0.8),
            'hand_l': (cx - s['shoulder_width']/2 - s['upper_arm_length'] - s['forearm_length'] - s['hand_radius'],
                      cy - s['torso_height'] * 0.8),
            'thigh_r': (cx + 8, cy + s['thigh_length']/2),
            'shin_r': (cx + 8, cy + s['thigh_length'] + s['shin_length']/2),
            'thigh_l': (cx - 8, cy + s['thigh_length']/2),
            'shin_l': (cx - 8, cy + s['thigh_length'] + s['shin_length']/2),
        }
        
        for name, pos in positions.items():
            if name in self.bodies:
                self.bodies[name].position = pos
                self.bodies[name].velocity = (0, 0)
                self.bodies[name].angular_velocity = 0
                self.bodies[name].angle = 0  # 重置角度
        
        # 重置剑
        hand_pos = self.bodies['hand_r'].position
        self.sword_body.position = hand_pos.x + SWORD_LENGTH/2, hand_pos.y
        self.sword_body.velocity = (0, 0)
        self.sword_body.angular_velocity = 0
        self.sword_body.angle = self.facing * (-0.5)
        
        # 重置状态
        self.health = self.max_health
        self.alive = True
        self.attack_cooldown = 0
        self.blocking = False
        self.stun_timer = 0
        self.damage_dealt_this_step = 0
    
    def remove_from_space(self):
        """从物理空间移除"""
        for shape in self.shapes:
            if shape in self.space.shapes:
                self.space.remove(shape)
        for body in self.bodies.values():
            if body in self.space.bodies:
                self.space.remove(body)
        for joint in self.joints:
            if joint in self.space.constraints:
                self.space.remove(joint)
        if hasattr(self, 'sword_shape') and self.sword_shape in self.space.shapes:
            self.space.remove(self.sword_shape)
        if hasattr(self, 'sword_body') and self.sword_body in self.space.bodies:
            self.space.remove(self.sword_body)
