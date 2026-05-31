"""
火柴人击剑格斗游戏主逻辑 - Bloody Bastards 风格
包含物理渲染、碰撞检测、战斗逻辑
"""
import math
import random
import pygame
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d
from game_config import *
from stickman import StickMan


class FencingGame:
    """击剑格斗游戏主类"""
    
    def __init__(self, render=True, render_surface=None):
        self.render = render
        self.screen = None  # 主窗口 (单实例用)
        self.render_surface = render_surface  # 外部渲染表面 (多实例用)
        self.clock = pygame.time.Clock() if render else None
        self.draw_options = None
        
        # 物理空间
        self.space = pymunk.Space()
        self.space.gravity = GRAVITY
        self.space.collision_slop = 0.5
        self.space.collision_persistence = 3
        
        # 玩家和对手
        self.player1 = None
        self.player2 = None
        self.sword_hit_cooldowns = {}
        
        # 碰撞回调
        self._setup_collision_handlers()
        
        # 地面
        self._create_ground()
        
        # 战斗统计
        self.p1_score = 0
        self.p2_score = 0
        self.winner = None
        self.round_timer = 0
        
        if render and render_surface is None:
            self._init_renderer()
        elif render:
            self._init_fonts()
        
        self.reset()
    
    def _init_fonts(self):
        """初始化字体"""
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 28)
        self.font_tiny = pygame.font.Font(None, 20)
    
    def _init_renderer(self):
        """初始化独立渲染器"""
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("火柴人击剑格斗 - Bloody Bastards 风格 RL")
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
        self._init_fonts()
    
    def _create_ground(self):
        """创建地面"""
        ground_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        ground_shape = pymunk.Segment(ground_body, (50, GROUND_Y), 
                                      (SCREEN_WIDTH - 50, GROUND_Y), 20)
        ground_shape.elasticity = 0.0
        ground_shape.friction = 0.3
        ground_shape.collision_type = COLLISION_TYPES['ground']
        self.space.add(ground_body, ground_shape)
        
        # 左右墙壁 - 更薄更滑, 避免吸附
        # 直接用ground_body (已在space中)
        wall_left = pymunk.Segment(ground_body, (5, 0), (5, SCREEN_HEIGHT), 5)
        wall_left.elasticity = 0.0
        wall_left.friction = 0.0
        wall_left.collision_type = COLLISION_TYPES['ground']
        self.space.add(wall_left)
        
        wall_right = pymunk.Segment(ground_body, (SCREEN_WIDTH - 5, 0), 
                                     (SCREEN_WIDTH - 5, SCREEN_HEIGHT), 5)
        wall_right.elasticity = 0.0
        wall_right.friction = 0.0
        wall_right.collision_type = COLLISION_TYPES['ground']
        self.space.add(wall_right)
    
    def _setup_collision_handlers(self):
        """设置碰撞回调 (每人独立冷却)"""
        def _get_attacker(sword_body):
            if self.player1 and sword_body == self.player1.sword_body:
                return self.player1, self.player2
            if self.player2 and sword_body == self.player2.sword_body:
                return self.player2, self.player1
            return None, None
        
        def _can_attack(attacker):
            cd = self.sword_hit_cooldowns.get(id(attacker), 0)
            return cd <= 0
        
        def _set_cooldown(attacker, cd):
            self.sword_hit_cooldowns[id(attacker)] = cd
        
        def _hit_logic(arbiter, is_head):
            shapes = arbiter.shapes
            sword_shape = shapes[0] if shapes[0].collision_type == COLLISION_TYPES['sword'] else shapes[1]
            
            attacker, defender = _get_attacker(sword_shape.body)
            if attacker is None or defender is None or not defender.alive:
                return True
            if not _can_attack(attacker):
                return True
            
            tip_vel = attacker.get_sword_velocity()
            if tip_vel < FIGHT['sword_hit_velocity_threshold']:
                return True
            
            base_dmg = FIGHT['sword_hit_damage'] * (2 if is_head else 1)
            dmg_mult = min(tip_vel / 500, 2.5 if is_head else 2.0)
            damage = int(base_dmg * dmg_mult)
            
            hit_normal = arbiter.normal
            knockback = FIGHT['knockback_force'] * (1.5 if is_head else 1.0) * dmg_mult
            impulse = hit_normal * knockback
            
            defender.take_damage(damage, impulse)
            _set_cooldown(attacker, 0.3 if is_head else 0.2)
            return True
        
        def sword_hit_body(arbiter, space, data):
            return _hit_logic(arbiter, is_head=False)
        
        def sword_hit_head(arbiter, space, data):
            return _hit_logic(arbiter, is_head=True)
        
        # 注册碰撞回调 (PyMunk 7.x API)
        self.space.on_collision(
            collision_type_a=COLLISION_TYPES['sword'],
            collision_type_b=COLLISION_TYPES['body'],
            post_solve=sword_hit_body)
        
        self.space.on_collision(
            collision_type_a=COLLISION_TYPES['sword'],
            collision_type_b=COLLISION_TYPES['head'],
            post_solve=sword_hit_head)
    
    def reset(self):
        """重置游戏"""
        # 清除旧角色
        if self.player1:
            self.player1.remove_from_space()
        if self.player2:
            self.player2.remove_from_space()
        
        # 创建新角色 - 初始距离更近以便快速交戰
        p1_x = SCREEN_WIDTH * 0.35
        p2_x = SCREEN_WIDTH * 0.55
        center_y = GROUND_Y - 20
        
        self.player1 = StickMan(self.space, p1_x, center_y, 
                                facing_right=True, color_scheme='player1')
        self.player2 = StickMan(self.space, p2_x, center_y, 
                                facing_right=False, color_scheme='player2')
        
        self.sword_hit_cooldowns.clear()
        self.winner = None
        self.round_timer = 0
        return self._get_state(perspective=1)
    
    def step(self, action1, action2):
        """执行一步游戏逻辑
        action1, action2: 两个角色的动作索引
        返回: state, reward1, reward2, done, info
        """
        dt = 1.0 / FPS
        
        # 更新每人独立冷却
        for k in list(self.sword_hit_cooldowns.keys()):
            self.sword_hit_cooldowns[k] = max(0, self.sword_hit_cooldowns[k] - dt)
        
        # 应用动作
        self._apply_action(self.player1, action1)
        self._apply_action(self.player2, action2)
        
        # 物理步进
        sub_dt = dt / PHYSICS_STEPS
        for _ in range(PHYSICS_STEPS):
            self.space.step(sub_dt)
        
        # 更新角色状态
        self.player1.update(dt)
        self.player2.update(dt)
        
        # 防止角色走出屏幕 (用冲量方式)
        self._clamp_positions()
        
        self.round_timer += dt
        
        # 检查胜负
        done = False
        if not self.player1.alive:
            self.winner = self.player2
            self.p2_score += 1
            done = True
        elif not self.player2.alive:
            self.winner = self.player1
            self.p1_score += 1
            done = True
        elif self.round_timer > 30:
            done = True
            if self.player1.health > self.player2.health:
                self.winner = self.player1
                self.p1_score += 1
            elif self.player2.health > self.player1.health:
                self.winner = self.player2
                self.p2_score += 1
        
        # 计算奖励 (含塑形奖励)
        reward1, reward2 = self._calculate_rewards(done)
        
        # 各自视角的状态
        state1 = self._get_state(perspective=1)
        state2 = self._get_state(perspective=2)
        info = {
            'health1': self.player1.health,
            'health2': self.player2.health,
            'winner': self.winner,
        }
        
        return state1, reward1, state2, reward2, done, info
    
    def _apply_action(self, stickman, action_idx):
        """对角色应用动作"""
        if action_idx not in ACTION_MAP:
            return
        
        action = ACTION_MAP[action_idx]
        
        # 攻击
        if action['attack'] != 0:
            stickman.apply_attack(action['attack'])
        
        # 格挡
        stickman.apply_block(action['block'])
        
        # 移动
        if action['move'] != 0:
            stickman.move(action['move'])
    
    def _calculate_rewards(self, done):
        """计算奖励 (含塑形奖励)"""
        r1 = 0.0
        r2 = 0.0
        
        p1, p2 = self.player1, self.player2
        
        # 血量差奖励
        health_diff = p1.health - p2.health
        r1 += health_diff * 0.2
        r2 += -health_diff * 0.2
        
        # 位置奖励 - 鼓励面对面靠近
        dist = abs(p2.get_center_x() - p1.get_center_x())
        facing_right = p1.facing > 0
        p1_right_of_p2 = p1.get_center_x() > p2.get_center_x()
        if facing_right != p1_right_of_p2:
            proximity = max(0, (500 - dist) / 5000)
            r1 += proximity
            r2 += proximity
        
        # 剑尖速度奖励 - 鼓励攻击
        r1 += min(p1.get_sword_velocity() / 2000, 0.5)
        r2 += min(p2.get_sword_velocity() / 2000, 0.5)
        
        # 击败/超时
        if done:
            if self.winner == p1:
                r1 += 100; r2 -= 100
            elif self.winner == p2:
                r2 += 100; r1 -= 100
        
        return r1, r2
    
    def _clamp_positions(self):
        """用冲量限制位置 - 避免直接修改物理体位置"""
        margin = 30
        for stickman in [self.player1, self.player2]:
            if not stickman:
                continue
            for name, body in stickman.bodies.items():
                x, y = body.position.x, body.position.y
                if x < margin:
                    # 用冲量推回 + 极低弹性排斥
                    body.apply_impulse_at_local_point((500 + (margin - x) * 10, 0), (0, 0))
                    body.velocity = (max(body.velocity.x, 0), body.velocity.y)
                elif x > SCREEN_WIDTH - margin:
                    body.apply_impulse_at_local_point((-500 - (x - SCREEN_WIDTH + margin) * 10, 0), (0, 0))
                    body.velocity = (min(body.velocity.x, 0), body.velocity.y)
                if y > GROUND_Y + 50:
                    body.apply_impulse_at_local_point((0, -800), (0, 0))
                    body.velocity = (body.velocity.x, min(body.velocity.y, -20))
    
    def _get_state(self, perspective=1):
        """获取状态向量 (perspective=1: P1视角, 2: P2镜像视角)
        P2视角时镜像x坐标、翻转角度符号、交换自身/对手信息位置
        """
        p1, p2 = self.player1, self.player2
        
        if perspective == 1:
            me, opp = p1, p2
            flip = 1  # 不翻转
        else:
            me, opp = p2, p1
            flip = -1  # 镜像翻转
        
        def get_self_features(agent):
            return [
                agent.get_center_x() / SCREEN_WIDTH,
                agent.get_center_y() / SCREEN_HEIGHT,
                agent.health / agent.max_health,
                agent.bodies['torso'].velocity.x / 500,
                agent.bodies['torso'].velocity.y / 500,
                agent.bodies['upper_arm_r'].angle * flip / math.pi,
                agent.bodies['forearm_r'].angle * flip / math.pi,
                agent.bodies['upper_arm_r'].angular_velocity * flip / 10,
                agent.bodies['forearm_r'].angular_velocity * flip / 10,
                agent.sword_body.angle * flip / math.pi,
                agent.sword_body.angular_velocity * flip / 10,
                float(agent.blocking),
                agent.attack_cooldown / FIGHT['attack_cooldown'],
                agent.stun_timer / FIGHT['stun_duration'],
            ]
        
        def get_opp_features(agent):
            return [
                agent.get_center_x() / SCREEN_WIDTH,
                agent.get_center_y() / SCREEN_HEIGHT,
                agent.health / agent.max_health,
                agent.bodies['torso'].velocity.x / 500,
                agent.bodies['torso'].velocity.y / 500,
                agent.bodies['upper_arm_r'].angle * flip / math.pi,
                agent.bodies['forearm_r'].angle * flip / math.pi,
                agent.bodies['upper_arm_r'].angular_velocity * flip / 10,
                agent.bodies['forearm_r'].angular_velocity * flip / 10,
                agent.sword_body.angle * flip / math.pi,
                agent.sword_body.angular_velocity * flip / 10,
                float(agent.blocking),
                agent.attack_cooldown / FIGHT['attack_cooldown'],
                agent.stun_timer / FIGHT['stun_duration'],
            ]
        
        state = get_self_features(me) + get_opp_features(opp)
        # 相对距离 (从自身视角)
        rel_dist = (opp.get_center_x() - me.get_center_x()) * flip / SCREEN_WIDTH
        state.append(rel_dist)
        return state
    
    def get_surface(self):
        """获取当前应绘制的表面"""
        return self.render_surface if self.render_surface is not None else self.screen
    
    def render_frame(self):
        """渲染一帧"""
        if not self.render:
            return True
        
        surf = self.get_surface()
        if surf is None:
            return True
        
        is_main_window = self.render_surface is None
        
        # 只在主窗口模式处理事件
        if is_main_window:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
        
        w, h = surf.get_size()
        # 缩放比例: 如果render_surface尺寸与标准不同
        sx = w / SCREEN_WIDTH
        sy = h / SCREEN_HEIGHT
        
        # 背景
        surf.fill(COLORS['background'])
        
        # 地面 (在当前表面坐标系中)
        ground_y = int(GROUND_Y * sy)
        pygame.draw.rect(surf, COLORS['ground'],
                        (0, ground_y, w, h - ground_y))
        
        # 绘制火柴人
        self._draw_stickman(surf, self.player1, COLORS['player1_body'], sx, sy)
        self._draw_stickman(surf, self.player2, COLORS['player2_body'], sx, sy)
        
        # 绘制剑
        self._draw_sword(surf, self.player1, COLORS['player1_sword'], sx, sy)
        self._draw_sword(surf, self.player2, COLORS['player2_sword'], sx, sy)
        
        # 血条
        self._draw_health_bar(surf, sx, sy)
        
        # 圈号文字
        if hasattr(self, '_viewer_index') and self.font_tiny:
            label = self.font_tiny.render(f"#{self._viewer_index}", True, (200, 200, 200))
            surf.blit(label, (5, 5))
        
        if is_main_window:
            pygame.display.flip()
            self.clock.tick(FPS)
        
        return True
    
    def _draw_stickman(self, surf, stickman, color, sx=1.0, sy=1.0):
        """绘制火柴人"""
        if not stickman.alive:
            color = (color[0]//2, color[1]//2, color[2]//2)
        
        b = stickman.bodies
        s = STICKMAN
        
        def gp(name):
            return (int(b[name].position.x * sx), int(b[name].position.y * sy))
        
        connections = [
            ('head', 'torso'), ('torso', 'upper_arm_r'),
            ('upper_arm_r', 'forearm_r'), ('forearm_r', 'hand_r'),
            ('torso', 'upper_arm_l'), ('upper_arm_l', 'forearm_l'),
            ('forearm_l', 'hand_l'), ('torso', 'thigh_r'),
            ('thigh_r', 'shin_r'), ('torso', 'thigh_l'), ('thigh_l', 'shin_l'),
        ]
        
        for start, end in connections:
            if start in b and end in b:
                pygame.draw.line(surf, color, gp(start), gp(end), max(2, int(3 * sx)))
        
        head_pos = gp('head')
        r = int(s['head_radius'] * sx)
        pygame.draw.circle(surf, color, head_pos, r, 2)
        pygame.draw.circle(surf, color, head_pos, r - 1)
        
        torso_pos = gp('torso')
        tw = int(s['torso_width'] * sx)
        th = int(s['torso_height'] * sy)
        torso_rect = pygame.Rect(0, 0, tw, th)
        torso_rect.center = torso_pos
        armor_c = COLORS['player1_armor'] if stickman == self.player1 else COLORS['player2_armor']
        pygame.draw.rect(surf, armor_c, torso_rect)
        pygame.draw.rect(surf, color, torso_rect, 2)
        
        for hand_name in ['hand_r', 'hand_l']:
            hr = int(s['hand_radius'] * sx)
            pygame.draw.circle(surf, (255, 220, 180), gp(hand_name), hr)
    
    def _draw_sword(self, surf, stickman, color, sx=1.0, sy=1.0):
        """绘制剑"""
        if not hasattr(stickman, 'sword_body'):
            return
        tip = stickman.get_sword_tip_position()
        base = stickman.get_sword_base_position()
        handle = stickman.sword_body.local_to_world(Vec2d(-SWORD_HANDLE, 0))
        bw = max(2, int((SWORD_WIDTH + 2) * sx))
        pygame.draw.line(surf, color, (int(base.x*sx), int(base.y*sy)),
                        (int(tip.x*sx), int(tip.y*sy)), bw)
        hlw = max(1, int(2 * sx))
        pygame.draw.line(surf, (255, 255, 255, 128),
                        (int(base.x*sx), int(base.y*sy)),
                        (int(tip.x*sx), int(tip.y*sy)), hlw)
        hw = max(2, int(4 * sx))
        pygame.draw.line(surf, (139, 69, 19),
                        (int(handle.x*sx), int(handle.y*sy)),
                        (int(base.x*sx), int(base.y*sy)), hw)
    
    def _draw_health_bar(self, surf, sx=1.0, sy=1.0):
        """绘制血条"""
        w = surf.get_width()
        bw = int(350 * sx)
        bh = int(20 * sy)
        by = int(10 * sy)
        
        # P1
        x1 = int(10 * sx)
        r1 = max(0, self.player1.health / self.player1.max_health)
        pygame.draw.rect(surf, COLORS['health_bg'], (x1, by, bw, bh))
        pygame.draw.rect(surf, COLORS['health_bar'], (x1, by, int(bw * r1), bh))
        pygame.draw.rect(surf, (255, 255, 255), (x1, by, bw, bh), max(1, int(2*sx)))
        
        # P2
        x2 = w - bw - int(10 * sx)
        r2 = max(0, self.player2.health / self.player2.max_health)
        pygame.draw.rect(surf, COLORS['health_bg'], (x2, by, bw, bh))
        pygame.draw.rect(surf, COLORS['health_bar'],
                        (x2 + int(bw * (1 - r2)), by, int(bw * r2), bh))
        pygame.draw.rect(surf, (255, 255, 255), (x2, by, bw, bh), max(1, int(2*sx)))
        
        if self.font_tiny:
            fs = max(8, int(20 * sx))
            f = pygame.font.Font(None, fs)
            t1 = f.render(f"{int(self.player1.health)}", True, (255,255,255))
            surf.blit(t1, (x1 + int(5*sx), by + int(2*sy)))
            t2 = f.render(f"{int(self.player2.health)}", True, (255,255,255))
            surf.blit(t2, (x2 + int(5*sx), by + int(2*sy)))
    
    def _draw_mouse_trail(self):
        """绘制鼠标拖拽轨迹"""
        if not hasattr(self, '_prev_mouse'):
            return
        surf = self.get_surface()
        if surf is None:
            return
        mx, my = pygame.mouse.get_pos()
        cx = int(self.player1.get_center_x())
        cy = int(self.player1.get_center_y())
        for i in range(0, 100, 8):
            t = i / 100
            x = int(cx + (mx - cx) * t)
            y = int(cy + (my - cy) * t)
            pygame.draw.circle(surf, (255, 255, 100, 128), (x, y), 2)
        pygame.draw.circle(surf, (255, 200, 50), (mx, my), 8, 2)
        pygame.draw.circle(surf, (255, 200, 50, 64), (mx, my), 15, 1)
    
    def _draw_ui(self):
        """绘制UI"""
        if not self.font_small:
            return
        surf = self.get_surface()
        if surf is None:
            return
        w = surf.get_width()
        
        score_text = self.font_small.render(
            f"P1:{self.p1_score} P2:{self.p2_score}",
            True, COLORS['ui_text'])
        sr = score_text.get_rect(center=(w//2, int(30)))
        surf.blit(score_text, sr)
    
    def _is_on_ground(self, stickman):
        """检测角色是否在地面上 (小腿是否接触地面)"""
        if not stickman:
            return False
        for leg in ['shin_r', 'shin_l']:
            body = stickman.bodies.get(leg)
            if not body:
                continue
            # 小腿底部接近地面高度
            foot_y = body.position.y + 14  # 小腿长度一半
            if foot_y >= GROUND_Y - 5:
                return True
        return False
    
    def handle_human_input(self):
        """处理人类玩家的WASD+鼠标输入 - 在step前调用"""
        if not self.render:
            return
        
        keys = pygame.key.get_pressed()
        move_dir = 0
        
        # WASD移动
        if keys[pygame.K_d]:
            move_dir = 1
        elif keys[pygame.K_a]:
            move_dir = -1
        # W跳跃 - 需要在地面上
        if keys[pygame.K_w] and self._is_on_ground(self.player1):
            self.player1.bodies['torso'].apply_impulse_at_local_point((0, -800), (0, 0))
            for leg in ['thigh_r', 'thigh_l', 'shin_r', 'shin_l']:
                b = self.player1.bodies.get(leg)
                if b:
                    b.apply_impulse_at_local_point((0, -200), (0, 0))
        
        # 鼠标状态
        mouse_buttons = pygame.mouse.get_pressed()
        mouse_x, mouse_y = pygame.mouse.get_pos()
        
        # 鼠标增量 (用于计算挥剑速度)
        if not hasattr(self, '_prev_mouse'):
            self._prev_mouse = (mouse_x, mouse_y)
            self._mouse_delta = (0, 0)
        
        prev_x, prev_y = self._prev_mouse
        self._mouse_delta = (mouse_x - prev_x, mouse_y - prev_y)
        self._prev_mouse = (mouse_x, mouse_y)
        
        # 应用鼠标控制 - 控制P1右臂挥剑
        mouse_down = mouse_buttons[0]
        self.player1.apply_mouse_swing(
            mouse_x, mouse_y,
            self._mouse_delta[0], self._mouse_delta[1],
            mouse_down
        )
        
        # 应用WASD移动
        self.player1.move(move_dir)
    
    def human_step(self, ai_action):
        """人机对战一步 - 处理人类输入+AI动作+物理步进
        ai_action: AI选择的动作索引
        返回: (state, reward, done, info)
        """
        dt = 1.0 / FPS
        
        # 更新每人独立冷却
        for k in list(self.sword_hit_cooldowns.keys()):
            self.sword_hit_cooldowns[k] = max(0, self.sword_hit_cooldowns[k] - dt)
        
        # 处理人类WASD+鼠标输入 (含地面检测)
        self.handle_human_input()
        
        # AI控制P2
        self._apply_action(self.player2, ai_action)
        
        # 物理步进
        sub_dt = dt / PHYSICS_STEPS
        for _ in range(PHYSICS_STEPS):
            self.space.step(sub_dt)
        
        # 更新角色状态
        self.player1.update(dt)
        self.player2.update(dt)
        
        # 防止走出屏幕
        self._clamp_positions()
        
        self.round_timer += dt
        
        # 检查胜负
        done = False
        if not self.player1.alive:
            self.winner = self.player2
            self.p2_score += 1
            done = True
        elif not self.player2.alive:
            self.winner = self.player1
            self.p1_score += 1
            done = True
        elif self.round_timer > 30:
            done = True
            if self.player1.health > self.player2.health:
                self.winner = self.player1
                self.p1_score += 1
            elif self.player2.health > self.player1.health:
                self.winner = self.player2
                self.p2_score += 1
        
        reward = 0
        if done:
            if self.winner == self.player1:
                reward = 10
            elif self.winner == self.player2:
                reward = -10
        
        state = self._get_state(perspective=1)
        info = {
            'health1': self.player1.health,
            'health2': self.player2.health,
            'winner': self.winner,
        }
        
        return state, reward, done, info
    
    def close(self):
        """关闭游戏"""
        if self.render:
            pygame.quit()
