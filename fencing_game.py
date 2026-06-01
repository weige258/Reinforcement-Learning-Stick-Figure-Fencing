"""
主游戏逻辑 - 基于Bloody Bastards风格的火柴人击剑格斗游戏
包含: PyMunk物理世界、Pygame渲染、碰撞处理、战斗系统
"""
import sys
import math
import random
import pygame
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d

import game_config as cfg
from stickman import StickMan


class FencingGame:
    """击剑格斗游戏主类"""

    def __init__(self, render=True):
        self.render = render
        self.dt = 1.0 / cfg.FPS
        self.clock = pygame.time.Clock()
        self.space = None
        self.screen = None
        self.draw_options = None
        self.player1 = None
        self.player2 = None
        self.ground_body = None
        self.ground_shape = None

        # 战斗状态
        self.hit_effects = []  # 打击特效
        self.blood_particles = []  # 血液粒子
        self.combat_log = []  # 战斗日志
        self.round_time = 0
        self.winner = None
        self.game_over = False

        # 鼠标位置
        self.mouse_pos = (0, 0)

        # 碰撞处理器
        self.collision_handler = None

        if render:
            self._init_pygame()

        self._init_physics()
        self._setup_collision_handlers()

    def _init_pygame(self):
        """初始化Pygame"""
        pygame.init()
        self.screen = pygame.display.set_mode((cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT))
        pygame.display.set_caption("火柴人击剑格斗 - Bloody Bastards风格")
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 20)
        self.big_font = pygame.font.Font(None, 48)
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

    def _init_physics(self):
        """初始化物理世界"""
        self.space = pymunk.Space()
        self.space.gravity = cfg.GRAVITY
        self.space.damping = cfg.DAMPING

        # 创建地面
        self.ground_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.ground_body.position = (cfg.WINDOW_WIDTH // 2, cfg.GROUND_Y + cfg.GROUND_THICKNESS // 2)

        self.ground_shape = pymunk.Segment(
            self.ground_body,
            (-cfg.WINDOW_WIDTH, 0),
            (cfg.WINDOW_WIDTH * 2, 0),
            cfg.GROUND_THICKNESS // 2
        )
        self.ground_shape.friction = 0.8
        self.ground_shape.elasticity = 0.1
        self.ground_shape.collision_type = 4  # COLL_GROUND
        self.space.add(self.ground_body, self.ground_shape)

        # 创建墙壁（防止角色出界）
        self._create_walls()

    def _create_walls(self):
        """创建边界墙壁"""
        # 左墙
        left_wall_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        left_wall = pymunk.Segment(left_wall_body, (0, 0), (0, cfg.WINDOW_HEIGHT), 10)
        left_wall.friction = 0.5
        left_wall.elasticity = 0.3
        self.space.add(left_wall_body, left_wall)

        # 右墙
        right_wall_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        right_wall = pymunk.Segment(right_wall_body,
                                    (cfg.WINDOW_WIDTH, 0),
                                    (cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT), 10)
        right_wall.friction = 0.5
        right_wall.elasticity = 0.3
        self.space.add(right_wall_body, right_wall)

    def _setup_collision_handlers(self):
        """设置碰撞处理器 (pymunk 7.x on_collision API)"""
        # 剑 vs 身体 (COLL_SWORD=2, COLL_BODY=1)
        self.space.on_collision(
            collision_type_a=2, collision_type_b=1,
            begin=self._sword_vs_body_begin,
            separate=self._sword_vs_body_separate,
            data=self
        )
        # 剑 vs 头部 (COLL_SWORD=2, COLL_HEAD=5)
        self.space.on_collision(
            collision_type_a=2, collision_type_b=5,
            begin=self._sword_vs_head_begin,
            data=self
        )
        # 剑 vs 盾 (COLL_SWORD=2, COLL_SHIELD=3)
        self.space.on_collision(
            collision_type_a=2, collision_type_b=3,
            begin=self._sword_vs_shield_begin,
            data=self
        )

        # 追踪每把剑最后击中的对手和冷却
        self._sword_hit_targets = {}
        self._sword_cooldowns = {}

    def _sword_vs_body_begin(self, arbiter, space, data):
        """剑击中身体"""
        game = data
        s0, s1 = arbiter.shapes
        if s0.collision_type == 2:
            sword_shape, body_shape = s0, s1
        else:
            sword_shape, body_shape = s1, s0

        sword_stickman = getattr(sword_shape, 'stickman', None)
        body_stickman = getattr(body_shape, 'stickman', None)

        if sword_stickman and body_stickman and sword_stickman is not body_stickman:
            hit_key = id(sword_stickman)
            current_time = game.round_time
            last_hit = game._sword_cooldowns.get(hit_key, -1)
            if current_time - last_hit < 0.2:
                return True  # 保持物理碰撞, 仅跳过伤害

            sword_tip_vel = sword_stickman.get_sword_tip_velocity()
            speed = sword_tip_vel.length

            if speed > cfg.ATTACK_SPEED_THRESHOLD:
                game._sword_cooldowns[hit_key] = current_time
                # 伤害 = 基础18 × 刀速/50 (速度越快伤害越高)
                damage = cfg.SWORD_BASE_DAMAGE * (speed / 50.0)
                damage = min(damage, cfg.SWORD_MAX_DAMAGE)

                contact_point = arbiter.contact_point_set
                if contact_point and contact_point.points:
                    hit_pos = contact_point.points[0].point_b
                else:
                    hit_pos = body_stickman.get_position()

                died = body_stickman.take_damage(damage, hit_pos)

                if game.render:
                    game._spawn_blood_particles(hit_pos, speed)
                    game._add_hit_effect(hit_pos, damage)

                if died:
                    game.winner = sword_stickman.player_id
                    game.game_over = True

                return True

        return True  # 始终保留物理碰撞

    def _sword_vs_body_separate(self, arbiter, space, data):
        """剑与身体分离"""
        return True

    def _sword_vs_head_begin(self, arbiter, space, data):
        """剑击中头部 - 暴击"""
        game = data
        s0, s1 = arbiter.shapes
        if s0.collision_type == 2:
            sword_shape, head_shape = s0, s1
        else:
            sword_shape, head_shape = s1, s0

        sword_stickman = getattr(sword_shape, 'stickman', None)
        head_stickman = getattr(head_shape, 'stickman', None)

        if sword_stickman and head_stickman and sword_stickman is not head_stickman:
            hit_key = id(sword_stickman)
            current_time = game.round_time
            last_hit = game._sword_cooldowns.get(hit_key, -1)
            if current_time - last_hit < 0.3:
                return True  # 保持物理碰撞

            sword_tip_vel = sword_stickman.get_sword_tip_velocity()
            speed = sword_tip_vel.length

            if speed > cfg.ATTACK_SPEED_THRESHOLD:
                game._sword_cooldowns[hit_key] = current_time
                # 头部暴击: 基础18 × 刀速/50 × 2倍
                damage = cfg.SWORD_BASE_DAMAGE * (speed / 50.0) * cfg.HEAD_DAMAGE_MULTIPLIER
                damage = min(damage, cfg.SWORD_MAX_DAMAGE * 2)

                contact_point = arbiter.contact_point_set
                if contact_point and contact_point.points:
                    hit_pos = contact_point.points[0].point_b
                else:
                    hit_pos = head_stickman.get_head_position()

                died = head_stickman.take_damage(damage, hit_pos)

                if game.render:
                    game._spawn_blood_particles(hit_pos, speed * 1.5)
                    game._add_hit_effect(hit_pos, damage, headshot=True)

                if died:
                    game.winner = sword_stickman.player_id
                    game.game_over = True

                return True

        return True  # 始终保留物理碰撞

    def _sword_vs_shield_begin(self, arbiter, space, data):
        """剑击中盾牌"""
        game = data
        s0, s1 = arbiter.shapes
        if s0.collision_type == 2:
            sword_shape, shield_shape = s0, s1
        else:
            sword_shape, shield_shape = s1, s0

        sword_stickman = getattr(sword_shape, 'stickman', None)
        shield_stickman = getattr(shield_shape, 'stickman', None)

        if sword_stickman and shield_stickman and sword_stickman != shield_stickman:
            sword_tip_vel = sword_stickman.get_sword_tip_velocity()
            speed = sword_tip_vel.length

            if speed > cfg.ATTACK_SPEED_THRESHOLD:
                if game.render:
                    contact_point = arbiter.contact_point_set
                    if contact_point and contact_point.points:
                        hit_pos = contact_point.points[0].point_b
                        game._add_hit_effect(hit_pos, 0, block=True)
                        game._spawn_spark_particles(hit_pos, speed)

                if sword_stickman.sword_body:
                    recoil = -sword_tip_vel * 0.3
                    sword_stickman.sword_body.apply_impulse_at_world_point(
                        recoil, sword_stickman.get_sword_tip_position()
                    )

                return True

        return True  # 始终保留物理碰撞

    def _spawn_blood_particles(self, pos, speed):
        """生成血液粒子特效"""
        for _ in range(min(int(speed / 30), 8)):
            particle = {
                'pos': Vec2d(pos.x, pos.y),
                'vel': Vec2d(random.uniform(-200, 200), random.uniform(-300, 0)),
                'life': random.uniform(0.3, 0.8),
                'max_life': 0.8,
                'size': random.uniform(3, 6),
                'color': cfg.BLOOD_COLOR
            }
            self.blood_particles.append(particle)

    def _spawn_spark_particles(self, pos, speed):
        """生成火花粒子（格挡时）"""
        for _ in range(min(int(speed / 50), 5)):
            particle = {
                'pos': Vec2d(pos.x, pos.y),
                'vel': Vec2d(random.uniform(-150, 150), random.uniform(-200, 50)),
                'life': random.uniform(0.2, 0.5),
                'max_life': 0.5,
                'size': random.uniform(2, 4),
                'color': (255, 255, 200)
            }
            self.blood_particles.append(particle)

    def _add_hit_effect(self, pos, damage, headshot=False, block=False):
        """添加打击特效"""
        effect = {
            'pos': Vec2d(pos.x, pos.y),
            'life': 0.5,
            'max_life': 0.5,
            'damage': damage,
            'headshot': headshot,
            'block': block
        }
        self.hit_effects.append(effect)

    def create_players(self):
        """创建两个玩家"""
        self.player1 = StickMan(
            self.space, cfg.WINDOW_WIDTH * 0.25, cfg.GROUND_Y - 80,
            facing_right=True, player_id=1
        )
        self.player2 = StickMan(
            self.space, cfg.WINDOW_WIDTH * 0.75, cfg.GROUND_Y - 80,
            facing_right=False, player_id=2
        )

        # 为形状设置stickman引用
        for shape in self.player1.shapes.values():
            shape.stickman = self.player1
        for shape in self.player2.shapes.values():
            shape.stickman = self.player2

    def reset(self):
        """重置游戏状态"""
        if self.player1:
            self.player1.remove()
        if self.player2:
            self.player2.remove()

        self.hit_effects.clear()
        self.blood_particles.clear()
        self.combat_log.clear()
        self.round_time = 0
        self.winner = None
        self.game_over = False
        self._sword_hit_targets.clear()
        self._sword_cooldowns.clear()

        self.create_players()

        return self._get_state()

    def _get_state(self):
        """获取游戏状态（用于RL）"""
        if not self.player1 or not self.player2:
            return None

        p1_state = self.player1.get_state_vector(self.player2.get_position())
        p2_state = self.player2.get_state_vector(self.player1.get_position())

        return {
            'player1': p1_state,
            'player2': p2_state,
        }

    def get_state_for_agent(self, agent_id=1):
        """为指定智能体获取状态"""
        state = self._get_state()
        if state is None:
            return None
        return state[f'player{agent_id}']

    def step(self, action_p1=None, action_p2=None):
        """执行一步游戏更新"""
        # 检查窗口关闭: peek不消费事件, 留给handle_events处理
        if self.render and pygame.event.peek(pygame.QUIT):
            self.game_over = True
            return {'player1': 0, 'player2': 0}

        self.round_time += self.dt

        # 应用动作
        if action_p1 and self.player1 and self.player1.is_alive():
            self.player1.apply_movement(*action_p1)

        if action_p2 and self.player2 and self.player2.is_alive():
            self.player2.apply_movement(*action_p2)

        # 更新物理
        self.space.step(self.dt)

        # 更新粒子特效
        self._update_particles()

        # 更新打击特效
        self._update_hit_effects()

        # 检查是否有玩家死亡
        if self.player1 and self.player2:
            if not self.player1.is_alive():
                self.winner = 2
                self.game_over = True
            elif not self.player2.is_alive():
                self.winner = 1
                self.game_over = True

        # 获取奖励
        reward = self._calculate_rewards(action_p1, action_p2)

        return reward

    def _update_particles(self):
        """更新粒子"""
        for p in self.blood_particles[:]:
            p['pos'] += p['vel'] * self.dt
            p['vel'] *= 0.95
            p['vel'] = Vec2d(p['vel'].x, p['vel'].y + 200 * self.dt)  # 重力
            p['life'] -= self.dt
            if p['life'] <= 0:
                self.blood_particles.remove(p)

    def _update_hit_effects(self):
        """更新打击特效"""
        for e in self.hit_effects[:]:
            e['life'] -= self.dt
            if e['life'] <= 0:
                self.hit_effects.remove(e)

    def _calculate_rewards(self, action_p1, action_p2):
        """计算奖励

        Returns:
            dict: {'player1': reward1, 'player2': reward2}
        """
        reward1 = 0.0
        reward2 = 0.0

        if not self.player1 or not self.player2:
            return {'player1': 0, 'player2': 0}

        p1_pos = self.player1.get_position()
        p2_pos = self.player2.get_position()

        # 向对手移动奖励
        prev_dist = getattr(self, '_prev_dist', abs(p1_pos.x - p2_pos.x))
        curr_dist = abs(p1_pos.x - p2_pos.x)

        if curr_dist < prev_dist:
            reward1 += cfg.REWARD_MOVE_TOWARD
            reward2 += cfg.REWARD_MOVE_TOWARD
        elif curr_dist > prev_dist:
            reward1 -= cfg.REWARD_MOVE_TOWARD * 0.5
            reward2 -= cfg.REWARD_MOVE_TOWARD * 0.5

        self._prev_dist = curr_dist

        # 生命值变化奖励(修复符号: 掉血=负, 对手掉血=正)
        if hasattr(self, '_prev_health_p1'):
            hd1 = self.player1.health - self._prev_health_p1
            hd2 = self.player2.health - self._prev_health_p2

            # 自己掉血(hd<0) → 负奖励 ✓
            reward1 += hd1 * (abs(cfg.REWARD_DAMAGE_TAKEN) / cfg.SWORD_BASE_DAMAGE)
            reward2 += hd2 * (abs(cfg.REWARD_DAMAGE_TAKEN) / cfg.SWORD_BASE_DAMAGE)

            # 对手掉血(hd2<0) → 正奖励 ✓
            reward1 -= hd2 * (cfg.REWARD_HIT / cfg.SWORD_BASE_DAMAGE)
            reward2 -= hd1 * (cfg.REWARD_HIT / cfg.SWORD_BASE_DAMAGE)

        self._prev_health_p1 = self.player1.health
        self._prev_health_p2 = self.player2.health

        # 死亡/击杀奖励
        if self.game_over and self.winner:
            if self.winner == 1:
                reward1 += cfg.REWARD_KILL
                reward2 += cfg.REWARD_DEATH
            else:
                reward2 += cfg.REWARD_KILL
                reward1 += cfg.REWARD_DEATH

        # 每步惩罚（促进快速取胜）
        reward1 += cfg.REWARD_STEP_PENALTY
        reward2 += cfg.REWARD_STEP_PENALTY

        # 剑挥动速度奖励（鼓励攻击）
        if action_p1 and action_p1[3]:  # 攻击
            sword_speed = self.player1.get_sword_tip_velocity().length
            reward1 += cfg.REWARD_SWORD_SPEED * (sword_speed / 500)

        if action_p2 and action_p2[3]:  # 攻击
            sword_speed = self.player2.get_sword_tip_velocity().length
            reward2 += cfg.REWARD_SWORD_SPEED * (sword_speed / 500)

        return {'player1': reward1, 'player2': reward2}

    def handle_events(self):
        """处理Pygame事件"""
        if not self.render:
            return []

        events = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return ['quit']
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return ['quit']
                elif event.key == pygame.K_r and self.game_over:
                    self.reset()
                    return ['reset']
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONDOWN:
                events.append(('mouse_down', event.button, event.pos))
            elif event.type == pygame.MOUSEBUTTONUP:
                events.append(('mouse_up', event.button, event.pos))

        return events

    def get_human_input(self, player_id=1):
        """获取人类玩家的输入

        WASD移动，鼠标左键挥剑，鼠标右键举盾
        """
        keys = pygame.key.get_pressed()
        mouse_buttons = pygame.mouse.get_pressed()

        move_x = 0
        move_y = 0
        jump = False
        attack = False
        block = False

        # WASD移动
        if keys[pygame.K_a] or keys[pygame.K_d]:
            if keys[pygame.K_a]:
                move_x = -1
            if keys[pygame.K_d]:
                move_x = 1
        if keys[pygame.K_w] or keys[pygame.K_s]:
            if keys[pygame.K_w]:
                move_y = -1
            if keys[pygame.K_s]:
                move_y = 1

        # W键跳跃（按上）
        if keys[pygame.K_w]:
            jump = True

        # 鼠标左键挥剑
        if mouse_buttons[0]:
            attack = True

        # 鼠标右键举盾
        if mouse_buttons[2]:
            block = True

        return (move_x, move_y, jump, attack, block)

    def render_frame(self):
        """渲染一帧"""
        if not self.render or not self.screen:
            return

        self.screen.fill(cfg.BACKGROUND_COLOR)

        # 绘制背景
        self._draw_background()

        # 绘制地面
        self._draw_ground()

        # 绘制物理对象
        self._draw_physics()

        # 绘制粒子特效
        self._draw_particles()

        # 绘制打击特效
        self._draw_hit_effects()

        # 绘制HUD
        self._draw_hud()

        # 绘制游戏结束信息
        if self.game_over:
            self._draw_game_over()

        pygame.display.flip()
        self.clock.tick(cfg.FPS)

    def _draw_background(self):
        """绘制背景"""
        # 渐变天空
        for i in range(cfg.GROUND_Y):
            color = (
                cfg.SKY_COLOR[0] + i * 10 // cfg.GROUND_Y,
                cfg.SKY_COLOR[1] + i * 5 // cfg.GROUND_Y,
                cfg.SKY_COLOR[2] + i * 8 // cfg.GROUND_Y
            )
            pygame.draw.line(self.screen, color, (0, i), (cfg.WINDOW_WIDTH, i))

    def _draw_ground(self):
        """绘制地面"""
        ground_rect = pygame.Rect(
            0, cfg.GROUND_Y,
            cfg.WINDOW_WIDTH, cfg.GROUND_THICKNESS
        )
        pygame.draw.rect(self.screen, cfg.GROUND_COLOR, ground_rect)
        pygame.draw.line(self.screen, (100, 90, 80),
                        (0, cfg.GROUND_Y), (cfg.WINDOW_WIDTH, cfg.GROUND_Y), 2)

    def _draw_physics(self):
        """绘制物理对象"""
        if not self.player1 or not self.player2:
            return

        for player in [self.player1, self.player2]:
            self._draw_stickman(player)

    def _draw_stickman(self, stickman):
        """绘制火柴人"""
        if not stickman.is_alive() and stickman.get_health_ratio() <= 0:
            alpha = max(0.3, stickman.get_health_ratio())
        else:
            alpha = 1.0

        color = stickman.color

        # 绘制身体部分
        body_parts = [
            ('torso', color),
            ('head', cfg.HEAD_COLOR),
            ('left_upper_arm', color),
            ('left_lower_arm', color),
            ('right_upper_arm', color),
            ('right_lower_arm', color),
            ('left_upper_leg', color),
            ('left_lower_leg', color),
            ('right_upper_leg', color),
            ('right_lower_leg', color),
        ]

        for name, part_color in body_parts:
            if name in stickman.bodies and name in stickman.shapes:
                body = stickman.bodies[name]
                shape = stickman.shapes[name]
                if isinstance(shape, pymunk.Circle):
                    pos = body.position
                    pygame.draw.circle(self.screen, part_color,
                                     (int(pos.x), int(pos.y)),
                                     int(shape.radius))
                    # 脸部细节
                    if name == 'head':
                        self._draw_face(stickman, body, shape)
                elif isinstance(shape, pymunk.Poly):
                    vertices = [body.local_to_world(v) for v in shape.get_vertices()]
                    points = [(int(v.x), int(v.y)) for v in vertices]
                    pygame.draw.polygon(self.screen, part_color, points)
                    pygame.draw.polygon(self.screen,
                                       (max(0, part_color[0]-40),
                                        max(0, part_color[1]-40),
                                        max(0, part_color[2]-40)),
                                       points, 1)

        # 绘制剑
        self._draw_sword(stickman)

        # 绘制盾牌
        self._draw_shield(stickman)

        # 绘制连接线（关节）
        self._draw_joints(stickman)

    def _draw_face(self, stickman, head_body, head_shape):
        """绘制脸部表情"""
        pos = head_body.position
        radius = head_shape.radius
        # 眼睛
        eye_offset = radius * 0.4
        eye_size = 3
        direction = 1 if stickman.facing_right else -1
        # 左眼(水平排列)
        pygame.draw.circle(self.screen, (0, 0, 0),
                         (int(pos.x + direction * eye_offset - 5), int(pos.y - 3)), eye_size)
        # 右眼
        pygame.draw.circle(self.screen, (0, 0, 0),
                         (int(pos.x + direction * eye_offset + 5), int(pos.y - 3)), eye_size)

    def _draw_sword(self, stickman):
        """绘制剑"""
        if not stickman.sword_body:
            return

        body = stickman.sword_body
        # 剑身
        tip = body.local_to_world((cfg.SWORD_LENGTH / 2, 0))
        base = body.local_to_world((-cfg.SWORD_LENGTH / 2, 0))

        # 剑身（金属色）
        pygame.draw.line(self.screen, cfg.SWORD_COLOR,
                        (int(base.x), int(base.y)),
                        (int(tip.x), int(tip.y)), cfg.SWORD_WIDTH)

        # 剑刃高光
        highlight = body.local_to_world((cfg.SWORD_LENGTH / 4, -2))
        pygame.draw.line(self.screen, (230, 230, 240),
                        (int(base.x), int(base.y)),
                        (int(highlight.x), int(highlight.y)), 2)

        # 剑格
        guard_left = body.local_to_world((-cfg.SWORD_LENGTH / 2 + 2, -8))
        guard_right = body.local_to_world((-cfg.SWORD_LENGTH / 2 + 2, 8))
        pygame.draw.line(self.screen, (180, 160, 100),
                        (int(guard_left.x), int(guard_left.y)),
                        (int(guard_right.x), int(guard_right.y)), 4)

        # 剑柄
        handle_end = body.local_to_world((-cfg.SWORD_LENGTH / 2 - 8, 0))
        pygame.draw.line(self.screen, (120, 80, 40),
                        (int(base.x), int(base.y)),
                        (int(handle_end.x), int(handle_end.y)), 4)

    def _draw_shield(self, stickman):
        """绘制盾牌"""
        if not stickman.shield_body:
            return

        body = stickman.shield_body
        shape = stickman.shield_shape

        vertices = [body.local_to_world(v) for v in shape.get_vertices()]
        points = [(int(v.x), int(v.y)) for v in vertices]

        # 盾牌主体
        pygame.draw.polygon(self.screen, cfg.SHIELD_COLOR, points)

        # 盾牌边框
        border_color = (120, 80, 40)
        pygame.draw.polygon(self.screen, border_color, points, 3)

        # 盾牌图案（十字）
        cx, cy = int(body.position.x), int(body.position.y)
        pygame.draw.line(self.screen, (180, 150, 100),
                        (cx - 10, cy), (cx + 10, cy), 2)
        pygame.draw.line(self.screen, (180, 150, 100),
                        (cx, cy - 14), (cx, cy + 14), 2)

    def _draw_joints(self, stickman):
        """绘制关节连接（使火柴人看起来连续）"""
        connections = [
            ('head', 'torso'),
            ('torso', 'left_upper_arm'),
            ('torso', 'right_upper_arm'),
            ('left_upper_arm', 'left_lower_arm'),
            ('right_upper_arm', 'right_lower_arm'),
            ('torso', 'left_upper_leg'),
            ('torso', 'right_upper_leg'),
            ('left_upper_leg', 'left_lower_leg'),
            ('right_upper_leg', 'right_lower_leg'),
        ]

        for name_a, name_b in connections:
            if name_a in stickman.bodies and name_b in stickman.bodies:
                pos_a = stickman.bodies[name_a].position
                pos_b = stickman.bodies[name_b].position
                pygame.draw.line(self.screen, stickman.color,
                               (int(pos_a.x), int(pos_a.y)),
                               (int(pos_b.x), int(pos_b.y)), 3)

    def _draw_particles(self):
        """绘制粒子特效"""
        for p in self.blood_particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            size = int(p['size'] * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            # 由于pygame不支持alpha，直接绘制
            pygame.draw.circle(self.screen, p['color'],
                             (int(p['pos'].x), int(p['pos'].y)), max(1, size))

    def _draw_hit_effects(self):
        """绘制打击特效"""
        for e in self.hit_effects:
            progress = 1 - (e['life'] / e['max_life'])
            size = int(20 * (1 - progress))

            if e['block']:
                # 格挡 - 黄色闪光
                pygame.draw.circle(self.screen, (255, 255, 100),
                                 (int(e['pos'].x), int(e['pos'].y)), size, 3)
            elif e['headshot']:
                # 爆头 - 红色大爆炸
                pygame.draw.circle(self.screen, (255, 50, 50),
                                 (int(e['pos'].x), int(e['pos'].y)), size, 2)
                # 文字
                text = self.small_font.render("爆头!", True, (255, 100, 100))
                self.screen.blit(text,
                               (int(e['pos'].x) - 20, int(e['pos'].y) - size - 20))
            else:
                # 普通打击 - 白色闪光
                pygame.draw.circle(self.screen, (255, 200, 200),
                                 (int(e['pos'].x), int(e['pos'].y)), size, 2)

    def _draw_hud(self):
        """绘制HUD"""
        if not self.player1 or not self.player2:
            return

        # 玩家1信息 (左侧)
        self._draw_health_bar(50, 30, 350, 25, self.player1.health,
                             self.player1.max_health, cfg.PLAYER1_COLOR, "玩家1")

        # 玩家2信息 (右侧)
        self._draw_health_bar(cfg.WINDOW_WIDTH - 400, 30, 350, 25,
                             self.player2.health, self.player2.max_health,
                             cfg.PLAYER2_COLOR, "玩家2")

        # 时间
        time_text = self.font.render(f"时间: {int(self.round_time)}s", True, (255, 255, 255))
        time_rect = time_text.get_rect(center=(cfg.WINDOW_WIDTH // 2, 30))
        self.screen.blit(time_text, time_rect)

        # 操作提示
        if self.round_time < 5:
            hints = [
                "WASD 移动 | 鼠标左键 挥剑 | 鼠标右键 举盾",
                "剑尖速度越快伤害越高 | 盾牌可格挡伤害"
            ]
            for i, hint in enumerate(hints):
                hint_text = self.small_font.render(hint, True, (200, 200, 200))
                hint_rect = hint_text.get_rect(center=(cfg.WINDOW_WIDTH // 2,
                                                       cfg.WINDOW_HEIGHT - 50 + i * 22))
                self.screen.blit(hint_text, hint_rect)

    def _draw_health_bar(self, x, y, width, height, health, max_health, color, label):
        """绘制血条"""
        # 背景
        pygame.draw.rect(self.screen, (60, 60, 60), (x, y, width, height))
        pygame.draw.rect(self.screen, (80, 80, 80), (x, y, width, height), 2)

        # 血量
        if health > 0:
            health_width = int(width * (health / max_health))
            # 渐变色
            if health / max_health > 0.5:
                bar_color = color
            elif health / max_health > 0.25:
                bar_color = (255, 255, 0)
            else:
                bar_color = (255, 0, 0)
            pygame.draw.rect(self.screen, bar_color, (x + 2, y + 2, health_width - 4, height - 4))

        # 标签
        label_text = self.small_font.render(f"{label}: {int(health)}", True, (255, 255, 255))
        self.screen.blit(label_text, (x, y - 20))

    def _draw_game_over(self):
        """绘制游戏结束画面"""
        if not self.winner:
            return

        # 半透明蒙层
        overlay = pygame.Surface((cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        # 胜利文字
        winner_text = f"玩家 {self.winner} 获胜!"
        if self.winner == 1:
            winner_color = cfg.PLAYER1_COLOR
        else:
            winner_color = cfg.PLAYER2_COLOR

        text = self.big_font.render(winner_text, True, winner_color)
        text_rect = text.get_rect(center=(cfg.WINDOW_WIDTH // 2, cfg.WINDOW_HEIGHT // 2 - 30))
        self.screen.blit(text, text_rect)

        # 提示
        restart_text = self.font.render("按 R 重新开始 | 按 ESC 退出", True, (200, 200, 200))
        restart_rect = restart_text.get_rect(center=(cfg.WINDOW_WIDTH // 2,
                                                     cfg.WINDOW_HEIGHT // 2 + 20))
        self.screen.blit(restart_text, restart_rect)

    def close(self):
        """关闭游戏"""
        if self.render:
            pygame.quit()
