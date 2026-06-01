"""
火柴人物理 - 躯干主体 + 头紧贴 + 双段四肢 + 轻柔弹簧
"""
import math
import pymunk
from pymunk import Vec2d
import game_config as cfg


class StickMan:

    def __init__(self, space, x, y, facing_right=True, player_id=1):
        self.space = space
        self.player_id = player_id
        self.facing_right = facing_right
        self.health = cfg.MAX_HEALTH
        self.max_health = cfg.MAX_HEALTH
        self.COLL_BODY = 1; self.COLL_SWORD = 2; self.COLL_SHIELD = 3
        self.COLL_GROUND = 4; self.COLL_HEAD = 5
        self.bodies = {}
        self.shapes = {}
        self.joints = []
        self.sword_body = None; self.sword_shape = None
        self.shield_body = None; self.shield_shape = None
        self.is_attacking = False; self.attack_cooldown_timer = 0
        self.is_blocking = False
        self._build_body(x, y)
        self.color = cfg.PLAYER1_COLOR if player_id == 1 else cfg.PLAYER2_COLOR

    def _add_box(self, name, mass, size, pos, angle=0, ct=1):
        body = pymunk.Body(mass, pymunk.moment_for_box(mass, size))
        body.position = pos; body.angle = angle
        shape = pymunk.Poly.create_box(body, size)
        shape.friction = 0.8; shape.elasticity = 0.1; shape.collision_type = ct
        shape.filter = pymunk.ShapeFilter(group=self.player_id, categories=0b1, mask=0b11111)
        self.space.add(body, shape)
        self.bodies[name] = body; self.shapes[name] = shape
        return body, shape

    def _add_circle(self, name, mass, radius, pos, ct=1):
        body = pymunk.Body(mass, pymunk.moment_for_circle(mass, 0, radius))
        body.position = pos
        shape = pymunk.Circle(body, radius)
        shape.friction = 0.5; shape.elasticity = 0.2; shape.collision_type = ct
        shape.filter = pymunk.ShapeFilter(group=self.player_id, categories=0b1, mask=0b11111)
        self.space.add(body, shape)
        self.bodies[name] = body; self.shapes[name] = shape
        return body, shape

    def _pivot(self, ba, bb, a_local, b_local, mf=30000):
        j = pymunk.PivotJoint(ba, bb, a_local, b_local)
        j.max_force = mf; j.error_bias = 0.15
        self.space.add(j); self.joints.append(j)
        return j

    def _pin(self, ba, bb, a_local, b_local):
        j = pymunk.PinJoint(ba, bb, a_local, b_local)
        self.space.add(j); self.joints.append(j)
        return j

    def _spring(self, ba, bb, ra=0, st=6000, da=600):
        j = pymunk.DampedRotarySpring(ba, bb, ra, st, da)
        self.space.add(j); self.joints.append(j)
        return j

    def _build_body(self, x, y):
        f = 1 if self.facing_right else -1
        tw, th = cfg.TORSO_WIDTH, cfg.TORSO_HEIGHT
        ual, lal = cfg.UPPER_ARM_LENGTH, cfg.LOWER_ARM_LENGTH
        ull, lll = cfg.UPPER_LEG_LENGTH, cfg.LOWER_LEG_LENGTH
        hr = cfg.HEAD_RADIUS

        # ==== 躯干 ====
        tp = Vec2d(x, y)
        self._add_box('torso', cfg.TORSO_MASS, (tw, th), tp)
        # 躯干垂直稳定: SimpleMotor保持上身不倾倒
        tm = pymunk.SimpleMotor(self.bodies['torso'], self.space.static_body, 0)
        tm.max_force = 80000; self.space.add(tm); self.joints.append(tm)

        # ==== 头(紧贴,双关节固定) ====
        hy = y - th/2 - hr
        self._add_circle('head', cfg.HEAD_MASS, hr, (x, hy), self.COLL_HEAD)
        self._pivot(self.bodies['head'], self.bodies['torso'], (0, hr), (0, -th/2), 50000)
        self._pin(self.bodies['head'], self.bodies['torso'], (0, hr+2), (0, -th/2-2))

        # ==== 左上臂 ====
        lax = x - f*(tw/2 + ual/2); lay = y - th/3
        self._add_box('left_upper_arm', cfg.UPPER_ARM_MASS, (ual, 8), (lax, lay), math.radians(-15*f))
        self._pivot(self.bodies['torso'], self.bodies['left_upper_arm'], (-f*tw/2, -th/3), (f*ual/2, 0))
        self._spring(self.bodies['torso'], self.bodies['left_upper_arm'], math.radians(-20*f), 6000, 600)

        # ==== 左前臂 ====
        lfx = lax - f*(ual/2 + lal/2); lfy = lay + 8
        self._add_box('left_lower_arm', cfg.LOWER_ARM_MASS, (lal, 7), (lfx, lfy), math.radians(-30*f))
        self._pivot(self.bodies['left_upper_arm'], self.bodies['left_lower_arm'], (-f*ual/2, 0), (f*lal/2, 0))
        self._spring(self.bodies['left_upper_arm'], self.bodies['left_lower_arm'], math.radians(-15*f), 5000, 500)

        # ==== 右上臂(持剑) ====
        rax = x + f*(tw/2 + ual/2); ray = y - th/3
        self._add_box('right_upper_arm', cfg.UPPER_ARM_MASS, (ual, 8), (rax, ray), math.radians(15*f))
        self._pivot(self.bodies['torso'], self.bodies['right_upper_arm'], (f*tw/2, -th/3), (-f*ual/2, 0))
        self._spring(self.bodies['torso'], self.bodies['right_upper_arm'], math.radians(20*f), 6000, 600)

        # ==== 右前臂(持剑) ====
        rfx = rax + f*(ual/2 + lal/2); rfy = ray + 8
        self._add_box('right_lower_arm', cfg.LOWER_ARM_MASS, (lal, 7), (rfx, rfy), math.radians(30*f))
        self._pivot(self.bodies['right_upper_arm'], self.bodies['right_lower_arm'], (f*ual/2, 0), (-f*lal/2, 0))
        self._spring(self.bodies['right_upper_arm'], self.bodies['right_lower_arm'], math.radians(15*f), 5000, 500)

        # ==== 左大腿 ====
        lhx = x - 7; lhy = y + th/2 + ull/2
        self._add_box('left_upper_leg', cfg.UPPER_LEG_MASS, (10, ull), (lhx, lhy))
        self._pivot(self.bodies['torso'], self.bodies['left_upper_leg'], (-6, th/2), (0, -ull/2))
        self._spring(self.bodies['torso'], self.bodies['left_upper_leg'], 0, 8000, 800)
        # 腿垂直稳定: SimpleMotor强制腿指向地面
        m = pymunk.SimpleMotor(self.bodies['left_upper_leg'], self.space.static_body, 0)
        m.max_force = 40000; self.space.add(m); self.joints.append(m)

        # ==== 左小腿 ====
        lkx = lhx; lky = lhy + ull/2 + lll/2
        self._add_box('left_lower_leg', cfg.LOWER_LEG_MASS, (8, lll), (lkx, lky))
        self._pivot(self.bodies['left_upper_leg'], self.bodies['left_lower_leg'], (0, ull/2), (0, -lll/2))
        self._spring(self.bodies['left_upper_leg'], self.bodies['left_lower_leg'], 0, 6000, 600)

        # ==== 右大腿 ====
        rhx = x + 7; rhy = y + th/2 + ull/2
        self._add_box('right_upper_leg', cfg.UPPER_LEG_MASS, (10, ull), (rhx, rhy))
        self._pivot(self.bodies['torso'], self.bodies['right_upper_leg'], (6, th/2), (0, -ull/2))
        self._spring(self.bodies['torso'], self.bodies['right_upper_leg'], 0, 8000, 800)
        m = pymunk.SimpleMotor(self.bodies['right_upper_leg'], self.space.static_body, 0)
        m.max_force = 40000; self.space.add(m); self.joints.append(m)

        # ==== 右小腿 ====
        rkx = rhx; rky = rhy + ull/2 + lll/2
        self._add_box('right_lower_leg', cfg.LOWER_LEG_MASS, (8, lll), (rkx, rky))
        self._pivot(self.bodies['right_upper_leg'], self.bodies['right_lower_leg'], (0, ull/2), (0, -lll/2))
        self._spring(self.bodies['right_upper_leg'], self.bodies['right_lower_leg'], 0, 6000, 600)

        # ==== 武器 ====
        self._create_sword(); self._attach_sword_to_hand()
        self._create_shield(); self._attach_shield_to_hand()

    def _create_sword(self):
        hp = self._hand_pos('right')
        self.sword_body = pymunk.Body(cfg.SWORD_MASS, pymunk.moment_for_box(cfg.SWORD_MASS, (cfg.SWORD_LENGTH, cfg.SWORD_WIDTH)))
        self.sword_body.position = hp; self.sword_body.angle = math.radians(-90)
        self.sword_shape = pymunk.Poly.create_box(self.sword_body, (cfg.SWORD_LENGTH, cfg.SWORD_WIDTH))
        self.sword_shape.friction = 0.7; self.sword_shape.elasticity = 0.05
        self.sword_shape.collision_type = self.COLL_SWORD
        self.sword_shape.filter = pymunk.ShapeFilter(group=self.player_id, categories=0b10, mask=0b11111)
        self.sword_shape.stickman = self
        self.space.add(self.sword_body, self.sword_shape)

    def _attach_sword_to_hand(self):
        hb = self.bodies['right_lower_arm']
        f = 1 if self.facing_right else -1
        off = -f * cfg.LOWER_ARM_LENGTH/2
        # 剑柄(后端)连接手, 剑尖在前
        self._pivot(hb, self.sword_body, (off-3, 0), (-cfg.SWORD_LENGTH/2+5, 0), 15000)
        self._pin(hb, self.sword_body, (off-3, 0), (-cfg.SWORD_LENGTH/2+5, 0))

    def _create_shield(self):
        hp = self._hand_pos('left')
        self.shield_body = pymunk.Body(cfg.SHIELD_MASS, pymunk.moment_for_box(cfg.SHIELD_MASS, (cfg.SHIELD_WIDTH, cfg.SHIELD_HEIGHT)))
        self.shield_body.position = hp
        self.shield_shape = pymunk.Poly.create_box(self.shield_body, (cfg.SHIELD_WIDTH, cfg.SHIELD_HEIGHT))
        self.shield_shape.friction = 0.8; self.shield_shape.elasticity = 0.1
        self.shield_shape.collision_type = self.COLL_SHIELD
        self.shield_shape.filter = pymunk.ShapeFilter(group=self.player_id, categories=0b100, mask=0b11111)
        self.shield_shape.stickman = self
        self.space.add(self.shield_body, self.shield_shape)

    def _attach_shield_to_hand(self):
        hb = self.bodies['left_lower_arm']
        f = 1 if self.facing_right else -1
        off = f * cfg.LOWER_ARM_LENGTH/2
        self._pivot(hb, self.shield_body, (off+3, 0), (-cfg.SHIELD_WIDTH/2, 0), 15000)
        self._pin(hb, self.shield_body, (off+3, 0), (-cfg.SHIELD_WIDTH/2, 0))

    def _hand_pos(self, side='right'):
        n = f'{side}_lower_arm'
        if n in self.bodies:
            f = 1 if self.facing_right else -1
            return self.bodies[n].position + Vec2d(f*(cfg.LOWER_ARM_LENGTH/2+5), 0)
        return self.get_position()

    def get_position(self): return self.bodies['torso'].position
    def get_head_position(self): return self.bodies['head'].position

    def get_sword_tip_position(self):
        if self.sword_body: return self.sword_body.local_to_world((cfg.SWORD_LENGTH/2, 0))
        return self.get_position()

    def apply_movement(self, move_x=0, move_y=0, jump=False, attack=False, block=False):
        torso = self.bodies['torso']
        force = 10000.0

        # 限速
        v = torso.velocity; mx, my = 400, 500
        if abs(v.x) > mx: torso.velocity = Vec2d(math.copysign(mx, v.x), v.y)
        if abs(v.y) > my: torso.velocity = Vec2d(v.x, math.copysign(my, v.y))
        # 辅助扶正(SimpleMotor控角速度, 扭矩控角度)
        a = torso.angle
        if abs(a) > 0.1: torso.torque += -a * 50000.0

        if move_x != 0:
            torso.apply_force_at_local_point((move_x*force, 0), (0, 0))
            for leg in ['left_upper_leg', 'right_upper_leg']:
                self.bodies[leg].apply_force_at_local_point((move_x*force*0.2, 0), (0, 0))

        # 跳跃: 仅当躯干在地面附近时才能跳
        if jump and torso.position.y > cfg.GROUND_Y - 80:
            torso.apply_impulse_at_local_point((0, -25000), (0, 0))

        if attack and self.attack_cooldown_timer <= 0:
            self._perform_attack()
        if block: self._perform_block()
        else: self.is_blocking = False
        if self.attack_cooldown_timer > 0:
            self.attack_cooldown_timer -= 1/cfg.FPS

    def _perform_attack(self):
        self.is_attacking = True; self.attack_cooldown_timer = cfg.ATTACK_COOLDOWN
        d = 1 if self.facing_right else -1
        self.sword_body.angular_velocity += d*35.0
        self.sword_body.torque = d*cfg.ATTACK_TORQUE*0.5
        self.sword_body.apply_impulse_at_world_point(Vec2d(d*300, -150), self.get_sword_tip_position())
        self.bodies['right_upper_arm'].angular_velocity += d*10.0
        self.bodies['right_upper_arm'].torque += d*50000

    def _perform_block(self):
        self.is_blocking = True
        d = 1 if self.facing_right else -1
        if self.shield_body:
            self.shield_body.apply_force_at_world_point(Vec2d(d*4000, -1000), self.shield_body.position)

    def aim_arm_at(self, tx, ty):
        shoulder = self.bodies['right_upper_arm'].position
        angle = math.atan2(ty-shoulder.y, tx-shoulder.x)
        diff = angle - self.bodies['right_upper_arm'].angle
        diff = max(min(diff, 0.5), -1.0)
        self.bodies['right_upper_arm'].torque = diff * 40000

    def take_damage(self, damage, hit_point=None):
        if self.is_blocking: damage *= (1-cfg.SHIELD_BLOCK_DAMAGE_REDUCTION)
        if hit_point and self._is_head_hit(hit_point): damage *= cfg.HEAD_DAMAGE_MULTIPLIER
        self.health -= damage
        if self.health < 0: self.health = 0
        return self.health <= 0

    def _is_head_hit(self, hp):
        return self.bodies['head'].position.get_distance(hp) < cfg.HEAD_RADIUS*2

    def is_alive(self): return self.health > 0
    def get_health_ratio(self): return self.health/self.max_health

    def get_sword_tip_velocity(self):
        if self.sword_body:
            return self.sword_body.velocity_at_world_point(self.get_sword_tip_position())
        return Vec2d(0, 0)

    def get_state_vector(self, enemy_pos):
        p = self.get_position(); h = self.get_head_position()
        er = enemy_pos - p; d = min(er.get_distance((0,0))/500.0, 1.0)
        return [
            p.x/cfg.WINDOW_WIDTH, p.y/cfg.WINDOW_HEIGHT,
            self.bodies['torso'].velocity.x/300, self.bodies['torso'].velocity.y/300,
            er.x/cfg.WINDOW_WIDTH, er.y/cfg.WINDOW_HEIGHT, d,
            self.health/self.max_health,
            enemy_pos.x/cfg.WINDOW_WIDTH, enemy_pos.y/cfg.WINDOW_HEIGHT,
            self.sword_body.angle%(2*math.pi)/(2*math.pi) if self.sword_body else 0,
            self.sword_body.angular_velocity/15 if self.sword_body else 0,
            h.x/cfg.WINDOW_WIDTH, h.y/cfg.WINDOW_HEIGHT,
            self.bodies['head'].velocity.x/200, self.bodies['head'].velocity.y/200,
            1.0 if self.is_attacking else 0, 1.0 if self.is_blocking else 0,
            self.bodies['torso'].angle/math.pi,
            self.bodies['right_upper_arm'].angle/math.pi,
            self.bodies['left_upper_arm'].angle/math.pi,
            self.shield_body.angle/math.pi if self.shield_body else 0,
            self.sword_body.velocity.x/300 if self.sword_body else 0,
            self.sword_body.velocity.y/300 if self.sword_body else 0,
        ]

    def remove(self):
        for sn, s in list(self.shapes.items()):
            b = self.bodies.get(sn)
            if s in self.space.shapes: self.space.remove(s)
            if b and b in self.space.bodies: self.space.remove(b)
        for j in self.joints:
            if j in self.space.constraints: self.space.remove(j)
        if self.sword_shape and self.sword_shape in self.space.shapes: self.space.remove(self.sword_shape)
        if self.sword_body and self.sword_body in self.space.bodies: self.space.remove(self.sword_body)
        if self.shield_shape and self.shield_shape in self.space.shapes: self.space.remove(self.shield_shape)
        if self.shield_body and self.shield_body in self.space.bodies: self.space.remove(self.shield_body)
        self.bodies.clear(); self.shapes.clear(); self.joints.clear()
