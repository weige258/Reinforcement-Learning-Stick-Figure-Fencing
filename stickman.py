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
        body.angular_damping = cfg.BODY_ANGULAR_DAMPING  # 旋转阻尼，防止倒地后疯狂旋转
        shape = pymunk.Poly.create_box(body, size)
        shape.friction = cfg.BODY_FRICTION; shape.elasticity = cfg.BODY_ELASTICITY; shape.collision_type = ct
        shape.filter = pymunk.ShapeFilter(group=self.player_id, categories=0b1, mask=0b11111)
        self.space.add(body, shape)
        self.bodies[name] = body; self.shapes[name] = shape
        return body, shape

    def _add_circle(self, name, mass, radius, pos, ct=1):
        body = pymunk.Body(mass, pymunk.moment_for_circle(mass, 0, radius))
        body.position = pos
        body.angular_damping = cfg.BODY_ANGULAR_DAMPING  # 旋转阻尼
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

    def _stand_spring(self, body, stiffness, damping):
        """站立弹簧: 将身体部位连接到世界坐标系，保持rest_angle=0（直立）
        
        关键区别: DampedRotarySpring 控制的是角度位置(P控制)，
        而非 SimpleMotor 的角速度控制。这样才能主动推回直立角度。
        """
        j = pymunk.DampedRotarySpring(body, self.space.static_body, 0, stiffness, damping)
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
        # 躯干站立: DampedRotarySpring角度伺服(非SimpleMotor速度控制!)
        self._stand_spring(self.bodies['torso'], cfg.STAND_TORSO_STIFFNESS, cfg.STAND_TORSO_DAMPING)

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
        # 大腿通过强弹簧跟随躯干(躯干直立时自然带动大腿回正)
        self._spring(self.bodies['torso'], self.bodies['left_upper_leg'], 0, cfg.LEG_SPRING_STIFFNESS, cfg.LEG_SPRING_DAMPING)

        # ==== 左小腿 ====
        lkx = lhx; lky = lhy + ull/2 + lll/2
        self._add_box('left_lower_leg', cfg.LOWER_LEG_MASS, (8, lll), (lkx, lky))
        self._pivot(self.bodies['left_upper_leg'], self.bodies['left_lower_leg'], (0, ull/2), (0, -lll/2))
        # 小腿通过强力弹簧跟随大腿(大腿站立时自然带动小腿回正)
        self._spring(self.bodies['left_upper_leg'], self.bodies['left_lower_leg'], 0, cfg.CALF_SPRING_STIFFNESS, cfg.CALF_SPRING_DAMPING)

        # ==== 右大腿 ====
        rhx = x + 7; rhy = y + th/2 + ull/2
        self._add_box('right_upper_leg', cfg.UPPER_LEG_MASS, (10, ull), (rhx, rhy))
        self._pivot(self.bodies['torso'], self.bodies['right_upper_leg'], (6, th/2), (0, -ull/2))
        # 大腿通过强弹簧跟随躯干
        self._spring(self.bodies['torso'], self.bodies['right_upper_leg'], 0, cfg.LEG_SPRING_STIFFNESS, cfg.LEG_SPRING_DAMPING)

        # ==== 右小腿 ====
        rkx = rhx; rky = rhy + ull/2 + lll/2
        self._add_box('right_lower_leg', cfg.LOWER_LEG_MASS, (8, lll), (rkx, rky))
        self._pivot(self.bodies['right_upper_leg'], self.bodies['right_lower_leg'], (0, ull/2), (0, -lll/2))
        # 小腿通过强力弹簧跟随大腿(大腿站立时自然带动小腿回正)
        self._spring(self.bodies['right_upper_leg'], self.bodies['right_lower_leg'], 0, cfg.CALF_SPRING_STIFFNESS, cfg.CALF_SPRING_DAMPING)

        # ==== 武器 ====
        self._create_sword(); self._attach_sword_to_hand()
        self._create_shield(); self._attach_shield_to_hand()

    def _create_sword(self):
        hp = self._hand_pos('right')
        self.sword_body = pymunk.Body(cfg.SWORD_MASS, pymunk.moment_for_box(cfg.SWORD_MASS, (cfg.SWORD_LENGTH, cfg.SWORD_WIDTH)))
        self.sword_body.position = hp; self.sword_body.angle = math.radians(-90)
        self.sword_body.angular_damping = cfg.WEAPON_ANGULAR_DAMPING
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
        self.shield_body.angular_damping = cfg.WEAPON_ANGULAR_DAMPING
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

    def _is_on_ground(self):
        """检查是否着地: 任意一只脚接近地面"""
        for leg_name in ['left_lower_leg', 'right_lower_leg']:
            if leg_name in self.bodies:
                foot_y = self.bodies[leg_name].position.y
                foot_bottom = foot_y + cfg.LOWER_LEG_LENGTH / 2
                if foot_bottom >= cfg.GROUND_Y - 15:
                    return True
        # 回退: 躯干不能太高(防止空中二段跳)
        return self.get_position().y > cfg.GROUND_Y - 130

    def apply_movement(self, move_x=0, move_y=0, jump=False, attack=False, block=False):
        torso = self.bodies['torso']
        force = 10000.0

        # 限速
        v = torso.velocity; mx, my = 400, 800
        if abs(v.x) > mx: torso.velocity = Vec2d(math.copysign(mx, v.x), v.y)
        if abs(v.y) > my: torso.velocity = Vec2d(v.x, math.copysign(my, v.y))
        # 辅助扶正：倒地时用强力扭矩翻身，直立时用柔和扭矩
        a = torso.angle
        if abs(a) > cfg.STAND_TILT_THRESHOLD:  # 躯干倾斜超过阈值 = 倒地状态
            torso.torque += -a * cfg.FALL_RECOVERY_TORQUE  # 强力翻身扭矩
        elif abs(a) > 0.05:
            torso.torque += -a * cfg.STAND_CORRECT_TORQUE   # 轻微扶正

        if move_x != 0:
            # 世界坐标系施加力，方向不受躯干旋转影响
            fd = 1 if self.facing_right else -1
            world_force = (move_x * fd * force, 0)
            torso.apply_force_at_world_point(world_force, torso.position)
            for leg in ['left_upper_leg', 'right_upper_leg']:
                self.bodies[leg].apply_force_at_world_point(
                    (move_x * fd * force * 0.2, 0), self.bodies[leg].position)
            # 倒地时额外给躯干扭矩辅助翻身
            if abs(torso.angle) > cfg.STAND_TILT_THRESHOLD:
                torso.torque += -move_x * fd * 200000.0

        # 跳跃: 着地时直接设置向上速度(比impulse更可控)
        if jump and self._is_on_ground():
            torso.velocity = Vec2d(torso.velocity.x, cfg.JUMP_VELOCITY)
            # 也给大腿一个向上的初速度，让整个身体一起跳
            for leg in ['left_upper_leg', 'right_upper_leg']:
                if leg in self.bodies:
                    self.bodies[leg].velocity = Vec2d(
                        self.bodies[leg].velocity.x,
                        min(self.bodies[leg].velocity.y, cfg.JUMP_VELOCITY * 0.6))

        if attack and self.attack_cooldown_timer <= 0:
            self._perform_attack()
        if block: self._perform_block()
        else: self.is_blocking = False
        if self.attack_cooldown_timer > 0:
            self.attack_cooldown_timer -= 1/cfg.FPS

    def _perform_attack(self):
        self.is_attacking = True; self.attack_cooldown_timer = cfg.ATTACK_COOLDOWN
        # 根据剑当前指向计算攻击方向(跟随鼠标)
        if self.sword_body:
            sword_angle = self.sword_body.angle
            dx = math.cos(sword_angle); dy = math.sin(sword_angle)
            self.sword_body.angular_velocity += dx * 35.0
            self.sword_body.torque = dx * cfg.ATTACK_TORQUE * 0.5
            self.sword_body.apply_impulse_at_world_point(
                Vec2d(dx * 300, dy * 150), self.get_sword_tip_position())
            # 手臂也朝剑的方向挥动
            self.bodies['right_upper_arm'].angular_velocity += dx * 10.0
            self.bodies['right_upper_arm'].torque += dx * 80000
        else:
            d = 1 if self.facing_right else -1
            self.sword_body.angular_velocity += d * 35.0
            self.sword_body.torque = d * cfg.ATTACK_TORQUE * 0.5
            self.sword_body.apply_impulse_at_world_point(
                Vec2d(d * 300, -150), self.get_sword_tip_position())
            self.bodies['right_upper_arm'].angular_velocity += d * 10.0
            self.bodies['right_upper_arm'].torque += d * 80000

    def _perform_block(self):
        self.is_blocking = True
        d = 1 if self.facing_right else -1
        if self.shield_body:
            self.shield_body.apply_force_at_world_point(Vec2d(d*4000, -1000), self.shield_body.position)

    def aim_arm_at(self, tx, ty):
        """手臂指向目标点(鼠标), 连续调用保持跟随
        注意: 仅控制手臂指向, 剑身仅在攻击时挥动
        """
        shoulder = self.bodies['right_upper_arm'].position
        angle = math.atan2(ty - shoulder.y, tx - shoulder.x)
        diff = angle - self.bodies['right_upper_arm'].angle
        while diff > math.pi: diff -= 2*math.pi
        while diff < -math.pi: diff += 2*math.pi
        diff = max(min(diff, 0.5), -1.0)
        self.bodies['right_upper_arm'].torque = diff * 200000
        # 手腕跟随: 让前臂也参与瞄准
        if 'right_lower_arm' in self.bodies:
            elbow = self.bodies['right_lower_arm'].position
            elbow_angle = math.atan2(ty - elbow.y, tx - elbow.x)
            ediff = elbow_angle - self.bodies['right_lower_arm'].angle
            while ediff > math.pi: ediff -= 2*math.pi
            while ediff < -math.pi: ediff += 2*math.pi
            ediff = max(min(ediff, 0.5), -1.0)
            self.bodies['right_lower_arm'].torque = ediff * 120000

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
        """镜像归一化状态: 全部相对于自身朝向和位置"""
        p = self.get_position(); h = self.get_head_position()
        fs = 1 if self.facing_right else -1

        rel_x = (enemy_pos.x - p.x) * fs / cfg.WINDOW_WIDTH
        rel_y = (enemy_pos.y - p.y) / cfg.WINDOW_HEIGHT
        dist = min(abs(enemy_pos.x - p.x) / 500.0, 1.0)
        mx = self.bodies['torso'].velocity.x * fs / 300.0
        my = self.bodies['torso'].velocity.y / 300.0

        return [
            rel_x, rel_y, dist,
            mx, my,
            self.health / self.max_health,
            1.0 if self.is_attacking else 0,
            1.0 if self.is_blocking else 0,
            self.bodies['torso'].angle / math.pi,
            self.bodies['right_upper_arm'].angle / math.pi,
            self.bodies['left_upper_arm'].angle / math.pi,
            self.sword_body.angle % (2*math.pi) / (2*math.pi) if self.sword_body else 0,
            self.sword_body.angular_velocity / 15.0 if self.sword_body else 0,
            (self.sword_body.velocity.x * fs) / 300.0 if self.sword_body else 0,
            self.sword_body.velocity.y / 300.0 if self.sword_body else 0,
            h.x * fs / cfg.WINDOW_WIDTH,
            h.y / cfg.WINDOW_HEIGHT,
            float(self.attack_cooldown_timer),
            self.shield_body.angle / math.pi if self.shield_body else 0,
            p.y / cfg.WINDOW_HEIGHT,
            self.bodies['head'].velocity.x * fs / 200.0 if self.bodies.get('head') else 0,
            enemy_pos.x / cfg.WINDOW_WIDTH,
            enemy_pos.y / cfg.WINDOW_HEIGHT,
            float(fs),
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
