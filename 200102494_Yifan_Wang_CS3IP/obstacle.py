import pybullet as p
import random
import math

class DynamicObstacle:
    """
    Scripted "rod" obstacle that wanders in front of the arm to stress-test
    the avoider. State machine: IDLE -> MOVING -> {HOLDING, RETREATING} -> IDLE.
    Movement mode is chosen at random each time it leaves IDLE.

    Tracked position is the obstacle's tip (closer-to-arm end), not the body
    center — the perception layer sees the tip first, so distances make more
    sense in tip coordinates.
    """

    MODE_NAMES = {"LINEAR": "linear", "ZIGZAG": "zigzag", "CIRCULAR": "circular", "RANDOM_WALK": "random-walk"}

    def __init__(self):
        # y-bound is tighter than x/z because the main top-down camera only sees
        # ~+/-0.22 along y at the obstacle's tip distance — anything outside that
        # is invisible and the predictor goes blind. The 0.06 half-width plus a
        # small margin gives us 0.15 here.
        self.bounds = {
            'x': (0.35, 0.65), 'y': (-0.15, 0.15), 'z': (0.15, 0.35)
        }
        self.base_y, self.base_z = 0.0, 0.25
        self.arm_length, self.arm_width = 0.8, 0.06

        half_ext = [self.arm_length/2, self.arm_width, self.arm_width]
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_ext, rgbaColor=[0.8, 0.1, 0.1, 1.0])
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext)

        # Spawn parked at the far edge so the IDLE warm-up doesn't visually
        # cross the workspace before the first move command lands.
        self.current_pos = [self.bounds['x'][1] + self.arm_length/2, self.base_y, self.base_z]
        self.body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision,
                                          baseVisualShapeIndex=visual, basePosition=self.current_pos)

        self.state, self.movement_mode = "IDLE", "LINEAR"
        self.wait_timer, self.target_pos = 0, list(self.current_pos)
        self.base_speed, self.current_speed = 0.001, 0.001
        self.speed_change_timer, self.direction_change_timer = 0, 0
        self.direction_change_interval = random.randint(30, 100)

        self.circle = {'center': [0.6, 0.0, 0.3], 'radius': 0.2, 'angle': 0, 'speed': 0.01}
        self.zigzag = {'amplitude': 0.2, 'frequency': 0.025, 'progress': 0}

        print(f"Dynamic random obstacle loaded (ID: {self.body_id})")

    def get_id(self): return self.body_id
    # Tip position = body center shifted by half the rod length toward the arm.
    def get_position(self): return [self.current_pos[0] - self.arm_length/2, self.current_pos[1], self.current_pos[2]]
    def is_in_work_area(self, threshold=0.8): return self.current_pos[0] - self.arm_length/2 < threshold

    def get_state_info(self):
        pos = self.get_position()
        return {"state": self.state, "mode": self.movement_mode,
                "tip_x": pos[0], "tip_y": pos[1], "tip_z": pos[2], "in_work_area": self.is_in_work_area()}

    def _rand(self, key): return random.uniform(self.bounds[key][0], self.bounds[key][1])
    def _clamp(self, val, key): return max(self.bounds[key][0], min(self.bounds[key][1], val))
    def _distance(self, p1, p2): return math.sqrt(sum((a-b)**2 for a, b in zip(p1, p2)))

    def _generate_random_target(self):
        return [self._rand('x') + self.arm_length/2, self._rand('y'), self._rand('z')]

    def _move_toward(self, threshold=0.01):
        """Step toward the target. Returns True once we're within threshold."""
        dist = self._distance(self.current_pos, self.target_pos)
        if dist < threshold: return True
        for i in range(3):
            self.current_pos[i] += (self.target_pos[i] - self.current_pos[i]) / dist * self.current_speed
        return False

    def _choose_movement_mode(self):
        # Weighted sampling produces a mixed obstacle-motion distribution with
        # linear motion as the most common mode.
        self.movement_mode = random.choices(
            ["LINEAR", "ZIGZAG", "CIRCULAR", "RANDOM_WALK"],
            weights=[0.3, 0.25, 0.2, 0.25]
        )[0]

        if self.movement_mode == "CIRCULAR":
            self.circle = {
                'center': [random.uniform(0.45, 0.55), random.uniform(-0.05, 0.1), random.uniform(0.22, 0.28)],
                'radius': random.uniform(0.08, 0.12),
                'angle': random.uniform(0, 2 * math.pi),
                'speed': random.uniform(0.005, 0.02)
            }
        elif self.movement_mode == "ZIGZAG":
            self.zigzag = {'amplitude': random.uniform(0.05, 0.12), 'frequency': random.uniform(0.01, 0.025), 'progress': 0}
            self.target_pos = self._generate_random_target()

    def _move_linear(self): return self._move_toward()

    def _move_zigzag(self):
        # Straight-line travel with a sinusoidal y-axis wobble. The predictor
        # treats this as constant velocity + noise, so it's a useful test of
        # how well the KF smooths through the wobble.
        dist = self._distance(self.current_pos, self.target_pos)
        if dist < 0.02: return True

        self.zigzag['progress'] += self.zigzag['frequency']
        lateral = math.sin(self.zigzag['progress']) * self.zigzag['amplitude'] * self.current_speed * 2

        for i in range(3):
            delta = (self.target_pos[i] - self.current_pos[i]) / dist * self.current_speed
            self.current_pos[i] += delta + (lateral if i == 1 else 0)
        self.current_pos[1] = self._clamp(self.current_pos[1], 'y')
        return False

    def _move_circular(self):
        # Parametric circle in xy with a small z bob (3rd harmonic) so the
        # motion isn't purely planar — gives the side-camera height estimator
        # something to work with.
        c = self.circle
        c['angle'] += c['speed']

        target = [
            c['center'][0] + c['radius'] * math.cos(c['angle']) + self.arm_length/2,
            c['center'][1] + c['radius'] * math.sin(c['angle']),
            c['center'][2] + c['radius'] * 0.3 * math.sin(c['angle'] * 2)
        ]
        # Lerp toward the parametric target to keep motion smooth.
        for i in range(3):
            self.current_pos[i] += (target[i] - self.current_pos[i]) * 0.1
        self.current_pos[2] = self._clamp(self.current_pos[2], 'z')
        return False

    def _move_random_walk(self):
        # Pick a fresh target every direction_change_interval steps. Within
        # each leg it's just linear motion, so the predictor can lock on
        # briefly between the abrupt direction changes.
        self.direction_change_timer += 1
        if self.direction_change_timer > self.direction_change_interval:
            self.target_pos = self._generate_random_target()
            self.direction_change_interval = random.randint(30, 100)
            self.direction_change_timer = 0
        return self._move_linear()

    def _update_speed(self):
        # Random speed jitter every 20-60 sim steps. Multiplier in [0.2, 1.5]
        # of base, so the obstacle sometimes crawls and sometimes lunges —
        # exercises the avoider's velocity-aware safety ring expansion.
        self.speed_change_timer += 1
        if self.speed_change_timer > random.randint(20, 60):
            self.current_speed = self.base_speed * random.uniform(0.2, 1.5)
            self.speed_change_timer = 0

    def update(self):
        self._update_speed()

        if self.state == "IDLE":
            self.wait_timer += 1
            if self.wait_timer > random.randint(30, 150):
                self.state, self.wait_timer = "MOVING", 0
                self._choose_movement_mode()
                self.target_pos = self._generate_random_target()
                print(f">>> Obstacle started {self.MODE_NAMES[self.movement_mode]} motion!")

        elif self.state == "MOVING":
            move_funcs = {"LINEAR": self._move_linear, "ZIGZAG": self._move_zigzag,
                          "CIRCULAR": self._move_circular, "RANDOM_WALK": self._move_random_walk}
            reached = move_funcs[self.movement_mode]()

            # Circular and random-walk don't naturally terminate — they'd just
            # loop forever — so we cap them by step count and force a state change.
            if self.movement_mode in ("CIRCULAR", "RANDOM_WALK"):
                self.wait_timer += 1
                limit = random.randint(200, 400) if self.movement_mode == "CIRCULAR" else random.randint(150, 300)
                if self.wait_timer > limit:
                    reached, self.wait_timer = True, 0

            if reached:
                # 30% chance to pause in place, 70% to immediately pick another mode.
                # Pausing tests the "obstacle goes static" path in the avoider.
                if random.random() < 0.3:
                    self.state, self.wait_timer = "HOLDING", 0
                    print("--- Obstacle paused...")
                else:
                    self._choose_movement_mode()
                    self.target_pos = self._generate_random_target()

        elif self.state == "HOLDING":
            self.wait_timer += 1
            if self.wait_timer > random.randint(50, 200):
                self.wait_timer = 0
                # 20% retreat off-stage, 80% jump back into a new movement.
                # Retreating gives the arm a clean window to finish the task.
                if random.random() < 0.2:
                    self.state = "RETREATING"
                    self.target_pos = [self.bounds['x'][1] + self.arm_length/2, self._rand('y'), self.base_z]
                    print("<<< Obstacle moving to the edge...")
                else:
                    self.state = "MOVING"
                    self._choose_movement_mode()
                    self.target_pos = self._generate_random_target()
                    print(">>> Obstacle resumed moving!")

        elif self.state == "RETREATING":
            if self._move_linear():
                self.state = "IDLE"
                print("--- Obstacle reached the edge; entering idle state")

        # Apply hard clamps before setting the body — every once in a while a
        # fast linear move overshoots its target between frames and we'd render
        # the rod outside the camera FOV for one tick.
        self.current_pos[0] = max(self.bounds['x'][0] + self.arm_length/2,
                                   min(self.bounds['x'][1] + self.arm_length/2, self.current_pos[0]))
        self.current_pos[1] = self._clamp(self.current_pos[1], 'y')
        self.current_pos[2] = self._clamp(self.current_pos[2], 'z')
        p.resetBasePositionAndOrientation(self.body_id, self.current_pos, [0,0,0,1])
