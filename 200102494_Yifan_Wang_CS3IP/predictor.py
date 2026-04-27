import numpy as np
import time
from collections import deque

class ObstaclePredictor:
    """
    Constant-velocity Kalman filter on (x, y, z, vx, vy, vz), plus a
    confidence model and a forward-collision threat evaluator.

    State vector:
        [x, y, z, vx, vy, vz]

    The filter assumes constant velocity between observations. The Q matrix
    gives the velocity rows enough process noise to adapt when the obstacle
    changes direction, and the confidence score falls during unstable motion.
    """

    def __init__(self, history_size=15, prediction_horizon=0.5):
        self.history_size = history_size
        self.prediction_horizon = prediction_horizon
        self.position_history = deque(maxlen=history_size)

        # Lazily initialized from the first observation — see update().
        self.state = None
        self.covariance = None
        self.initialized = False

        # PyBullet runs at 240 Hz. tick_dt is the fallback interval; actual
        # perception intervals are measured from timestamps in update().
        self.tick_dt = 1.0 / 240.0
        self.last_update_time = None
        self.dt = self.tick_dt

        # Cached velocity-derived quantities. Recomputed on every update so
        # downstream callers (avoider, threat eval) can read them cheaply.
        self.velocity_estimate = np.zeros(3)
        self.speed_magnitude = 0.0
        self.is_moving = False
        self.movement_direction = None

        self._init_kalman_params()

    def _init_kalman_params(self):
        # H, Q, R don't depend on dt so they're set once. F is rebuilt every
        # update because dt isn't actually constant (see update()).

        # H selects the position rows from the state — we only observe x, y, z,
        # never velocity directly. Velocity is inferred from how position
        # changes between updates.
        self.H = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0]])

        # Q is process noise — how much we expect the constant-velocity model
        # to drift each step. Position rows are 10x tighter than velocity rows
        # because real obstacles do change direction (so velocity SHOULD be
        # adaptable) but they don't teleport (so position SHOULDN'T jump).
        self.Q = np.diag([0.001, 0.001, 0.001, 0.01, 0.01, 0.01])

        # R is sensor noise. ~1 cm std (sqrt(0.01)) matches the per-frame
        # jitter we measured on the depth scan in noisy lighting. Loosen it
        # if the camera ever gets shakier.
        self.R = np.diag([0.01, 0.01, 0.01])

    def _build_F(self, dt):
        # Constant-velocity transition: x_next = x + vx*dt, vx stays put.
        # The off-diagonal dt entries on rows 0-2 are what propagate position
        # forward; the identity on rows 3-5 keeps velocity unchanged.
        return np.array([[1,0,0,dt,0,0], [0,1,0,0,dt,0], [0,0,1,0,0,dt],
                         [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]])

    def update(self, observed_position, timestamp=None):
        # No observation this tick — the perception layer can return None when
        # the obstacle is occluded or out of FOV. Just hold the previous state.
        if observed_position is None:
            return
        obs = np.array(observed_position)
        now = timestamp if timestamp is not None else time.time()
        self.position_history.append(obs.copy())

        # First observation seeds everything: position from the obs, velocity
        # = 0 (we have no history yet), covariance = 0.1*I (loose, so the
        # second update can pull hard toward the new info).
        if not self.initialized:
            self.state = np.concatenate([obs, [0, 0, 0]])
            self.covariance = np.eye(6) * 0.1
            self.last_update_time = now
            self.initialized = True
            return

        # Build F from the measured interval between camera observations.
        # A 1e-4 floor keeps F well-conditioned if two updates share nearly
        # the same timestamp.
        actual_dt = max(now - self.last_update_time, 1e-4) if self.last_update_time else self.tick_dt
        self.last_update_time = now
        self.dt = actual_dt
        F = self._build_F(actual_dt)

        # ---- Standard Kalman: predict, then correct ----

        # 1. Predict where the obstacle "should" be now, using last state and F.
        state_pred = F @ self.state

        # 2. Predicted covariance grows because we propagated forward and the
        #    world has had time to drift. Q is added to express that drift.
        cov_pred = F @ self.covariance @ F.T + self.Q

        # 3. Innovation y: how far the new observation is from where we
        #    predicted. H @ state_pred drops velocity, leaving only the
        #    predicted position to compare against the observed position.
        y = obs - self.H @ state_pred

        # 4. Innovation covariance S: total uncertainty for this comparison =
        #    our predicted error mapped into observation space + sensor noise.
        S = self.H @ cov_pred @ self.H.T + self.R

        # 5. Kalman gain K. Conceptually it's cov_pred / S — the fraction of
        #    the new info we should believe. If S is huge (noisy sensor), K
        #    shrinks and we mostly stay with the prediction. If cov_pred is
        #    huge (we're very uncertain), K grows and we trust the obs.
        K = cov_pred @ self.H.T @ np.linalg.inv(S)

        # 6. Pull the state toward the observation by gain * innovation, and
        #    shrink the covariance because we just gained information.
        self.state = state_pred + K @ y
        self.covariance = (np.eye(6) - K @ self.H) @ cov_pred

        # Cache derived quantities so callers don't have to slice + norm
        # the state every frame.
        self.velocity_estimate = self.state[3:6]
        self.speed_magnitude = np.linalg.norm(self.velocity_estimate)
        # 0.5 mm/s threshold — below this we treat as stationary. Any lower
        # and noise alone keeps is_moving flipping every frame.
        self.is_moving = self.speed_magnitude > 0.0005

        # Direction along the +x axis (which points away from the arm base).
        # Negative vx => obstacle heading toward the arm => "approaching".
        # The 0.0003 dead band prevents the label from flipping on tiny noisy
        # velocities — without it, "approaching"/"leaving" alternates frame-
        # to-frame and the avoider's threat-aware safety distance flickers.
        vx = self.velocity_estimate[0]
        self.movement_direction = 'approaching' if vx < -0.0003 else ('leaving' if vx > 0.0003 else 'stationary')

    def predict_position(self, steps_ahead=None):
        if not self.initialized:
            return None, 0.0
        if steps_ahead is None:
            steps_ahead = int(self.prediction_horizon / self.dt)

        # Linear extrapolation, equivalent to F^steps_ahead @ state for the
        # constant-velocity model but cheaper and easier to read. We use the
        # filter's smoothed velocity, not raw frame deltas.
        pred = self.state[:3] + self.state[3:6] * steps_ahead * self.dt

        # Confidence is the product of three dampeners — any one of them
        # going low knocks the whole prediction down, which is the desired
        # "don't trust this" behavior.

        # data_conf: how full the history buffer is. Cold start (1-2 obs) =>
        # near-zero confidence. Buffer filled => 1.0.
        data_conf = min(len(self.position_history) / self.history_size, 1.0)

        # time_decay: predictions further into the future are exponentially
        # less trustworthy. tau = prediction_horizon, so confidence at the
        # nominal horizon is 1/e (~0.37).
        time_decay = np.exp(-steps_ahead * self.dt / self.prediction_horizon)

        # stability: low when recent observations are jittery. The multiplier
        # maps position-difference variance into a confidence dampener.
        if len(self.position_history) >= 3:
            vels = np.diff(np.array(list(self.position_history)), axis=0)
            stability = 1.0 / (1.0 + np.sum(np.std(vels, axis=0)) * 100) if len(vels) > 1 else 0.5
        else:
            stability = 0.3

        return pred.tolist(), data_conf * time_decay * stability

    def get_avoidance_position(self, current_robot_pos, lead_time_factor=1.5):
        # Returns the position the avoider should treat as "the obstacle".
        # Not necessarily the current obstacle position — could be a confidence-
        # blended forecast of where it'll be by the time the arm reacts.
        if not self.initialized:
            return None, {"status": "not_initialized"}

        pos = self.state[:3]
        info = {"current_pos": pos.tolist(), "velocity": self.velocity_estimate.tolist(),
                "speed": self.speed_magnitude, "is_moving": self.is_moving, "direction": self.movement_direction}

        # Stationary obstacle => no point predicting, just return where it is.
        if not self.is_moving:
            info.update({"status": "stationary", "predicted_pos": pos.tolist()})
            return pos.tolist(), info

        # Asymmetric look-ahead. Approaching obstacles need MORE lead time
        # because they're closing the gap on us; leaving obstacles need LESS
        # because by the time the arm gets there, they've already cleared out.
        # The 1.5x / 0.5x ratio came from collision-rate sweeps — anything
        # less aggressive on "approaching" let the arm walk into fast obstacles.
        base = int(self.prediction_horizon / self.dt)
        mult = {"approaching": 1.5, "leaving": 0.5}.get(self.movement_direction, 1.0)
        steps = int(base * lead_time_factor * mult)
        info["strategy"] = {"approaching": "aggressive", "leaving": "relaxed"}.get(self.movement_direction, "normal")

        pred, conf = self.predict_position(steps)
        info.update({"predicted_pos": pred, "confidence": conf, "status": "predicted"})

        # Blend current position with prediction by confidence. Low confidence
        # keeps the effective obstacle near the current observation; higher
        # confidence moves it toward the forecast. The cap keeps a minimum
        # weight on the latest observation.
        alpha = conf if conf < 0.3 else min(conf * 1.2, 0.9)
        eff = [pos[i] * (1 - alpha) + pred[i] * alpha for i in range(3)]
        info.update({"effective_pos": eff, "blend_alpha": alpha})

        return eff, info

    def should_preemptive_avoid(self, robot_pos, robot_target, safety_margin=0.15):
        # Looks ahead in time and asks: "if the arm walks toward its goal
        # and the obstacle keeps doing what it's doing, do their paths come
        # within safety_margin?" Returns a threat tier so the controller can
        # bump up its safety distance before things get tight.

        if not self.initialized or not self.is_moving:
            return False, 0.0, "proceed_normal"

        r_pos, r_tgt = np.array(robot_pos), np.array(robot_target)
        r_dir = r_tgt - r_pos
        r_dist = np.linalg.norm(r_dir)
        if r_dist < 0.01:
            return False, 0.0, "at_target"
        r_dir = r_dir / r_dist

        # Conservative arm speed estimate. The IK loop can hit ~1 m/s peak
        # but maintains nowhere near that in practice — 0.5 m/s tracks our
        # typical effective rate during a transport leg. Underestimating
        # here makes the threat eval more aggressive, which is the safer side.
        robot_speed = 0.5
        max_threat = 0.0

        # Sweep the look-ahead window in 100ms increments. For each future
        # moment t, predict where the obstacle will be AND where the arm will
        # be (linear motion at robot_speed along r_dir, capped at the goal),
        # then measure the gap.
        for t_sec in np.arange(0.1, 1.0, 0.1):
            steps = max(int(t_sec / self.dt), 1)
            pred, conf = self.predict_position(steps)
            if not pred:
                continue
            r_future = r_pos + r_dir * min(robot_speed * t_sec, r_dist)
            dist = np.linalg.norm(r_future - np.array(pred))
            if dist < safety_margin:
                # Threat ramps from 0 (at safety_margin) to 1 (at zero gap),
                # weighted by predictor confidence so noisy long-range
                # predictions can't trigger emergency moves on their own.
                threat = (safety_margin - dist) / safety_margin * conf
                max_threat = max(max_threat, threat)

        # Three escalation tiers, tuned by counting collisions vs flinches
        # over the benchmark sweep. Below 0.2 we ignore (too many false
        # positives); above 0.7 we treat as emergency (corresponds roughly
        # to <5cm projected gap at non-trivial confidence).
        if max_threat > 0.7: return True, max_threat, "emergency_avoid"
        if max_threat > 0.4: return True, max_threat, "preemptive_avoid"
        if max_threat > 0.2: return True, max_threat, "cautious_proceed"
        return False, max_threat, "proceed_normal"

    def get_motion_trend(self):
        if not self.initialized:
            return {"status": "no_data", "is_moving": False}
        return {"status": "tracking", "is_moving": self.is_moving, "direction": self.movement_direction,
                "velocity": self.velocity_estimate.tolist(), "speed": self.speed_magnitude,
                "position": self.state[:3].tolist()}
