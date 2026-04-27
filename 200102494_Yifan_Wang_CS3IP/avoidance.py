import numpy as np

class VisualAvoidanceSystem:
    """
    Modified Artificial Potential Field with vertical-clearance support and
    payload geometry awareness.

    Each control cycle the controller calls compute_modified_step(curr, targ,
    obs) and gets back one small movement step toward the target. The step is
    shaped by local repulsion, vertical clearance, and lateral sidestep terms.

    The mode cascade handles clearance and overhead-crossing states first, then
    applies the local potential-field update near the obstacle.
    """

    def __init__(self, safe_distance=0.45, stop_distance=0.15):
        # Two thresholds, named to match the paper's notation:
        #   d_th2 = warning ring  -> start bending the path here
        #   d_th1 = stop ring     -> never get closer than this
        # Risk smoothly interpolates between them; below d_th1 risk saturates
        # at 1.0 and the avoider switches to maximum-effort lateral + upward push.
        self.d_th2 = safe_distance
        self.d_th1 = stop_distance

        # Obstacle motion state (set externally from the predictor each cycle).
        # We keep it nullable so the avoider can fall back to "static obstacle"
        # behavior if the predictor isn't ready or has been reset.
        self.obstacle_velocity = np.array([0.0, 0.0, 0.0])
        self.obstacle_is_moving = False
        self.obstacle_direction = None
        self.obstacle_height_info = {"max_height": 0.0, "clearance_height": 0.15, "confidence": 0.0}

        # When the gripper is holding something, the EEF point isn't actually
        # the closest collidable geometry anymore — the cube extends below
        # (grabbed_bottom_extend) and outward (grabbed_radius_extend). These
        # get folded into distance and clearance checks so we don't think we're
        # safe just because the EEF point is.
        self.grabbed_radius_extend = 0.0
        self.grabbed_bottom_extend = 0.0

    def set_grabbed_object_bounds(self, radius_extend=0.0, bottom_extend=0.0):
        self.grabbed_radius_extend = radius_extend
        self.grabbed_bottom_extend = bottom_extend

    def set_obstacle_height_info(self, info):
        if info: self.obstacle_height_info.update(info)

    def set_obstacle_motion(self, velocity, is_moving, direction):
        self.obstacle_velocity = np.array(velocity) if velocity else np.array([0, 0, 0])
        self.obstacle_is_moving, self.obstacle_direction = is_moving, direction

    def _get_clearance_height(self, obs_z):
        # Clearance height = minimum z we need to fly at to safely cross the
        # obstacle. The side camera estimates this from the obstacle's actual
        # silhouette; if that estimate isn't confident (cold start, occlusion),
        # we fall back to "10cm above whatever z the main camera reports".
        # 0.3 confidence cutoff matches what the perception layer flags as
        # "borderline" — below that the height read tends to underestimate
        # because we only see the top edge of the obstacle.
        if self.obstacle_height_info.get("confidence", 0) < 0.3:
            return obs_z + 0.10
        return self.obstacle_height_info.get("clearance_height", 0.15)

    def _compute_overhead_move(self, curr, targ, obs, v_dir, h_above_obs, vert_diff, dist):
        # Entered when we're already above the clearance line. The job is to
        # cruise across horizontally and then descend on the far side, without
        # clipping the obstacle's top edge or the corners of its silhouette.

        h_dist = np.linalg.norm(curr[:2] - obs[:2])

        # Project target direction onto the XY plane so cruise motion is
        # purely horizontal — descent is decided separately below.
        move_dir = np.array([v_dir[0], v_dir[1], 0])

        # Degenerate case: target is directly above/below us, so the XY
        # component is near zero. The original 3D direction preserves the
        # vertical component for the next step.
        if np.linalg.norm(move_dir) < 0.01:
            return (curr + v_dir * 0.05).tolist(), "OVERHEAD_VERTICAL"
        move_dir = move_dir / np.linalg.norm(move_dir)

        # If we're horizontally close (<15cm) to the obstacle's projected
        # position, bend toward one side so we don't graze the corner. There
        # are two perpendiculars to "toward the obstacle"; pick whichever
        # has a higher dot product with our travel direction (= the side
        # that's roughly toward where we want to end up).
        if h_dist < 0.15:
            to_obs = obs[:2] - curr[:2]
            if np.linalg.norm(to_obs) > 0.01:
                to_obs = to_obs / np.linalg.norm(to_obs)
                bl, br = np.array([-to_obs[1], to_obs[0], 0]), np.array([to_obs[1], -to_obs[0], 0])
                bypass = bl if np.dot(bl[:2], v_dir[:2]) > np.dot(br[:2], v_dir[:2]) else br
                # 50/50 blend: full bypass overshoots the obstacle by half its
                # width and wastes time recovering; no bypass clips the corner.
                move_dir = (bypass * 0.5 + move_dir * 0.5)
                move_dir = move_dir / np.linalg.norm(move_dir)

        # Allow descent only if it's safe to start dropping. Tiered by
        # horizontal distance to the obstacle and vertical clearance:
        #   - far away (h_dist > 25cm)         -> drop fast
        #   - mid range with comfy margin       -> drop medium
        #   - directly over with tight margin   -> drop very slow
        #   - too close + tight margin          -> don't drop yet
        # vert_diff is signed (target_z - curr_z); we only descend when negative.
        if vert_diff < -0.05:
            if h_dist > 0.25: rate = max(vert_diff / dist, -0.3)
            elif h_dist > 0.15 and h_above_obs > 0.12: rate = max(vert_diff / dist, -0.15)
            elif h_above_obs > 0.15: rate = max(vert_diff / dist, -0.08)
            else: rate = 0
            if rate != 0:
                move_dir[2] = rate
                move_dir = move_dir / np.linalg.norm(move_dir)

        return (curr + move_dir * 0.05).tolist(), "OVERHEAD_CROSS"

    def _compute_escape_direction(self, v_dir, risk):
        # Sidestep force perpendicular to the obstacle's velocity. The idea:
        # don't bounce backward along the obstacle's path (that's where it's
        # ABOUT to be) — step out of the lane it's sweeping through.

        if not self.obstacle_is_moving or np.linalg.norm(self.obstacle_velocity) < 0.0003:
            return np.array([0, 0, 0])

        # Project obstacle velocity onto XY — vertical sidesteps don't make
        # sense (the obstacle's tip can wobble in z without sweeping a "lane").
        obs_vel_h = np.array([self.obstacle_velocity[0], self.obstacle_velocity[1], 0])
        v_norm = np.linalg.norm(obs_vel_h)
        if v_norm < 0.0001:
            return np.array([0, 0, 0])

        # The two perpendiculars to obs_dir. Pick the one whose horizontal
        # component is more aligned with where we want to go, so we step
        # AWAY from the lane AND toward our target.
        obs_dir = obs_vel_h / v_norm
        el, er = np.array([-obs_dir[1], obs_dir[0], 0]), np.array([obs_dir[1], -obs_dir[0], 0])
        target_h = np.array([v_dir[0], v_dir[1], 0])
        escape = el if np.dot(el, target_h) > np.dot(er, target_h) else er

        # Strength scales with both speed and risk:
        #   - Slow obstacle far away  -> tiny escape force (don't bother)
        #   - Fast obstacle close in  -> strong sidestep
        # The 200x velocity multiplier came from tuning — most obstacles cap
        # near 0.005 m/s so 200x lands somewhere reasonable. The 0.5 cap
        # prevents berserk sidesteps when the obstacle is unusually fast.
        return escape * min(v_norm * 200, 0.5) * risk * 1.5

    def compute_modified_step(self, current_pos, target_pos, obstacle_pos):
        # Main entry. Returns (next_position, status_string). Called every
        # control cycle — keep it fast, no allocations in the hot path.

        curr, targ, obs = np.array(current_pos), np.array(target_pos), np.array(obstacle_pos)

        v_full = targ - curr            # vector to goal
        dist_targ = np.linalg.norm(v_full)
        c_vec = obs - curr              # vector to obstacle
        dist_obs_raw = np.linalg.norm(c_vec)

        # Payload-aware effective distance: shrink the apparent distance by
        # the cube's horizontal half-extent so the danger ring effectively
        # wraps the cube edge, not the EEF point. The 0.01 floor is a guard
        # for divide-by-zero downstream — we never actually hit it because
        # set_grabbed_object_bounds clamps inputs.
        dist_obs = max(dist_obs_raw - self.grabbed_radius_extend, 0.01)

        # ---- Cascade of fast exits, ordered cheapest-check first ----

        # Already at target — nothing to do.
        if dist_targ < 0.02: return current_pos, "ARRIVED"
        v_dir = v_full / dist_targ
        c_hat = c_vec / dist_obs_raw if dist_obs_raw > 0.001 else np.array([1, 0, 0])

        # Docking phase: small step, no avoidance gymnastics. We need fine
        # control here for clean placement; a 5cm step would overshoot the
        # tray by half its width.
        if dist_targ < 0.18: return (curr + v_dir * 0.04).tolist(), "DOCKING"

        # Obstacle far enough away that no force adjustment matters. 0.60m
        # is comfortably outside d_th2 (0.40-0.55) plus any plausible payload
        # radius. Above this we take the straight-line path.
        if dist_obs > 0.60: return (curr + v_dir * 0.05).tolist(), "CLEAR_PATH"

        # ---- Vertical reasoning: are we above, level with, or below it? ----
        eff_clear = self._get_clearance_height(obs[2])

        # Effective lowest z of our payload in world coords. The cube hangs
        # below the EEF by grabbed_bottom_extend, so for any "are we above
        # the obstacle?" check we use this lowered z, NOT curr[2].
        effective_eef_z = curr[2] - self.grabbed_bottom_extend
        h_above_clear = effective_eef_z - eff_clear   # >0 means safely above clearance line
        h_above_obs = effective_eef_z - obs[2]        # >0 means above the obstacle's top
        vert_diff = targ[2] - curr[2]
        horiz_diff = np.linalg.norm(targ[:2] - curr[:2])

        # Cleanly above the clearance line — switch to overhead-cross logic.
        # That handles its own bypass + descent decisions internally.
        if h_above_clear > 0.02:
            return self._compute_overhead_move(curr, targ, obs, v_dir, h_above_obs, vert_diff, dist_targ)

        # Lift mode. Two triggers:
        #   1. We're below clearance AND inside the warning ring (need to climb
        #      to clear it before the avoidance step pushes us sideways into
        #      the obstacle's footprint).
        #   2. The target's vertical offset is large relative to its horizontal
        #      offset, so the step should allocate energy to climbing.
        # The (h_dist < 0.25) inside is what disambiguates "side-step instead"
        # — too close to the obstacle to side-step cleanly, so just lift.
        if (h_above_clear < 0 and dist_obs < self.d_th2) or (vert_diff > 0.05 and vert_diff > horiz_diff * 0.5):
            h_dist = np.linalg.norm(curr[:2] - obs[:2])
            deficit = eff_clear - curr[2]
            if deficit > 0 or h_dist < 0.25:
                # up_w = how much of this step's energy goes into vertical lift.
                # When we have a deficit (still below clearance), weight by
                # how big it is, capped at 1.0 (full vertical). Otherwise 0.3
                # baseline — a small upward bias to break ties when target
                # is mostly horizontal but slightly higher.
                up_w = min(deficit / 0.1, 1.0) if deficit > 0 else 0.3
                lift = np.array([0, 0, up_w]) + v_dir * (1.0 - up_w * 0.7)
                if np.linalg.norm(lift) > 0.01: lift = lift / np.linalg.norm(lift)
                # Smaller step (4cm) — bigger steps in lift mode tend to
                # overshoot vertically and oscillate around the clearance line.
                return (curr + lift * 0.04).tolist(), f"LIFTING_TO_{eff_clear:.2f}"

        # Transport-far mode: we're at obstacle z (not above clearance, but
        # not below it by much) and laterally distant. Just fly horizontally;
        # no point burning steps on z corrections we'd undo at the destination.
        if 0 < h_above_obs <= 0.02 and np.linalg.norm(curr[:2] - obs[:2]) > 0.25:
            m = np.array([v_dir[0], v_dir[1], 0])
            if np.linalg.norm(m) > 0.01:
                return (curr + m / np.linalg.norm(m) * 0.05).tolist(), "TRANSPORT_FAR"

        # Heading-away check: if our motion direction makes a >90 degree angle
        # with the direction-to-obstacle (dot < 0), the step is moving out of
        # the danger zone. The -0.1 margin stabilizes nearly tangential motion.
        if np.dot(v_dir, c_hat) < -0.1:
            return (curr + v_dir * 0.05).tolist(), "LEAVING"

        # ---- Lateral avoidance regime ----
        # Inside the danger zone, compute the local potential-field forces.

        # Inflate the safety ring when the obstacle is approaching. The velocity
        # multiplier converts obstacle speed into a distance buffer, and the
        # cap bounds the maximum expansion. Leaving obstacles use a smaller ring.
        eff_safe = self.d_th2
        if self.obstacle_is_moving and self.obstacle_direction == 'approaching':
            eff_safe += min(np.linalg.norm(self.obstacle_velocity) * 50, 0.15)
        elif self.obstacle_is_moving and self.obstacle_direction == 'leaving':
            eff_safe *= 0.85

        # Outside the (inflated) ring? Just go straight.
        if dist_obs > eff_safe:
            return (curr + v_dir * 0.05).tolist(), "NORMAL"

        # Risk: linear interpolation from 0 (at warning ring edge) to 1 (at
        # stop ring edge). Saturated at the boundaries by np.clip.
        risk = np.clip((eff_safe - dist_obs) / (eff_safe - self.d_th1), 0, 1)
        # Approaching obstacles get a 1.3x risk bump — they close the gap on
        # us while we're computing the next step, so we need extra urgency.
        if self.obstacle_is_moving and self.obstacle_direction == 'approaching':
            risk = min(risk * 1.3, 1.0)

        # v_perp: tangent component of "go to target" relative to the
        # obstacle direction. It is computed by subtracting the projection of
        # v_full onto c_hat, leaving the sideways component around the obstacle.
        v_perp = v_full - np.dot(v_full, c_hat) * c_hat

        # How much upward bias to add to v_perp. Tiered by vertical clearance
        # margin so we don't waste energy climbing when we're already above:
        #   plenty above  -> 0.0 (no z bias)
        #   barely above  -> 0.1
        #   below by a hair -> 0.3
        #   below by a lot  -> proportional, capped at 1.0
        if h_above_clear > 0.05: up_s = 0.0
        elif h_above_clear > 0.02: up_s = 0.1
        elif h_above_clear > 0: up_s = 0.3
        else: up_s = min(-h_above_clear / 0.05, 1.0)

        up = np.array([0.0, 0.0, 1.0])
        # Edge case: target is exactly through the obstacle (v_perp ≈ 0),
        # so there's no natural sideways direction to pick. Just shove
        # straight up — we'll re-evaluate next frame from a higher position.
        if np.linalg.norm(v_perp) < 0.1:
            v_perp = up * 2.0 * up_s
        else:
            # Normal case: stack the upward bias on top of the tangent direction.
            v_perp += up * risk * 2.0 * up_s

        # Repulsion: away from the obstacle in the horizontal plane, plus
        # the perpendicular-to-velocity escape term for moving obstacles.
        # c_h is the horizontal component of c_hat (we don't repel
        # vertically — that's what the lift logic above is for).
        c_h = np.array([c_hat[0], c_hat[1], 0])
        c_h = c_h / np.linalg.norm(c_h) if np.linalg.norm(c_h) > 0.01 else np.array([1, 0, 0])
        repel = -risk * c_h * 1.5 + self._compute_escape_direction(v_dir, risk)
        # If the vertical component dominates, dampen lateral repulsion so the
        # combined lift-and-dodge step remains smooth.
        if v_perp[2] > 0.5: repel *= 0.3

        v_mod = v_perp + repel
        # Guard for the "target is up and over the obstacle" case: if v_mod's z
        # is small while the EEF is still below the obstacle, enforce a minimum
        # upward component.
        if vert_diff > 0.05 and v_mod[2] < 0.3 and h_above_obs < 0:
            v_mod[2] = max(v_mod[2], 0.5)

        if np.linalg.norm(v_mod) > 0:
            move = v_mod / np.linalg.norm(v_mod)
            # Smaller step (3cm) at high risk to limit overshoot when the
            # next frame will drastically reweight the forces.
            return (curr + move * (0.03 if risk > 0.5 else 0.05)).tolist(), "AVOIDING"
        # All forces canceled out (rare; usually a numerical edge case).
        # Push straight up by 1cm to break out of the local minimum.
        return (curr + np.array([0, 0, 0.01])).tolist(), "STUCK_RECOVERY"
