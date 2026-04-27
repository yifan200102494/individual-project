import numpy as np
import pybullet as p
import math
import time
import avoidance
import predictor

class CameraSystem:
    """
    Two virtual cameras over the workspace:
      - Main top-down camera at z=0.6, looking straight down. Provides the
        XY position and a coarse z (depth) for AABB construction.
      - Side camera at y=-0.8, looking sideways. Used to estimate the
        obstacle's max height from its side profile.

    Both cameras feed into a shared depth-image -> world-points pipeline
    in _scan_camera. The unprojection math is the standard "NDC -> world"
    via inv(proj @ view); see comments there.
    """

    def __init__(self, robot_id, tray_id):
        self.robot_id, self.tray_id, self.plane_id = robot_id, tray_id, 0
        # IDs we never want to count as obstacles in the seg mask. Populated
        # by set_ignored_object() for the cube once the arm is committed to
        # picking it up — otherwise the cube is "in the way" of itself.
        self.ignored_ids = set()

        # Main camera: top-down, square aspect, 60° FOV. eye=(0.8, 0, 0.6) is
        # offset along +x so we see the workspace from above and slightly in
        # front, which makes the seg mask cleaner around the arm base.
        self.width, self.height = 128, 128
        self.view_matrix, self.proj_matrix, self.inv_pv_mat = self._init_camera(
            [0.8, 0, 0.6], [0.4, 0, 0], 60, 1.0)

        # Side camera: looks across the workspace from -y. Higher resolution
        # (160x120, narrower FOV) because we only care about the vertical
        # extent and we want enough z-resolution to estimate clearance height
        # to ~5cm precision.
        self.side_width, self.side_height = 160, 120
        self.side_view_matrix, self.side_proj_matrix, self.side_inv_pv_mat = self._init_camera(
            [0.5, -0.8, 0.4], [0.5, 0, 0.3], 50, 160/120)

        self.obstacle_height_info = {"max_height": 0.0, "clearance_height": 0.0, "confidence": 0.0}
        # The camera-level predictor keeps a 20-sample history and a 0.8s
        # horizon for threat evaluation.
        self.obstacle_predictor = predictor.ObstaclePredictor(history_size=20, prediction_horizon=0.8)

    def _init_camera(self, eye_pos, target_pos, fov, aspect):
        # Pre-compute and cache view / proj matrices, plus the inverse of
        # their product. We'll use inv_pv every frame to map depth pixels
        # back to world-space points, so doing the inversion once up front
        # saves ~0.3 ms per frame.
        view_mat = p.computeViewMatrix(eye_pos, target_pos, [0, 0, 1])
        proj_mat = p.computeProjectionMatrixFOV(fov, aspect, 0.1, 2.0)

        # PyBullet returns matrices column-major as flat tuples. Reshape into
        # 4x4 row-major numpy and transpose for consistent math conventions.
        view_np = np.array(view_mat).reshape(4, 4).T
        proj_np = np.array(proj_mat).reshape(4, 4).T

        # Use the pseudo-inverse so the cached transform remains stable for
        # near-singular projection configurations.
        return view_mat, proj_mat, np.linalg.pinv(proj_np @ view_np)

    def _scan_camera(self, width, height, view_mat, proj_mat, inv_pv, step=4, z_thresh=0.08):
        # Pull a depth + segmentation render from PyBullet. ER_TINY_RENDERER
        # gives a deterministic CPU-rendered depth/segmentation image.
        img = p.getCameraImage(width, height, view_mat, proj_mat, renderer=p.ER_TINY_RENDERER)
        depth = np.reshape(img[3], (height, width))   # values in [0,1], 0 = near plane, 1 = far plane
        seg = np.reshape(img[4], (height, width))     # body id per pixel; -1 means background

        # Build a boolean mask of "interesting" pixels. We exclude robot
        # links, the tray, the ground plane, and any explicitly ignored ids
        # (e.g., the cube being grasped). depth < 0.95 drops far-plane sky
        # pixels that survived the seg filter.
        mask = (seg != self.robot_id) & (seg != self.tray_id) & (seg != self.plane_id) & (depth < 0.95)
        for ign_id in self.ignored_ids:
            mask &= (seg != ign_id)
        if not np.any(mask):
            return None

        # Subsample masked pixels by `step` to keep the unprojection cost
        # bounded. step=4 on a 128x128 grid gives ~1024 candidate points
        # max, which is plenty for a stable AABB.
        rows, cols = np.where(mask)
        rows, cols = rows[::step], cols[::step]
        depths = depth[rows, cols]

        # ---- Pixel coords -> Normalized Device Coordinates (NDC) ----
        # NDC is the [-1, 1]^3 cube that's the input to inv(proj @ view).
        #
        # x_ndc: cols go [0, width), so 2*col/width is [0, 2), minus 1 maps
        #        to [-1, 1). Standard.
        x_ndc = (2 * cols / width) - 1

        # y_ndc: image rows count DOWN from the top of the screen, but world
        # +y is UP. So we flip: row=0 (top of image) -> y_ndc=+1 (top of NDC).
        # Get this wrong and the AABB shows up mirrored about the camera axis.
        y_ndc = 1 - (2 * rows / height)

        # z_ndc: depth buffer is [0, 1] (near to far), NDC z is [-1, 1].
        z_ndc = 2 * depths - 1

        # ---- NDC -> world via inv_pv (precomputed in _init_camera) ----
        # Stack into homogeneous coords (w=1 means "this is a point, not a
        # direction"). The inv_pv multiplication gives us 4D world coords.
        world = np.dot(inv_pv, np.vstack([x_ndc, y_ndc, z_ndc, np.ones_like(x_ndc)]))

        # Perspective divide: drop back to 3D by dividing the xyz by w.
        # The .T at the end gives us shape (N, 3) — one row per point.
        points = (world[:3] / world[3]).T

        # Filter out anything below z_thresh — usually the table surface
        # and noise specks at floor level. Caller passes a tighter threshold
        # for the side camera (0.05) than the top-down (0.08).
        return points[points[:, 2] > z_thresh]

    def scan_obstacle_volume(self):
        # Top-down scan -> AABB. Returns None if too few points (occlusion,
        # obstacle out of FOV, or fully ignored). The min/max of the point
        # cloud gives us a tight axis-aligned bounding box; the centroid
        # feeds the Kalman filter as the position observation.
        points = self._scan_camera(self.width, self.height, self.view_matrix,
                                   self.proj_matrix, self.inv_pv_mat, step=4, z_thresh=0.08)
        # Need at least 5 points for a stable bbox — fewer than that and a
        # single noisy pixel can blow the AABB out by tens of cm.
        if points is None or len(points) < 5:
            return None

        min_bound, max_bound = np.min(points, axis=0), np.max(points, axis=0)
        center = np.mean(points, axis=0)
        self.draw_debug_box(min_bound, max_bound)
        self.obstacle_predictor.update(center)
        return {"min": min_bound, "max": max_bound, "center": center}

    def scan_obstacle_height_from_side(self):
        # Side scan -> just the height. We don't trust the side camera's xy
        # because perspective from -y compresses x-direction extent.
        points = self._scan_camera(self.side_width, self.side_height, self.side_view_matrix,
                                   self.side_proj_matrix, self.side_inv_pv_mat, step=2, z_thresh=0.05)
        # Too few points -> conservative defaults. 0.15m clearance is what
        # the avoider falls back to when confidence is low (see avoidance.py).
        if points is None or len(points) < 3:
            self.obstacle_height_info = {"max_height": 0.0, "clearance_height": 0.15, "confidence": 0.0}
            return self.obstacle_height_info

        # Use the 95th percentile height to reduce single-pixel outlier impact.
        # Then add 8cm of overhead for sensor noise and payload thickness.
        z = points[:, 2]
        h95 = float(np.percentile(z, 95))
        self.obstacle_height_info = {
            "max_height": float(np.max(z)),
            "clearance_height": h95 + 0.08,
            # Confidence saturates at 100 points — beyond that, more points
            # don't really make us more sure about the height.
            "confidence": min(len(points) / 100.0, 1.0),
        }
        return self.obstacle_height_info

    def get_predicted_obstacle_pos(self, robot_pos):
        return self.obstacle_predictor.get_avoidance_position(robot_pos)

    def get_motion_trend(self):
        return self.obstacle_predictor.get_motion_trend()

    def should_preemptive_avoid(self, robot_pos, robot_target):
        return self.obstacle_predictor.should_preemptive_avoid(robot_pos, robot_target)

    def draw_debug_box(self, min_pos, max_pos):
        # Wireframe AABB rendered as 12 debug lines, lifeTime=0.2s so it
        # auto-clears between scans without needing explicit cleanup.
        p.removeAllUserDebugItems()
        mn, mx = min_pos, max_pos
        corners = [[mn[0],mn[1],mn[2]], [mx[0],mn[1],mn[2]], [mn[0],mx[1],mn[2]], [mx[0],mx[1],mn[2]],
                   [mn[0],mn[1],mx[2]], [mx[0],mn[1],mx[2]], [mn[0],mx[1],mx[2]], [mx[0],mx[1],mx[2]]]
        for s, e in [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)]:
            p.addUserDebugLine(corners[s], corners[e], [0,1,0], lineWidth=2, lifeTime=0.2)


class RobotController:
    def __init__(self, robot_id, tray_id):
        self.robot_id, self.eef_id = robot_id, 11
        # Panda's last two link indices are the gripper fingers. We control
        # them as a pair (symmetric grip) and check both are touching for
        # grasp verification.
        self.finger_indices = [9, 10]
        # 5cm open / 3cm closed. Closed leaves a 6cm finger gap which is
        # tighter than the cube's 5cm width, so closing forces compression.
        self.gripper_open_pos, self.gripper_closed_pos = 0.05, 0.03

        self.vision_system = CameraSystem(robot_id, tray_id)
        # Avoider's d_th2 starts at 0.40 — gets bumped up by the threat eval
        # to 0.48 (preemptive) or 0.55 (emergency) when the predictor flags
        # something. d_th1 (stop ring) stays fixed at 0.15.
        self.avoider = avoidance.VisualAvoidanceSystem(safe_distance=0.40, stop_distance=0.15)
        self.sim_step_callback = None

        # Ablation switch for the paper. When True, all KF-based behavior
        # is disabled: no prediction, no motion direction, no threat-based
        # safety distance bump. The avoider just sees the latest AABB and
        # reacts. Used to isolate the contribution of the predictor.
        self.use_reactive_only = False

        # Geometry of the currently-grasped object (None when empty-handed).
        # Set in set_grabbed_object() after a successful grasp; cleared in
        # clear_grabbed_object() after release.
        self.grabbed_object_size = None         # [half-width, half-depth, half-height]
        self.grabbed_object_offset = 0.0        # how far the cube's bottom hangs below the EEF

        # 200 solver iterations is overkill for normal motion but matters
        # during the grasp — fewer iterations let the cube squirt out of
        # the gripper at high friction.
        p.setPhysicsEngineParameter(numSolverIterations=200, contactBreakingThreshold=0.001)
        # IK joint limits (loose [-7, 7] radians since Panda joints don't
        # actually saturate near these). rp = rest pose / preferred pose
        # for IK to bias toward, matches the home pose set in environmen.py.
        self.ll, self.ul, self.jr = [-7]*7, [7]*7, [7]*7
        self.rp = [0, -math.pi/4, 0, -math.pi/2, 0, math.pi/3, 0]

        # High lateral friction + frictionAnchor on the fingertips. Without
        # frictionAnchor, contact friction is reset every solver step and
        # the cube slowly slides out during long transports.
        for f in self.finger_indices:
            p.changeDynamics(self.robot_id, f, lateralFriction=10.0, frictionAnchor=True)

    def set_ignored_object(self, obj_id):
        self.vision_system.ignored_ids.add(obj_id)

    def set_grabbed_object(self, obj_id):
        # Read the object's actual AABB from the physics engine — the cube
        # might have been scaled or replaced with a different body, so
        # don't hardcode the size.
        aabb_min, aabb_max = p.getAABB(obj_id)
        half_size = [(aabb_max[i] - aabb_min[i]) / 2 for i in range(3)]
        self.grabbed_object_size = half_size
        # +2cm fudge: the gripper grabs the cube around its midline, so the
        # bottom hangs ~half_height below the EEF, plus a bit for finger
        # geometry that the AABB doesn't capture cleanly.
        self.grabbed_object_offset = half_size[2] + 0.02
        print(f"  [Object Volume] size: {[f'{s*2:.3f}' for s in half_size]}m, bottom offset: {self.grabbed_object_offset:.3f}m")

    def clear_grabbed_object(self):
        self.grabbed_object_size = None
        self.grabbed_object_offset = 0.0

    def get_effective_collision_bounds(self):
        # Return the per-axis extensions to feed into the avoider. When
        # holding nothing, both are zero — avoider treats the EEF as a point.
        if self.grabbed_object_size is None:
            return {"radius_extend": 0.0, "bottom_extend": 0.0}

        # Use the larger of width/depth for the horizontal radius — we treat
        # the cube as a vertical cylinder with this radius for distance checks.
        # Slight overestimate for non-cube payloads, which is the safe side.
        radius_extend = max(self.grabbed_object_size[0], self.grabbed_object_size[1])
        bottom_extend = self.grabbed_object_offset

        return {"radius_extend": radius_extend, "bottom_extend": bottom_extend}

    def get_current_eef_pos(self):
        return list(p.getLinkState(self.robot_id, self.eef_id)[4])

    def step_simulation_with_callback(self):
        # Wrapper around p.stepSimulation that runs the user callback and
        # paces to real-time. The callback is how the obstacle gets ticked
        # in lockstep with the physics, see main.py.
        p.stepSimulation()
        if self.sim_step_callback:
            self.sim_step_callback()
        time.sleep(1./240.)

    def move_gripper(self, open_state=True):
        pos = self.gripper_open_pos if open_state else self.gripper_closed_pos
        for i in self.finger_indices:
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, targetPosition=pos, force=500)
        # 20 steps (~83ms) is enough for the fingers to reach their target
        # at force=500 — we don't need to wait for full settle, the next
        # action will give the physics time to finish converging.
        for _ in range(20):
            self.step_simulation_with_callback()

    @staticmethod
    def _point_to_aabb_xy_dist(target_xy, obs_aabb):
        # Standard point-to-AABB distance restricted to XY. Each axis
        # contributes max(0, distance outside the box); axes containing the
        # point contribute 0.
        mn, mx = obs_aabb["min"], obs_aabb["max"]
        dx = max(mn[0] - target_xy[0], 0.0, target_xy[0] - mx[0])
        dy = max(mn[1] - target_xy[1], 0.0, target_xy[1] - mx[1])
        return math.sqrt(dx * dx + dy * dy)

    def _wait_for_safe_path(self, target_xy, safety_radius=0.22, max_wait=8.0, debug=True):
        # Used before any descent (grasp / drop) where the arm has to commit
        # to a vertical column over target_xy. If an obstacle's AABB
        # projection intrudes that column, we lift up first (so the arm is
        # safely above the obstacle), then poll until the obstacle moves
        # away. Times out after max_wait seconds, in which case the caller
        # will probably collide — but better to push through than to deadlock.
        start = time.time()
        step_counter, obs_aabb, retreated = 0, None, False

        while time.time() - start < max_wait:
            # Refresh the camera scan every 24 sim steps (~100ms). Same
            # cadence as move_arm_smart so the perception load is consistent.
            if step_counter % 24 == 0:
                obs_aabb = self.vision_system.scan_obstacle_volume()
                h_info = self.vision_system.scan_obstacle_height_from_side()
                self.avoider.set_obstacle_height_info(h_info)
            step_counter += 1

            if obs_aabb is None:
                if debug: print("  [Path Check] no obstacle, path is safe")
                return True

            h_dist = self._point_to_aabb_xy_dist(target_xy, obs_aabb)
            bounds = self.get_effective_collision_bounds()
            # eff_radius = the avoidance radius PLUS the held cube's horizontal
            # extension. So a 5cm cube + 22cm safety = 27cm effective.
            eff_radius = safety_radius + bounds["radius_extend"]

            if h_dist > eff_radius:
                if debug: print(f"  [Path Check] obstacle AABB is {h_dist:.3f}m from the path > {eff_radius:.3f}m, safe")
                return True

            # Special case: static obstacle outside the AABB (h_dist > 0)
            # is safe even if it's inside the safety_radius ring. The ring
            # exists to give us reaction time against MOVING obstacles —
            # static ones can't suddenly close the gap, so we don't need
            # the buffer.
            trend = self.vision_system.get_motion_trend()
            if not trend.get("is_moving", False) and h_dist > 0.0:
                if debug: print(f"  [Path Check] obstacle is static, descent column is outside the AABB (h={h_dist:.3f}m), proceeding")
                return True

            # The path is blocked. Two-stage response:
            #   Stage 1 (first time through): lift to 25cm above the
            #   obstacle's top, so we're physically out of the danger zone
            #   while waiting.
            #   Stage 2 (subsequent loops): just keep the simulation ticking
            #   so the obstacle gets a chance to move. Don't keep relifting.
            if not retreated:
                cur = self.get_current_eef_pos()
                safe_z = max(cur[2], float(obs_aabb["max"][2]) + 0.25)
                if debug: print(f"  [Path Check] obstacle AABB intruding (h={h_dist:.3f}m<{eff_radius:.3f}m), lifting to avoid at z={safe_z:.3f}")
                self.move_arm_smart([cur[0], cur[1], safe_z], timeout=4.0, debug=False)
                retreated = True
            else:
                if debug and step_counter % 60 == 0:
                    print(f"  [Path Check] already retreated, waiting for the obstacle to move away (h={h_dist:.3f}m)...")
                self.step_simulation_with_callback()

        if debug: print("  [Path Check] timed out, forcing continuation")
        return False

    def _get_effective_obstacle_pos(self, obs_aabb, current_eef_pos):
        # Returns the single point that the avoider should treat as "the
        # obstacle position" this frame: the closest point on the obstacle
        # AABB to the EEF.
        #
        # Because the avoider re-runs every control cycle, this point
        # naturally slides along the box surface as the EEF moves — and
        # the cumulative behavior is equivalent to "avoid the whole box",
        # but with a single position input.
        #
        # When the predictor is active, we also shift the AABB by the
        # predicted obstacle motion before computing the closest point.
        if obs_aabb is None:
            return [10.0, 10.0, 10.0], {"status": "no_obstacle"}

        eef = np.array(current_eef_pos)
        aabb_min = np.array(obs_aabb["min"])
        aabb_max = np.array(obs_aabb["max"])
        aabb_center = np.array(obs_aabb["center"])

        # Reactive mode (ablation): no KF, just clamp the EEF to the AABB.
        # If EEF is inside the box (clamp == EEF), fall back to the center
        # so we have a non-degenerate direction for repulsion.
        if self.use_reactive_only:
            closest = np.maximum(aabb_min, np.minimum(eef, aabb_max))
            if np.linalg.norm(closest - eef) < 0.01:
                closest = aabb_center
            return closest.tolist(), {"status": "reactive"}

        predicted_pos, pred_info = self.vision_system.get_predicted_obstacle_pos(current_eef_pos)

        if predicted_pos is not None:
            # Translate the whole AABB by (predicted - current_center). This
            # preserves the box's size and shape while drifting it forward
            # to where we expect the obstacle to be when the arm reacts.
            shift = np.array(predicted_pos) - aabb_center
            box_min = aabb_min + shift
            box_max = aabb_max + shift

            # Closest-point-on-box: clamp EEF to [box_min, box_max] per axis.
            closest = np.maximum(box_min, np.minimum(eef, box_max))

            # Degenerate case: EEF is inside the predicted box (shouldn't
            # happen, but the test suite manages it occasionally with
            # cube spawn jitter). Fall back to the predicted center so
            # the avoider has a sensible repulsion direction.
            if np.linalg.norm(closest - eef) < 0.01:
                closest = np.array(predicted_pos)

            # Approaching bias: pull the closest point another 5cm toward
            # the EEF when the obstacle is closing on us. This advances the
            # avoider's danger-ring trigger for fast approaching obstacles.
            if pred_info.get("direction") == "approaching":
                d = eef - closest
                if np.linalg.norm(d) > 0.01:
                    closest = closest + d / np.linalg.norm(d) * 0.05

            return closest.tolist(), pred_info

        # During predictor warm-up, use the raw AABB closest point. The wider
        # 0.05m degeneracy threshold absorbs early observation noise.
        closest = np.maximum(aabb_min, np.minimum(eef, aabb_max))
        if np.linalg.norm(closest - eef) < 0.05:
            closest = aabb_center
        return closest.tolist(), {"status": "fallback"}

    def move_arm_smart(self, target_pos, target_orn=None, timeout=10.0, debug=False):
        # Vision-aware motion: at each control cycle, scan the workspace,
        # ask the avoider for a one-step movement vector, then drive IK to
        # follow it. Each step is recomputed from the latest perception.
        # Transport legs use this reactive loop; precise descent uses
        # move_arm_exact for fixed Cartesian interpolation.

        if target_orn is None:
            # Default end-effector orientation: gripper pointing straight
            # down (math.pi rotation around x), wrist rotated 90° (math.pi/2
            # around z) to align fingers with the cube's faces.
            target_orn = p.getQuaternionFromEuler([math.pi, 0, math.pi/2])

        start_time = time.time()
        obs_aabb, step_counter = None, 0
        last_status, last_pred_info = "", None

        while True:
            current_eef_pos = self.get_current_eef_pos()

            # Refresh perception every 24 sim steps. Camera renders are the
            # most expensive thing in this loop, so scans run at about 10Hz
            # while the avoider still updates every physics step.
            if step_counter % 24 == 0:
                obs_aabb = self.vision_system.scan_obstacle_volume()
                h_info = self.vision_system.scan_obstacle_height_from_side()
                self.avoider.set_obstacle_height_info(h_info)
                if debug and h_info.get("confidence", 0) > 0.3:
                    print(f"  [Side View] height:{h_info.get('max_height',0):.3f}m, clearance:{h_info.get('clearance_height',0):.3f}m")
            step_counter += 1

            # Convert raw AABB into a single avoidance position. See
            # _get_effective_obstacle_pos for the closest-point trick.
            eff_obs_pos, pred_info = self._get_effective_obstacle_pos(obs_aabb, current_eef_pos)

            # Threat-aware safety distance. Run the predictor's threat eval
            # once per step and bump d_th2 accordingly. In reactive (ablation)
            # mode we skip this and pin d_th2 at the baseline 0.40.
            if self.use_reactive_only:
                self.avoider.d_th2 = 0.40
            else:
                _, _, rec = self.vision_system.should_preemptive_avoid(current_eef_pos, target_pos)
                self.avoider.d_th2 = {"emergency_avoid": 0.55, "preemptive_avoid": 0.48}.get(rec, 0.40)

            # Pass motion direction/velocity. Skipped in reactive mode so
            # the avoider can't use the velocity-aware ring expansion (a
            # KF-derived feature that we want excluded for the ablation).
            if self.use_reactive_only:
                self.avoider.set_obstacle_motion(None, False, None)
            else:
                trend = self.vision_system.get_motion_trend()
                self.avoider.set_obstacle_motion(trend.get("velocity"), trend.get("is_moving", False), trend.get("direction"))

            # Tell the avoider how big our payload is, so the danger ring
            # uses the cube edge as the effective collision boundary.
            bounds = self.get_effective_collision_bounds()
            self.avoider.set_grabbed_object_bounds(bounds["radius_extend"], bounds["bottom_extend"])

            # The actual step decision happens inside the avoider — see
            # avoidance.py compute_modified_step.
            virtual_next, status = self.avoider.compute_modified_step(current_eef_pos, target_pos, eff_obs_pos)

            # Only print status changes, not every frame (otherwise the log
            # is unreadable).
            if debug and (status != last_status or pred_info.get("direction") != (last_pred_info or {}).get("direction")):
                eef_dist = math.sqrt(sum((a-b)**2 for a,b in zip(current_eef_pos, eff_obs_pos)))
                print(f"  [Debug] status:{status}, dist-to-obstacle:{eef_dist:.3f}, direction:{pred_info.get('direction','-')}")
                last_status, last_pred_info = status, pred_info.copy() if isinstance(pred_info, dict) else pred_info

            # 8cm arrival threshold — we want this loose so the avoider can
            # hand off to move_arm_exact (which uses 1-2cm) for the final
            # placement without us oscillating around the target here.
            dist = math.sqrt(sum((a-b)**2 for a,b in zip(current_eef_pos, target_pos)))
            if dist < 0.08:
                if debug: print(f"  [Debug] target reached!")
                break

            # IK solver gives us joint angles. Cap velocity at 2.0 rad/s —
            # higher and the IK can request impossible accelerations,
            # producing visible jerk. force=500 is the joint torque limit.
            joints = p.calculateInverseKinematics(self.robot_id, self.eef_id, virtual_next, target_orn,
                lowerLimits=self.ll, upperLimits=self.ul, jointRanges=self.jr, restPoses=self.rp)
            for i in range(7):
                p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, targetPosition=joints[i], maxVelocity=2.0, force=500)

            self.step_simulation_with_callback()
            if time.time() - start_time > timeout:
                print("Move timed out")
                break

    def move_arm_exact(self, target_pos, target_orn=None, steps=80, avoid=False, safety_radius=0.22, debug=False):
        # Linear interpolation from current position to target_pos over
        # `steps` waypoints. Used for deterministic Cartesian paths such as
        # final grasp descent and drop descent.
        # When avoid=True, we still monitor for obstacle intrusion and use
        # a 3-stage response: observe -> retreat -> replan.
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler([math.pi, 0, math.pi/2])

        obs_aabb = None
        max_retries = 3
        retry_count = 0

        # Outer while: each iteration is one full attempt at the linear path.
        # On obstacle interruption we break out of the inner for, retreat,
        # and come back to recompute dx/dy/dz from the new (post-retreat)
        # position — DON'T just resume the old waypoints, because we're
        # now at a different starting point.
        while True:
            current_pos = self.get_current_eef_pos()
            dx = (target_pos[0] - current_pos[0]) / steps
            dy = (target_pos[1] - current_pos[1]) / steps
            dz = (target_pos[2] - current_pos[2]) / steps
            retry_needed = False

            for step in range(steps):
                if avoid:
                    # Scan every 12 inner steps. Faster than move_arm_smart's
                    # 24 because move_arm_exact runs in tighter time-critical
                    # contexts (final descent) where we want quicker reaction.
                    if step % 12 == 0:
                        obs_aabb = self.vision_system.scan_obstacle_volume()

                    if obs_aabb is not None:
                        target_xy = (target_pos[0], target_pos[1])
                        h_dist = self._point_to_aabb_xy_dist(target_xy, obs_aabb)
                        bounds = self.get_effective_collision_bounds()
                        eff_radius = safety_radius + bounds["radius_extend"]

                        if h_dist < eff_radius:
                            # Same static-obstacle exception as in
                            # _wait_for_safe_path: if the obstacle isn't
                            # moving and isn't actually blocking the column
                            # (h_dist > 0), let the move continue. Otherwise
                            # we'd freeze any time the static obstacle was
                            # within 22cm of the descent line.
                            trend = self.vision_system.get_motion_trend()
                            if not trend.get("is_moving", False) and h_dist > 0.0:
                                pass   # fall through to waypoint
                            else:
                                # ---- Stage 1: observe for 1.5s ----
                                # Many obstacle intrusions are short transients
                                # (the rod sweeping past). Don't pay the cost
                                # of a full retreat for those — just pause
                                # in place and watch.
                                if debug: print(f"  [Precise Move-Avoid] obstacle intruding (h={h_dist:.3f}m), observing for 1.5s")
                                pause_start = time.time()
                                cleared = False
                                while time.time() - pause_start < 1.5:
                                    self.step_simulation_with_callback()
                                    obs_aabb = self.vision_system.scan_obstacle_volume()
                                    if obs_aabb is None:
                                        cleared = True
                                        break
                                    if self._point_to_aabb_xy_dist(target_xy, obs_aabb) > eff_radius:
                                        cleared = True
                                        break

                                if cleared:
                                    if debug: print(f"  [Precise Move-Avoid] obstacle left quickly, resuming descent")
                                    continue   # skip this step, move on

                                # ---- Stage 2: lift and wait up to 5s ----
                                # Obstacle didn't clear in 1.5s — it's
                                # camping. Get out of the way (lift 25cm
                                # above its top) so we're safe, then keep
                                # watching for it to move.
                                cur = self.get_current_eef_pos()
                                safe_z = max(cur[2], float(obs_aabb["max"][2]) + 0.25)
                                if debug: print(f"  [Precise Move-Avoid] dynamic obstacle loitering, lifting to z={safe_z:.3f}")
                                self.move_arm_smart([cur[0], cur[1], safe_z], timeout=4.0, debug=False)

                                wait_start = time.time()
                                while time.time() - wait_start < 5.0:
                                    self.step_simulation_with_callback()
                                    obs_aabb = self.vision_system.scan_obstacle_volume()
                                    if obs_aabb is None:
                                        break
                                    if self._point_to_aabb_xy_dist(target_xy, obs_aabb) > eff_radius:
                                        break

                                # ---- Stage 3: replan from new position ----
                                # The retreat lifted us, so the dx/dy/dz
                                # we computed at the start are now wrong.
                                # Break out and let the outer while re-init.
                                if debug: print(f"  [Precise Move-Avoid] path cleared, replanning the descent from the current position")
                                retry_needed = True
                                break

                # Compute the next Cartesian waypoint and drive IK to it.
                waypoint = [
                    current_pos[0] + dx * (step + 1),
                    current_pos[1] + dy * (step + 1),
                    current_pos[2] + dz * (step + 1)
                ]

                joints = p.calculateInverseKinematics(self.robot_id, self.eef_id, waypoint, target_orn,
                    lowerLimits=self.ll, upperLimits=self.ul, jointRanges=self.jr, restPoses=self.rp)

                for i in range(7):
                    p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, targetPosition=joints[i], maxVelocity=2.0, force=500)
                self.step_simulation_with_callback()

            if not retry_needed:
                break
            retry_count += 1
            # Cap at 3 retries. After repeated retreat-and-replan cycles, the
            # move is forced to finish so the state machine can continue.
            if retry_count >= max_retries:
                if debug: print(f"  [Precise Move-Avoid] interrupted after {max_retries} retries, forcing completion")
                break

    def return_to_home(self, settle_steps=480, debug=True):
        # Joint-space move (no IK) back to the rest pose. We keep the
        # sim ticking the whole time so the obstacle keeps updating —
        # otherwise it freezes mid-air and an unlucky timing would have
        # it teleport into us when the next move starts.
        if debug: print("  [Return Home] driving joints back to restPoses ...")
        for i in range(7):
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL,
                                    targetPosition=self.rp[i], maxVelocity=2.0, force=500)
        # Poll for joint convergence — exit early once max-error < 0.02 rad
        # on all 7 joints. 5s timeout matches the slowest joint's natural
        # settling time at the configured maxVelocity.
        start = time.time()
        while time.time() - start < 5.0:
            cur = [p.getJointState(self.robot_id, i)[0] for i in range(7)]
            err = max(abs(cur[i] - self.rp[i]) for i in range(7))
            self.step_simulation_with_callback()
            if err < 0.02:
                break
        # A few extra frames after convergence let residual contact forces
        # settle before the next move begins.
        for _ in range(settle_steps // 10):
            self.step_simulation_with_callback()
        if debug: print("  [Return Home] back at the initial pose")

    def execute_pick_and_place(self, cube_id, tray_id):
        # Crank up cube friction. Default friction was letting the cube
        # squirt out during the transport leg — anything that didn't have
        # both spinning and rolling friction high would lose the cube on
        # ~15% of trials.
        p.changeDynamics(cube_id, -1, lateralFriction=20.0, spinningFriction=2.0, rollingFriction=2.0, mass=0.05)
        cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
        tray_pos, _ = p.getBasePositionAndOrientation(tray_id)

        # Three keypoints: 30cm above the cube (pre-grasp), 30cm above the
        # tray (drop hover), 15cm above the tray (drop release height).
        pre_grasp = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.3]
        drop_pos = [tray_pos[0], tray_pos[1], tray_pos[2] + 0.3]
        drop_low = [tray_pos[0], tray_pos[1], tray_pos[2] + 0.15]

        # Mark the cube as ignored so the obstacle scanner doesn't try
        # to avoid the very thing we're picking up.
        self.set_ignored_object(cube_id)

        def precise_descent():
            # Re-read cube pose right before we descend — the cube can drift
            # a few mm between the smart-move and now (wind from arm motion,
            # air resistance in the sim, etc.).
            current_cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
            self._wait_for_safe_path([current_cube_pos[0], current_cube_pos[1]], safety_radius=0.22, debug=True)
            # The wait_for_safe_path may have nudged us out of position.
            # Re-read everything and start over from a clean slate.
            current_cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
            pre_grasp_refreshed = [current_cube_pos[0], current_cube_pos[1], current_cube_pos[2] + 0.3]
            # Three-stage descent: rough align (40 steps), mid hover (50),
            # final insert (80). Step counts go up because we want finer
            # waypoint spacing closer to the cube, where any error is
            # immediately visible as a missed grasp.
            self.move_arm_exact(pre_grasp_refreshed, steps=40, avoid=True, debug=True)
            mid_grasp = [current_cube_pos[0], current_cube_pos[1], current_cube_pos[2] + 0.08]
            self.move_arm_exact(mid_grasp, steps=50, avoid=True, debug=True)
            # Final z=0.035 places the fingers low enough on the cube sides
            # for a stable grasp during transport.
            new_grasp_pos = [current_cube_pos[0], current_cube_pos[1], 0.035]
            self.move_arm_exact(new_grasp_pos, steps=80, avoid=True, debug=True)

        def _check_grasp_contact():
            # Returns 0/1/2 — number of fingers actually touching the cube.
            # contact tuple [3] is the linkA index (robot side); we check
            # which of those are in our finger_indices set.
            contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=cube_id)
            fingers_touched = {c[3] for c in contacts if c[3] in self.finger_indices}
            return len(fingers_touched)

        def grasp_and_record():
            # Close-and-verify with up to 2 retries. After closing, settle the
            # physics and require contact on both fingers; otherwise release
            # and retry from a freshly re-aligned descent.
            max_retries = 2
            for attempt in range(max_retries + 1):
                self.move_gripper(False)
                # 100 frames lets the contact forces and friction anchors
                # stabilize — checking earlier sometimes misses contacts
                # that haven't propagated through the solver yet.
                for _ in range(100):
                    self.step_simulation_with_callback()

                n_fingers = _check_grasp_contact()
                if n_fingers >= 2:
                    print(f"  [Grasp Check] both fingers contacting the cube ✓")
                    self.set_grabbed_object(cube_id)
                    return

                if attempt >= max_retries:
                    print(f"  [Grasp Check] only {n_fingers}/2 fingers contacting after {max_retries} retries, giving up (later steps may fail)")
                    # Even on failure, register the grabbed object so
                    # downstream code doesn't crash on None payload geometry.
                    # The transport leg might still work if friction is enough.
                    self.set_grabbed_object(cube_id)
                    return

                print(f"  [Grasp Check] only {n_fingers}/2 fingers contacting, retry #{attempt + 1}")
                # Open, lift 15cm, and re-descend. We don't redo the full
                # smart-move — we trust we're approximately over the cube,
                # so a quick exact lift+descend is enough.
                self.move_gripper(True)
                current_cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
                lift_pos = [current_cube_pos[0], current_cube_pos[1], current_cube_pos[2] + 0.15]
                self.move_arm_exact(lift_pos, steps=40)
                redescend_pos = [current_cube_pos[0], current_cube_pos[1], 0.035]
                self.move_arm_exact(redescend_pos, steps=80)

        def release_and_clear():
            # Same path-safety check as grasp, but for the drop column above
            # the tray. If the obstacle is camped over the tray, we'll
            # retreat and wait.
            self._wait_for_safe_path([drop_low[0], drop_low[1]], safety_radius=0.22, debug=True)
            # The wait may have lifted us; smart-move back to the drop hover.
            self.move_arm_smart(drop_pos, timeout=10.0, debug=True)
            # Then exact-line descend to release height. avoid=True so we
            # still react if the obstacle wanders in during the descent.
            self.move_arm_exact(drop_low, steps=100, avoid=True, debug=True)
            self.move_gripper(True)
            # Cube is now sitting in the tray — the avoider should stop
            # treating us as carrying anything.
            self.clear_grabbed_object()

        # The 7-step pipeline. Each step is (label, callable). Step labels
        # get printed before execution; failures print but don't abort
        # (the next step might still succeed, e.g., a partial grasp can
        # still complete the transport).
        steps = [("1. Approach above the cube", lambda: (self.move_gripper(True), self.move_arm_smart(pre_grasp, timeout=15.0))),
                 ("2. Precise descent (dynamic alignment)", precise_descent),
                 ("3. Close gripper and record object volume", grasp_and_record),
                 ("4. Lift (avoidance accounts for object volume)", lambda: self.move_arm_smart(pre_grasp, timeout=10.0, debug=True)),
                 ("5. Transport to drop point (avoidance accounts for object volume)", lambda: self.move_arm_smart(drop_pos, timeout=30.0, debug=True)),
                 ("6. Release and clear volume info", release_and_clear),
                 ("7. Return to initial pose", self.return_to_home)]

        for name, action in steps:
            print(f">>> {name}")
            action()
        print("Task complete")
