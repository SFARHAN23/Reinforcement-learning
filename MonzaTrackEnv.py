"""
MonzaTrackEnv_v2.py
====================
Monza GT3 environment with full cornering capability improvements.

TWO MODES — select via ENHANCED_OBS flag:
-----------------------------------------

MODE A — ENHANCED_OBS = False  (state_dim = 31, backward-compatible)
    Tier 1 improvements only. Works with EXISTING trained models.
    Changes vs current MonzaTrackEnv:
        - Grip utilisation reward bonus (dense cornering signal)
        - Reference speed potential shaping (braking gradient)
        - Corner exit bonus (sparse corner navigation reward)
    Observation space: identical 31 dims, no retraining required.

MODE B — ENHANCED_OBS = True   (state_dim = 34, requires fresh training)
    Tier 1 + Tier 2 improvements. Start new training run.
    Additional observation dims [31, 32, 33]:
        obs[31] = v_ref_ahead / 90    minimum reference speed in the dynamic
                                      braking-preview horizon
        obs[32] = long_acc / (mu*g)   normalised longitudinal acceleration
        obs[33] = grip_util           current grip utilisation from DAM

    The 3 far waypoints at 140m, 160m, 180m are REPLACED with:
        - 300m, 400m, 500m lookahead waypoints
    This extends the effective horizon from 200m to 500m while keeping
    the same observation structure (only the distances change):
        5m, 10m, 20m, 30m, 40m, 60m, 80m, 100m, 120m, 300m, 400m, 500m, 200m
    → 13 waypoints, same 26 dims for waypoints, same total = 31 for Tier 1
    → + 3 new physical state dims = 34 for Tier 2

Reward function (both modes):
    r = v·cos(θ)/10                           [progress — original]
      + grip_utilisation_bonus(util)          [Tier 1: dense cornering signal]
      + γ·Φ(s'_v_ref) - Φ(s_v_ref)          [Tier 1: braking gradient]
      + corner_exit_bonus(...)                [Tier 1: sparse corner bonus]
    r = -100 if failed

Why each addition:
    - Grip bonus: turns "I am cornering at 85% of my tyre limit" from an
      invisible state into a rewarded state. First dense signal for the
      policy that "using grip efficiently = good."
    - Speed potential: adds a gradient toward the correct braking point.
      Potential-based (Ng 1999) → cannot change the optimal policy, only
      accelerates convergence toward it.
    - Corner bonus: sparse +5 when the car navigates a corner and exits
      onto the next straight. Provides a checkpoint reward that motivates
      completing corners rather than avoiding them.

Usage
-----
    # Tier 1 (existing models):
    from MonzaTrackEnv_v2 import MonzaTrackEnvClass
    # state_dim stays 31

    # Tier 2 (fresh training):
    from MonzaTrackEnv_v2 import MonzaTrackEnvClass, ENHANCED_OBS
    # set ENHANCED_OBS = True at top of file, state_dim = 34

    # Pass am to env so it can read grip_util:
    env = MonzaTrackEnvClass(am=am)
    # and in the training loop, call env.update_am(am) after am.mapping()
"""

import numpy as np
from CarModel_Kinematic import CarModelClass
from MonzaTrack import MonzaTrackClass
from corner_physics import (
    scan_min_v_ref, grip_utilisation_bonus,
    speed_potential, corner_exit_bonus, corner_survival_bonus,
    ALPHA_GRIP, ALPHA_SPEED, ALPHA_CORNER, ALPHA_SURVIVAL,
    _MU, _G, _VNORM
)

# ── MODE SELECTOR ──────────────────────────────────────────────────────────
# False = 31 dims (backward compatible, existing models work)
# True  = 34 dims (requires new training run)
ENHANCED_OBS: bool = True
# ──────────────────────────────────────────────────────────────────────────

_OBS_SPD_NORM  = 90.0
_OBS_STEER_MAX = np.pi / 6   # GT3 MaxSteer = 30°
STATE_DIM      = 34 if ENHANCED_OBS else 31


class MonzaTrackEnvClass:
    """
    Monza GP circuit environment with corner-aware reward shaping.

    Parameters
    ----------
    am : DynamicActionMappingClass or None
        Pass the action mapping object so the env can read grip utilisation
        for reward shaping and (in Tier 2) the observation vector.
        If None, grip-based reward terms are skipped gracefully.
    gamma : float
        TD3 discount factor. Used in potential-based shaping.
        Must match the value used in TD3 training (default 0.99).
    """

    _STOP_SPEED: float = 5.0   # lowered from 8.0 for training — tight chicanes need room

    def __init__(self, am=None, gamma: float = 0.99):
        self.track = MonzaTrackClass()
        self.am    = am
        self.gamma = gamma
        self._last_ob       = None
        self._phi_prev      = 0.0    # previous potential for shaping
        self._was_in_corner = False  # for corner exit bonus
        self._last_v_ref    = _VNORM

    def set_am(self, am):
        """Update the action mapping reference (call after am = DynamicActionMappingClass())."""
        self.am = am

    # ── Episode management ────────────────────────────────────────────────

    def reset(self):
        pose     = self.track.random_car_pose()
        spd      = np.random.uniform(10, 20)
        self.car = CarModelClass(pose, spd)
        
        self.track.findcar(pose)
        self.start_trip = self.track.car_trip
        
        self.reset_flags()
        ob = self.observe()
        self._last_ob = ob
        return ob

    def test_reset(self):
        self.car = CarModelClass([0, 0, np.pi / 2], spd0=15)
        self.track.findcar([0, 0, np.pi / 2])
        self.start_trip = self.track.car_trip
        self.reset_flags()
        ob = self.observe()
        self._last_ob = ob
        return ob

    def step(self, action):
        self.car.step(action)
        ob = self.observe()
        self._last_ob = ob
        self.check_fails()
        r = self.reward()
        return ob, r, self.FAIL

    # ── Observation ───────────────────────────────────────────────────────

    def observe(self):
        pose    = self.car.pose
        self.spd = spd    = self.car.spd
        psi_dot            = self.car.psi_dot
        steer              = self.car.steer

        if self.track.findcar(pose):
            self.dist    = dist    = self.track.centerlinedist
            self.angle00 = angle00 = self.track.find_cartrack_angle(pose)

            # Lap Progress with Wrapping
            if not hasattr(self, 'start_trip'):
                self.start_trip = 0.0
            progress = (self.track.car_trip - self.start_trip) % self.track.total_trip
            if progress < 0:
                progress += self.track.total_trip
            
            self.lap_pct = (progress / self.track.total_trip) * 100.0
            if self.lap_pct > self.max_lap_pct:
                self.max_lap_pct = self.lap_pct

            # Track whether we're in a curve section (for corner exit bonus)
            current_unit = self.track.unit_list[self.track.car_in_unit]
            self._is_in_corner = (self.track.car_trip >
                                   current_unit.start_trip + current_unit.len1)
            self._corner_radius = current_unit.radius if self._is_in_corner else 0.0

            # Reference speed for upcoming corners (used in reward/obs[31])
            self._last_v_ref = scan_min_v_ref(
                self.track, self.track.car_trip, vx=spd
            )

            # ── Waypoint distances ────────────────────────────────────────
            if ENHANCED_OBS:
                # Extended lookahead: replace 140/160/180m with 300/400/500m
                lookahead_dists = [5, 10, 20, 30, 40, 60, 80, 100, 120,
                                   300, 400, 500, 200]
            else:
                # Original lookahead distances
                lookahead_dists = [5, 10, 20, 30, 40, 60, 80, 100, 120,
                                   140, 160, 180, 200]

            centerpoints = []
            for d in lookahead_dists:
                pt = self.track.find_relative_centerpoint(pose, d)
                scale = max(100.0, float(d))
                centerpoints += [pt[0] / scale, pt[1] / scale]

            # ── Normalise ─────────────────────────────────────────────────
            spd_n     = spd     / _OBS_SPD_NORM
            dist_n    = dist    / (self.track.width / 2)
            psi_dot_n = psi_dot / 1.57
            steer_n   = steer   / _OBS_STEER_MAX
            angle00_n = angle00 / np.pi

            ob = [spd_n, psi_dot_n, steer_n, dist_n, angle00_n] + centerpoints

            if ENHANCED_OBS:
                # ── 3 additional physical state dims ──────────────────────
                # obs[31]: dynamic-preview reference speed for upcoming corner
                v_ref_norm = self._last_v_ref / _OBS_SPD_NORM

                # obs[32]: normalised longitudinal acceleration
                long_acc_norm = self.car.long_acc / (_MU * _G)
                long_acc_norm = float(np.clip(long_acc_norm, -1.5, 1.5))

                # obs[33]: grip utilisation (from DAM last result)
                if self.am is not None and self.am.last_result is not None:
                    grip_util = self.am.last_result.grip_utilisation
                else:
                    grip_util = 0.0

                ob += [v_ref_norm, long_acc_norm, grip_util]

            return ob   # 31 dims (basic) or 34 dims (enhanced)

        else:
            # Graceful out-of-track fallback (FIX 1)
            self.OUT_TRACK = True
            self.FAIL      = True
            return list(self._last_ob) if self._last_ob is not None \
                   else [0.0] * STATE_DIM

    # ── Reward ────────────────────────────────────────────────────────────

    def reward(self) -> float:
        if getattr(self, 'FINISH', False):
            return 1000.0  # Big reward for lap completion!

        if self.FAIL:
            self._phi_prev      = 0.0
            self._was_in_corner = False
            return -1000.0

        # ── Base progress reward (original) ───────────────────────────────
        r = self.spd * np.cos(self.angle00) / 10.0

        # ── Tier 1A: Grip utilisation bonus ───────────────────────────────
        # Dense reward for using 50–92% of the friction ellipse.
        # Requires DAM to have been called this step.
        if self.am is not None and self.am.last_result is not None:
            util  = self.am.last_result.grip_utilisation
            r    += grip_utilisation_bonus(util)
        else:
            util  = 0.0

        # ── Tier 1B: Reference speed potential shaping ────────────────────
        # Penalise excess speed relative to the tightest upcoming corner.
        # Potential-based → cannot change the optimal policy.
        phi_curr, _ = speed_potential(self.spd, self._last_v_ref, self.gamma)
        r          += self.gamma * phi_curr - self._phi_prev
        self._phi_prev = phi_curr

        # ── Tier 1C: Corner exit bonus ────────────────────────────────────
        # Sparse +5 when the car successfully exits a corner onto a straight.
        r += corner_exit_bonus(
            self._was_in_corner, self._is_in_corner,
            self.spd, self._last_v_ref
        )
        self._was_in_corner = self._is_in_corner

        # ── Tier 1D: Corner survival bonus ─────────────────────────────────
        # Small per-step reward for surviving inside a corner, weighted by
        # 1/sqrt(radius) so tight chicanes get more reward per step.
        r += corner_survival_bonus(self._is_in_corner, self._corner_radius)

        return r

    # ── Flags ─────────────────────────────────────────────────────────────

    def reset_flags(self):
        self.OUT_TRACK      = False
        self.WRONG_DIR      = False
        self.MAX_ACC        = False
        self.STOP           = False
        self.FAIL           = False
        self.dist           = 0.0
        self.angle00        = 0.0
        self.lap_pct        = 0.0
        self.max_lap_pct    = 0.0
        self._phi_prev      = 0.0
        self._was_in_corner = False
        self._is_in_corner  = False
        self._corner_radius = 0.0
        self._last_v_ref    = _VNORM
        self.FINISH         = False

    def check_fails(self):
        if abs(self.angle00) > np.pi / 2:
            self.WRONG_DIR = True; self.FAIL = True
        if abs(self.dist) > self.track.width / 2 * 0.95:
            self.OUT_TRACK = True; self.FAIL = True
        if self.car.check_acc():
            self.MAX_ACC   = True; self.FAIL = True
        if self.car.spd < self._STOP_SPEED:
            self.STOP      = True; self.FAIL = True
            
        # Completion Success (only valid if remaining on track and facing correct direction)
        if getattr(self, 'lap_pct', 0.0) >= 99.8 and not self.WRONG_DIR and not self.OUT_TRACK:
            self.FINISH = True
            self.FAIL = False

    def query_fail_reason(self) -> str:
        if   self.OUT_TRACK: return 'OUT_TRACK'
        elif self.MAX_ACC:   return 'MAX_ACC'
        elif self.WRONG_DIR: return 'WRONG_DIR'
        elif self.STOP:      return 'STOP'
        elif getattr(self, 'FINISH', False): return 'FINISH'
        else:                return 'TIME_LIMIT'


# ── Self-test ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from DynamicActionMapping import DynamicActionMappingClass
    am  = DynamicActionMappingClass()
    env = MonzaTrackEnvClass(am=am)
    ob  = env.test_reset()
    print(f'ENHANCED_OBS = {ENHANCED_OBS}')
    print(f'obs dims     = {len(ob)}  (expected {STATE_DIM})')
    print(f'v_ref ahead  = {env._last_v_ref:.1f} m/s at start/finish')

    # Take 5 steps with full throttle — see reward breakdown
    print('\n5 steps at full throttle from start line:')
    for step in range(5):
        ax_prev   = env.car.long_acc
        action_in = am.mapping(env.car.spd, ax_prev, 1.0, 0.0)
        ob, r, done = env.step(action_in)
        util = am.last_result.grip_utilisation if am.last_result else 0
        print(f'  step {step}  spd={env.car.spd*3.6:.1f}km/h  '
              f'v_ref={env._last_v_ref*3.6:.0f}km/h  '
              f'util={util:.3f}  r={r:.3f}')
        if done: break

    print(f'\nReward coefficients:')
    print(f'  ALPHA_GRIP   = {ALPHA_GRIP}   (grip utilisation bonus)')
    print(f'  ALPHA_SPEED  = {ALPHA_SPEED}   (reference speed potential)')
    print(f'  ALPHA_CORNER = {ALPHA_CORNER}   (corner exit bonus)')
