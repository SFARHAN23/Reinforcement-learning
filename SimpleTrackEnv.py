"""
SimpleTrackEnv_v2.py
=====================
SimpleTrack GT3 environment with full cornering capability improvements.
Identical structure to MonzaTrackEnv_v2 — see that file for full docs.

MODE A (ENHANCED_OBS=False): 31 dims, backward-compatible, existing models work
MODE B (ENHANCED_OBS=True):  34 dims, extended lookahead + physical state dims
"""

import numpy as np
from CarModel_Kinematic import CarModelClass, MaxSteer, SPD_MAX
from SimpleTrack import SimpleTrackClass
from corner_physics import (
    scan_min_v_ref, grip_utilisation_bonus,
    speed_potential, corner_exit_bonus, corner_survival_bonus,
    _MU, _G, _VNORM
)

ENHANCED_OBS: bool = True
_OBS_SPD_NORM  = SPD_MAX       # 90.0
_OBS_STEER_MAX = MaxSteer      # π/6
STATE_DIM      = 34 if ENHANCED_OBS else 31


class SimpleTrackEnvClass:

    _STOP_SPEED: float = 6.0

    def __init__(self, am=None, gamma: float = 0.99):
        self.track = SimpleTrackClass()
        self.am    = am
        self.gamma = gamma
        self._last_ob       = None
        self._phi_prev      = 0.0
        self._was_in_corner = False
        self._last_v_ref    = _VNORM

    def set_am(self, am):
        self.am = am

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
        self.car = CarModelClass([0, 0, np.pi / 2], spd0=10)
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

            current_unit = self.track.unit_list[self.track.car_in_unit]
            self._is_in_corner = (self.track.car_trip >
                                   current_unit.start_trip + current_unit.len1)
            self._corner_radius = current_unit.radius if self._is_in_corner else 0.0

            self._last_v_ref = scan_min_v_ref(
                self.track, self.track.car_trip, vx=spd
            )

            if ENHANCED_OBS:
                # Extended: 140m/160m/180m replaced with 300m/400m/500m
                lookahead_dists = [5, 10, 20, 30, 40, 60, 80, 100, 120,
                                   300, 400, 500, 200]
            else:
                lookahead_dists = [5, 10, 20, 30, 40, 60, 80, 100, 120,
                                   140, 160, 180, 200]

            centerpoints = []
            for d in lookahead_dists:
                pt = self.track.find_relative_centerpoint(pose, d)
                scale = max(100.0, float(d))
                centerpoints += [pt[0] / scale, pt[1] / scale]

            spd_n     = spd     / _OBS_SPD_NORM
            dist_n    = dist    / (self.track.width / 2)
            psi_dot_n = psi_dot / 1.57
            steer_n   = steer   / _OBS_STEER_MAX
            angle00_n = angle00 / np.pi

            ob = [spd_n, psi_dot_n, steer_n, dist_n, angle00_n] + centerpoints

            if ENHANCED_OBS:
                # Dynamic preview target speed for the tightest upcoming corner.
                v_ref_norm    = self._last_v_ref / _OBS_SPD_NORM
                long_acc_norm = float(np.clip(
                    self.car.long_acc / (_MU * _G), -1.5, 1.5
                ))
                grip_util = (self.am.last_result.grip_utilisation
                             if (self.am and self.am.last_result) else 0.0)
                ob += [v_ref_norm, long_acc_norm, grip_util]

            return ob

        else:
            self.OUT_TRACK = True
            self.FAIL      = True
            return list(self._last_ob) if self._last_ob is not None \
                   else [0.0] * STATE_DIM

    def reward(self) -> float:
        if getattr(self, 'FINISH', False):
            return 1000.0  # Big reward for lap completion!

        if self.FAIL:
            self._phi_prev      = 0.0
            self._was_in_corner = False
            return -1000.0

        r = self.spd * np.cos(self.angle00) / 10.0

        if self.am is not None and self.am.last_result is not None:
            r += grip_utilisation_bonus(self.am.last_result.grip_utilisation)

        phi_curr, _ = speed_potential(self.spd, self._last_v_ref, self.gamma)
        r          += self.gamma * phi_curr - self._phi_prev
        self._phi_prev = phi_curr

        r += corner_exit_bonus(
            self._was_in_corner, self._is_in_corner,
            self.spd, self._last_v_ref
        )
        self._was_in_corner = self._is_in_corner

        # Corner survival bonus — per-step reward inside corners
        r += corner_survival_bonus(self._is_in_corner, self._corner_radius)

        return r

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


if __name__ == '__main__':
    from DynamicActionMapping import DynamicActionMappingClass
    am  = DynamicActionMappingClass()
    env = SimpleTrackEnvClass(am=am)
    ob  = env.test_reset()
    print(f'ENHANCED_OBS={ENHANCED_OBS}  dims={len(ob)}  expected={STATE_DIM}')
    print(f'v_ref at start: {env._last_v_ref:.1f} m/s')
    for step in range(3):
        ax = env.car.long_acc
        ai = am.mapping(env.car.spd, ax, 0.8, 0.0)
        ob, r, done = env.step(ai)
        print(f'  step {step}  spd={env.car.spd*3.6:.1f}km/h  '
              f'v_ref={env._last_v_ref*3.6:.0f}km/h  r={r:.3f}')
        if done: break
