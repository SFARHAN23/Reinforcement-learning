"""
MonzaTrack.py
=============
Autodromo Nazionale di Monza — track geometry for RL training.

Built using the same TrackUnitClass system as SimpleTrack.py (Wang et al., 2024).
Each TrackUnit = one straight section followed by one curved section.

Real-world reference
--------------------
    Monza GP circuit: 5,793 m (FIA homologated lap distance)
    Track width: ~12–14 m (modelled as 20 m, consistent with SimpleTrack)

Segment map (11 units, clockwise direction)
-------------------------------------------
    Unit  0 : Main straight (Rettifilo)       + T1 right  100°  r=15 m
    Unit  1 : 10 m connector                  + T2 left   108°  r=15 m  (chicane exit)
    Unit  2 : 250 m straight                  + Curva Grande right 95°  r=450 m
    Unit  3 : 480 m straight                  + Roggia left  85°  r=22 m
    Unit  4 : 20 m connector                  + Roggia exit right 82°  r=22 m
    Unit  5 : 420 m straight                  + Lesmo 1 right 68°  r=85 m
    Unit  6 : 220 m straight                  + Lesmo 2 right 82°  r=60 m
    Unit  7 : 1 000 m straight (Ascari app.)  + Ascari left  78°  r=50 m
    Unit  8 : 5 m connector                   + Ascari right 102°  r=65 m
    Unit  9 : 5 m connector                   + Ascari exit left 62°  r=110 m
    Unit 10 : 770 m straight                  + Parabolica right 164°  r=192 m

Angle closure derivation
------------------------
    Starting heading: π/2 (due north).  Cumulative heading after each unit:
        After T1  :  +90° - 100° = -10°
        After T2  :  -10° + 108° = +98°
        After CG  :  +98° -  95° =  +3°
        After R-L :   +3° +  85° = +88°
        After R-R :  +88° -  82° =  +6°
        After L1  :   +6° -  68° = -62°
        After L2  :  -62° -  82° = -144°
        After A-L : -144° +  78° = -66°
        After A-R :  -66° - 102° = -168°
        After A-X : -168° +  62° = -106°
        After Para: -106° - 164° = -270° ≡ +90°  ✓  (closed)

    Net angle change = -360° (one clockwise lap).  164° Parabolica (vs the
    commonly cited 150°) is necessary for geometric closure.

Speed / physics notes (CarModel_Kinematic.py at v_max = 30 m/s)
----------------------------------------------------------------
    Lesmo 2 r=60 m  → v_max for AccMax (1.2g) = √(11.77×60) ≈ 26.6 m/s  — must brake
    Roggia  r=22 m  → v_max ≈ 16.1 m/s                                   — heavy braking
    T1-2    r=15 m  → v_max ≈ 13.3 m/s                                   — deepest braking
    These constraints are physically meaningful and create the RL challenge.

Dependencies
------------
    SimpleTrack.py  (imports TrackUnitClass + utility functions)
"""

import numpy as np
import random
import matplotlib.pyplot as plt

from SimpleTrack import (
    TrackUnitClass,
    torad, todeg,
    get_distance, get_bearing, adjust_angle,
    get_vector_angle, get_angle_diff,
)


class MonzaTrackClass:
    """
    Monza GP circuit modelled as 11 TrackUnit segments.

    Public API is identical to SimpleTrackClass — any code that works
    with SimpleTrackClass works with MonzaTrackClass without changes,
    except that unit indices now run 0–10 instead of 0–4.
    """

    def __init__(self):
        self.width = 20  # track half-width [m], consistent with SimpleTrack

        # ── Chain builder ─────────────────────────────────────────────────────
        # All units start at the beginning of the main straight.
        zero_pos     = [0, 0]
        zero_angle   = np.pi / 2   # heading due north (up) at start/finish line
        rotate_angle = 0.0
        start_trip   = 0.0
        w            = self.width

        def _unit(len1, curve_deg, radius):
            """Convenience wrapper: degrees → radians, returns a TrackUnitClass."""
            return TrackUnitClass(
                width        = w,
                len1         = len1,
                curve_angle  = torad(curve_deg),
                radius       = radius,
                zero_pos     = zero_pos,
                zero_angle   = zero_angle,
                rotate_angle = rotate_angle,
                start_trip   = start_trip,
            )

        # ── Unit 0 : Rettifilo main straight + T1 right 100° ─────────────────
        # Straight = 949.945 - 46.43 = 903.515 m.
        # 46.43 m correction closes the y-position error (Parabolica exit was
        # 46.43 m past the start line on the first geometry run).
        print("── Monza Unit 0 ── Main Straight + T1 Rettifilo")
        u0 = _unit(len1=903.515, curve_deg=-100, radius=15)
        zero_pos, zero_angle, rotate_angle, start_trip = u0.connect_info()
        print(f"   Exit pos: ({zero_pos[0]:.2f}, {zero_pos[1]:.2f})  "
              f"heading: {todeg(zero_angle):.2f}°")

        # ── Unit 1 : 10 m connector + T2 left 108° (chicane exit) ─────────────
        print("── Monza Unit 1 ── T2 Rettifilo Exit")
        u1 = _unit(len1=10, curve_deg=108, radius=15)
        zero_pos, zero_angle, rotate_angle, start_trip = u1.connect_info()
        print(f"   Exit pos: ({zero_pos[0]:.2f}, {zero_pos[1]:.2f})  "
              f"heading: {todeg(zero_angle):.2f}°")

        # ── Unit 2 : 250 m + Curva Grande right 95° ───────────────────────────
        print("── Monza Unit 2 ── Curva Grande")
        u2 = _unit(len1=250, curve_deg=-95, radius=450)
        zero_pos, zero_angle, rotate_angle, start_trip = u2.connect_info()
        print(f"   Exit pos: ({zero_pos[0]:.2f}, {zero_pos[1]:.2f})  "
              f"heading: {todeg(zero_angle):.2f}°")

        # ── Unit 3 : 480 m + Roggia left 85° ──────────────────────────────────
        print("── Monza Unit 3 ── Roggia Entry (left)")
        u3 = _unit(len1=480, curve_deg=85, radius=22)
        zero_pos, zero_angle, rotate_angle, start_trip = u3.connect_info()
        print(f"   Exit pos: ({zero_pos[0]:.2f}, {zero_pos[1]:.2f})  "
              f"heading: {todeg(zero_angle):.2f}°")

        # ── Unit 4 : 20 m connector + Roggia right 82° ────────────────────────
        print("── Monza Unit 4 ── Roggia Exit (right)")
        u4 = _unit(len1=20, curve_deg=-82, radius=22)
        zero_pos, zero_angle, rotate_angle, start_trip = u4.connect_info()
        print(f"   Exit pos: ({zero_pos[0]:.2f}, {zero_pos[1]:.2f})  "
              f"heading: {todeg(zero_angle):.2f}°")

        # ── Unit 5 : 420 m + Lesmo 1 right 68° ───────────────────────────────
        print("── Monza Unit 5 ── Lesmo 1")
        u5 = _unit(len1=420, curve_deg=-68, radius=85)
        zero_pos, zero_angle, rotate_angle, start_trip = u5.connect_info()
        print(f"   Exit pos: ({zero_pos[0]:.2f}, {zero_pos[1]:.2f})  "
              f"heading: {todeg(zero_angle):.2f}°")

        # ── Unit 6 : 220 m + Lesmo 2 right 82° ───────────────────────────────
        print("── Monza Unit 6 ── Lesmo 2")
        u6 = _unit(len1=220, curve_deg=-82, radius=60)
        zero_pos, zero_angle, rotate_angle, start_trip = u6.connect_info()
        print(f"   Exit pos: ({zero_pos[0]:.2f}, {zero_pos[1]:.2f})  "
              f"heading: {todeg(zero_angle):.2f}°")

        # ── Unit 7 : 1 000 m + Ascari entry left 78° ─────────────────────────
        print("── Monza Unit 7 ── Ascari Entry (left)")
        u7 = _unit(len1=1000, curve_deg=78, radius=50)
        zero_pos, zero_angle, rotate_angle, start_trip = u7.connect_info()
        print(f"   Exit pos: ({zero_pos[0]:.2f}, {zero_pos[1]:.2f})  "
              f"heading: {todeg(zero_angle):.2f}°")

        # ── Unit 8 : 5 m connector + Ascari mid right 102° ───────────────────
        print("── Monza Unit 8 ── Ascari Mid (right)")
        u8 = _unit(len1=5, curve_deg=-102, radius=65)
        zero_pos, zero_angle, rotate_angle, start_trip = u8.connect_info()
        print(f"   Exit pos: ({zero_pos[0]:.2f}, {zero_pos[1]:.2f})  "
              f"heading: {todeg(zero_angle):.2f}°")

        # ── Unit 9 : 5 m connector + Ascari exit left 62° ────────────────────
        print("── Monza Unit 9 ── Ascari Exit (left)")
        u9 = _unit(len1=5, curve_deg=62, radius=110)
        zero_pos, zero_angle, rotate_angle, start_trip = u9.connect_info()
        print(f"   Exit pos: ({zero_pos[0]:.2f}, {zero_pos[1]:.2f})  "
              f"heading: {todeg(zero_angle):.2f}°")

        # ── Unit 10 : 770 m + Parabolica right 164° ───────────────────────────
        # 164° (not the commonly cited 150°) is required for angular closure.
        # Derivation: cumulative heading before Parabolica = -106°.
        # For exit heading = +90° (= start): -106° - X = -270° → X = 164°.
        print("── Monza Unit 10 ── Parabolica")
        u10 = _unit(len1=770, curve_deg=-164, radius=192)
        zero_pos, zero_angle, rotate_angle, start_trip = u10.connect_info()
        print(f"   Exit pos: ({zero_pos[0]:.2f}, {zero_pos[1]:.2f})  "
              f"heading: {todeg(zero_angle):.2f}°")

        # ── Closure diagnostic ─────────────────────────────────────────────────
        pos_error = get_distance(zero_pos, [0, 0])
        angle_error = abs(todeg(adjust_angle(zero_angle - np.pi / 2)))
        print("══════════════════════════════════════════")
        print("Monza Track Initialized")
        print(f"  Circuit closure — position error : {pos_error:.2f} m")
        print(f"  Circuit closure — heading  error : {angle_error:.4f}°")
        if pos_error > 100:
            print("  ⚠ Position error > 100 m — consider adjusting main straight length")
        if angle_error > 1.0:
            print("  ⚠ Heading error > 1° — check Parabolica angle")
        else:
            print("  ✓ Heading closure confirmed")
        print("══════════════════════════════════════════")

        # ── Register units ─────────────────────────────────────────────────────
        self.unit_list = [u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10]
        self._N = len(self.unit_list)   # 11 — used everywhere instead of hardcoded 4/5

        self.start_trip_list = [u.start_trip for u in self.unit_list]
        self.end_trip_list   = [u.end_trip   for u in self.unit_list]
        self.total_trip      = self.end_trip_list[-1]

        print(f"  Total circuit length : {self.total_trip:.1f} m")
        print(f"  Segment end trips    : {[f'{t:.0f}' for t in self.end_trip_list]}")
        print("══════════════════════════════════════════")

    # ── Car positioning ────────────────────────────────────────────────────────

    def findcar(self, pos):
        """Find which unit the car is in and update track state. Returns True/False."""
        for i, unit in enumerate(self.unit_list):
            if unit.findcar(pos):
                self.centerlinepoint = unit.centerlinepoint
                self.centerlinedist  = unit.centerlinedist
                self.track_dir       = unit.track_dir
                self.car_trip        = unit.start_trip + unit.unit_trip
                self.car_in_unit     = i
                return True
        self.car_in_unit     = None
        self.centerlinepoint = None
        self.centerlinedist  = None
        self.track_dir       = None
        return False

    # ── Forward point lookup (generalised for N units) ─────────────────────────

    def find_forward_point(self, addtrip):
        """Return the centreline point a given distance ahead of the car."""
        i         = self.car_in_unit
        temp_trip = self.car_trip + addtrip
        while temp_trip > self.unit_list[i].end_trip:
            i += 1
            if i >= self._N:                          # cross start/finish line
                temp_trip -= self.unit_list[self._N - 1].end_trip
                i = 0
        unit_addtrip         = temp_trip - self.unit_list[i].start_trip
        pt                   = self.unit_list[i].trip_to_centerlinepoint(unit_addtrip)
        self.fpoint_in_unit  = i
        self.fpoint_unit_trip = unit_addtrip
        return pt

    def find_forward_edgepoint(self, addtrip):
        """Return the (left, right) edge points a given distance ahead."""
        i         = self.car_in_unit
        temp_trip = self.car_trip + addtrip
        while temp_trip > self.unit_list[i].end_trip:
            i += 1
            if i >= self._N:
                temp_trip -= self.unit_list[self._N - 1].end_trip
                i = 0
        unit_addtrip          = temp_trip - self.unit_list[i].start_trip
        ptL, ptR              = self.unit_list[i].trip_to_edgelinepoint(unit_addtrip)
        self.fpoint_in_unit   = i
        self.fpoint_unit_trip = unit_addtrip
        return ptL, ptR

    # ── Angle helpers (identical API to SimpleTrackClass) ──────────────────────

    def find_cartrack_angle(self, pose):
        return get_angle_diff(self.track_dir, pose[2])

    def find_forward_angle(self, pose, addtrip):
        pt0         = [pose[0], pose[1]]
        heading_dir = pose[2]
        pt1         = self.find_forward_point(addtrip)
        pos_dir     = get_bearing(pt0, pt1)
        return adjust_angle(pos_dir - heading_dir)

    def find_forward_trackangle(self, pose, addtrip):
        pt1 = self.find_forward_point(addtrip)
        i   = self.fpoint_in_unit
        forward_track_dir = self.unit_list[i].get_point_track_direction(
            pt1, self.fpoint_unit_trip
        )
        return adjust_angle(forward_track_dir - pose[2])

    def find_relative_centerpoint(self, pose, addtrip):
        pt0     = [pose[0], pose[1]]
        heading = -pose[2]
        cpt0    = self.find_forward_point(addtrip)
        pt1     = [cpt0[0] - pt0[0], cpt0[1] - pt0[1]]
        def rotate(pt, a):
            x1 = pt[0] * np.cos(a) - pt[1] * np.sin(a)
            y1 = pt[0] * np.sin(a) + pt[1] * np.cos(a)
            return [x1, y1]
        return rotate(pt1, heading)

    def find_relative_edgepoint(self, pose, addtrip):
        pt0     = [pose[0], pose[1]]
        heading = pose[2] - np.pi / 2
        ptL0, ptR0 = self.find_forward_edgepoint(addtrip)
        ptL1    = [ptL0[0] - pt0[0], ptL0[1] - pt0[1]]
        ptR1    = [ptR0[0] - pt0[0], ptR0[1] - pt0[1]]
        def rotate(pt, a):
            x1 = pt[0] * np.cos(a) - pt[1] * np.sin(a)
            y1 = pt[0] * np.sin(a) + pt[1] * np.cos(a)
            return [x1, y1]
        return rotate(ptL1, heading), rotate(ptR1, heading)

    # ── Random spawn (generalised, biased toward straight sections) ────────────

    def random_car_pose(self):
        """
        Spawn car at a random position within any unit's straight section.

        Biased toward longer straights (units 0, 2, 3, 5, 6, 7, 10) to avoid
        spawning in the middle of a tight chicane (units 1, 4, 8, 9).
        """
        # Weight spawn probability by straight length
        weights = np.array([u.len1 for u in self.unit_list], dtype=float)
        weights /= weights.sum()
        i    = np.random.choice(self._N, p=weights)
        rate = 0.9   # stay away from track edges
        unit = self.unit_list[i]
        x0   = random.uniform(-self.width / 2 * rate, self.width / 2 * rate)
        y0   = random.uniform(0, unit.len1)
        movex, movey = unit.zero_pos[0], unit.zero_pos[1]
        a    = unit.rotate_angle
        x1   = x0 * np.cos(a) - y0 * np.sin(a)
        y1   = x0 * np.sin(a) + y0 * np.cos(a)
        return [x1 + movex, y1 + movey, adjust_angle(unit.zero_angle)]

    def custom_car_pose(self, trip, dist):
        if trip > self.unit_list[-1].end_trip:
            raise RuntimeError('custom_car_pose: trip > total track length')
        for i in range(self._N):
            if trip < self.unit_list[i].end_trip:
                unit      = self.unit_list[i]
                unit_trip = trip - unit.start_trip
                break
        x, y, psi = unit.trip_dist_to_custompose(unit_trip, dist)
        return [x, y, psi]

    # ── Visualisation ──────────────────────────────────────────────────────────

    def show(self, fig_name='monza', figsize=(14, 8),
             label_corners=True, show_start_marker=True,
             start_marker_style='square',
             title='Autodromo Nazionale di Monza — RL Track Model',
             show_legend=True):
        fig = plt.figure(num=fig_name, figsize=figsize)
        plt.clf()
        for unit in self.unit_list:
            unit.draw_unit(fig_name)

        # Label corners
        labels = {
            0:  'T1 Rettifilo',
            1:  'T2 Exit',
            2:  'Curva Grande',
            3:  'Roggia L',
            4:  'Roggia R',
            5:  'Lesmo 1',
            6:  'Lesmo 2',
            7:  'Ascari Entry',
            8:  'Ascari Mid',
            9:  'Ascari Exit',
            10: 'Parabolica',
        }
        if label_corners:
            for i, unit in enumerate(self.unit_list):
                cx, cy = unit.inlineC1[0], unit.inlineC1[1]
                plt.text(cx, cy, labels[i], fontsize=7, color='blue',
                         ha='center', va='bottom')

        if show_start_marker:
            if start_marker_style == 'line':
                plt.plot([-15, 15], [0, 0], lw=3, c='grey', label='Start/Finish')
            else:
                plt.plot(0, 0, 'rs', markersize=10, label='Start/Finish')
        plt.axis('equal')
        if title:
            plt.title(title)
        if show_legend and show_start_marker:
            plt.legend()
        plt.tight_layout()
        return fig


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    track = MonzaTrackClass()

    # Test findcar at start line
    print("\nTest findcar at start:")
    ok = track.findcar([0, 50])
    print(f"  Found: {ok}  trip={track.car_trip:.1f} m  "
          f"dist={track.centerlinedist:.2f} m")

    # Test lap progress at a few points
    print("\nTest lap progress across circuit:")
    test_points = [
        ([0, 500], "Rettifilo straight"),
        ([0, 960], "After T1"),
    ]
    for pt, name in test_points:
        ok = track.findcar(pt)
        if ok:
            pct = track.car_trip / track.total_trip * 100
            print(f"  {name}: trip={track.car_trip:.0f} m  ({pct:.1f}% lap)")

    # Draw
    fig = track.show()
    plt.show()
