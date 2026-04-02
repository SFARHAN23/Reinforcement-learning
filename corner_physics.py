"""
corner_physics.py
==================
Shared module for corner-aware physics computations used by both
MonzaTrackEnv and SimpleTrackEnv.

Provides two things:
    1. Reference speed lookup — given a circuit trip position, scan forward
       and return the minimum reference speed within a lookahead window.
       This gives the environment the ability to tell the policy
       "the next corner requires you to be at X m/s."

    2. Reward shaping utilities:
       - Grip utilisation bonus (dense signal for efficient tyre use)
       - Reference speed potential (dense braking gradient)
       - Corner exit bonus (sparse bonus for surviving a corner)

Physics
-------
    v_ref = sqrt(mu * g * r)   per corner (conservative, no aero)
    Adding aero would give:  v_ref(vx) = sqrt(mu * (g + F_aero/m) * r)
    We use the conservative no-aero version as the lower bound target.

Why potential-based reward shaping?
------------------------------------
    Potential-based shaping Φ(s) guarantees the optimal policy is
    unchanged (Ng et al., 1999):  r' = r + γΦ(s') - Φ(s)
    This means we cannot accidentally break a converging policy —
    we only add a dense gradient toward the desired behaviour.

    Our potential:   Φ(s) = -max(0, v_x - v_ref_ahead) / v_norm
    Interpretation:  large negative potential when going too fast for
    the upcoming corner, zero when already at or below the target.
    γΦ(s') - Φ(s) is positive when the car slows toward v_ref (correct)
    and negative when it accelerates toward v_ref (discouraged).
"""

import numpy as np

# Physics constants — must match CarModel_Kinematic.py and DynamicActionMapping.py
_MU   = 1.40
_G    = 9.81
_RHO  = 1.225
_S    = 2.50
_CDF  = 1.50
_M    = 1300.0
_VNORM = 90.0   # observation normalisation divisor

# Reward shaping coefficients — tunable
ALPHA_GRIP   = 0.30   # grip utilisation bonus weight
ALPHA_SPEED  = 0.50   # reference speed potential weight
ALPHA_CORNER = 5.00   # corner exit bonus (sparse)
ALPHA_SURVIVAL = 0.10 # per-step survival bonus inside corners

# Dynamic preview horizon for reference speed scan [m]
# We keep a 300 m floor, but expose tighter corners earlier by estimating
# a conservative braking distance from current speed to the corner's v_ref.
VREF_LOOKAHEAD_MIN = 300.0
VREF_LOOKAHEAD_MAX = 700.0
VREF_PREVIEW_BUFFER = 60.0
VREF_PLANNING_DECEL = 8.0
VREF_LOOKAHEAD = VREF_LOOKAHEAD_MIN


def v_ref_for_radius(r: float) -> float:
    """Minimum safe cornering speed for a given corner radius [m/s]."""
    return np.sqrt(_MU * _G * r)


def corner_preview_lookahead(vx: float, radius: float) -> float:
    """
    Preview distance for a given corner, with a 300 m minimum floor.

    This is intentionally more conservative than the car's true peak braking
    capability so the policy starts seeing the braking target early enough to
    learn long-horizon corner entries.
    """
    if radius <= 0.0:
        return VREF_LOOKAHEAD_MIN

    vx = max(0.0, float(vx))
    v_ref = v_ref_for_radius(radius)
    excess_sq = max(0.0, vx * vx - v_ref * v_ref)
    braking_distance = excess_sq / (2.0 * VREF_PLANNING_DECEL)
    lookahead = braking_distance + VREF_PREVIEW_BUFFER
    return float(np.clip(lookahead, VREF_LOOKAHEAD_MIN, VREF_LOOKAHEAD_MAX))


def scan_min_v_ref(track, car_trip: float, vx: float | None = None,
                   lookahead_min: float = VREF_LOOKAHEAD_MIN) -> float:
    """
    Backward-compatible wrapper that returns only the minimum reference speed.
    """
    return scan_v_ref_info(track, car_trip, vx=vx, lookahead_min=lookahead_min)[0]


def scan_v_ref_info(track, car_trip: float, vx: float | None = None,
                    lookahead_min: float = VREF_LOOKAHEAD_MIN) -> tuple[float, float, float]:
    """
    Scan the circuit ahead of car_trip and return the minimum reference
    speed (tightest upcoming corner) within the preview horizon, plus the
    distance to that limiting corner and its radius.

    If `vx` is provided, each corner gets its own preview distance based on
    current speed and that corner's radius-derived target speed. Tight,
    high-speed braking zones therefore appear earlier than wide, gentle turns.
    If `vx` is omitted, the function falls back to the minimum preview floor.

    Works for both SimpleTrackClass and MonzaTrackClass since both expose:
        track.unit_list    — list of TrackUnitClass
        track.total_trip   — total circuit length
        unit.start_trip    — start trip of this unit
        unit.end_trip      — end trip of this unit
        unit.len1          — straight length (arc starts here)
        unit.radius        — corner radius
        unit.angle         — signed corner angle (0 in a straight is not used here)

    Parameters
    ----------
    track       : SimpleTrackClass or MonzaTrackClass
    car_trip    : float  — current trip odometer [m]
    vx          : float or None  — current speed [m/s]
    lookahead_min : float  — minimum preview distance [m]

    Returns
    -------
    (min_v_ref, dist_to_corner, radius)
        min_v_ref       : minimum v_ref within preview horizon [m/s]
        dist_to_corner  : distance from car_trip to the limiting arc start [m]
        radius          : limiting-corner radius [m], or 0 if none found
    """
    total = track.total_trip
    min_v_ref = _VNORM   # default: no corner → no speed restriction
    dist_to_min = float(lookahead_min)
    radius_min = 0.0
    found = False

    for unit in track.unit_list:
        if abs(unit.angle) < 0.01:   # skip near-zero angle straights
            continue

        # Arc starts at: unit.start_trip + unit.len1
        # Arc ends at:   unit.end_trip
        arc_start = unit.start_trip + unit.len1
        arc_end   = unit.end_trip
        preview_dist = lookahead_min if vx is None else corner_preview_lookahead(vx, unit.radius)

        # Check if any part of the arc is within [car_trip, car_trip + preview_dist]
        # using modular arithmetic to handle wrap-around correctly.
        # Convert to "distance ahead of car_trip" space:
        dist_to_arc_start = (arc_start - car_trip) % total
        dist_to_arc_end   = (arc_end   - car_trip) % total

        # The arc is in the scan window if any part of it is within [0, preview_dist]
        # An arc segment is ahead if its start or end is within lookahead,
        # or if it spans across the car position (start > end in modular space)
        arc_in_window = (dist_to_arc_start < preview_dist or
                         dist_to_arc_end < preview_dist or
                         dist_to_arc_end < dist_to_arc_start)  # arc wraps around

        if arc_in_window:
            v_ref = v_ref_for_radius(unit.radius)
            if (not found or
                    v_ref < min_v_ref - 1e-9 or
                    (abs(v_ref - min_v_ref) <= 1e-9 and dist_to_arc_start < dist_to_min)):
                min_v_ref = v_ref
                dist_to_min = float(dist_to_arc_start)
                radius_min = float(unit.radius)
                found = True

    return min_v_ref, dist_to_min, radius_min


def grip_utilisation_bonus(grip_util: float) -> float:
    """
    Dense reward bonus for efficient tyre use.

    Design:
        - Zero below 50% utilisation (not cornering hard enough to matter)
        - Linear rise from 50% to 80% (rewarding committed cornering)
        - Constant peak at 80–92% (the ideal "driving at the limit" zone)
        - Sharp drop above 92% (approaching physical limit — discourage)

    This gives a dense gradient toward "drive at 80-92% of the friction limit."

    Parameters
    ----------
    grip_util : float ∈ [0, 1]  — grip utilisation from ActionMappingResult

    Returns
    -------
    float — reward bonus ∈ [0, ALPHA_GRIP]
    """
    if grip_util < 0.50:
        return 0.0
    elif grip_util < 0.80:
        return ALPHA_GRIP * (grip_util - 0.50) / 0.30
    elif grip_util < 0.92:
        return ALPHA_GRIP
    else:
        # Penalise near-violation zone
        return ALPHA_GRIP * max(0.0, 1.0 - (grip_util - 0.92) / 0.08)


def speed_potential(vx: float, v_ref_ahead: float,
                    gamma: float = 0.99) -> tuple[float, float]:
    """
    Compute the speed-based potential Φ(s) for potential-based reward shaping.

    Φ(s) = -ALPHA_SPEED * max(0, vx - v_ref_ahead) / _VNORM

    The shaped reward component is:  γ·Φ(s') - Φ(s)
    This is positive (reward) when the car slows toward v_ref_ahead and
    negative (penalty) when it accelerates beyond v_ref_ahead.

    Parameters
    ----------
    vx          : float — current speed [m/s]
    v_ref_ahead : float — minimum reference speed in lookahead window [m/s]
    gamma       : float — discount factor (default: 0.99)

    Returns
    -------
    (phi_current, gamma) — caller computes gamma*phi_next - phi_current
    """
    phi = -ALPHA_SPEED * max(0.0, vx - v_ref_ahead) / _VNORM
    return phi, gamma


def corner_exit_bonus(was_in_corner: bool, is_in_corner: bool,
                      vx: float, v_ref: float) -> float:
    """
    Sparse bonus awarded when the car exits a corner section successfully.

    A 'corner exit' is detected when:
        - Previous step: car was in a curve section (was_in_corner=True)
        - Current step:  car has moved to a straight (is_in_corner=False)
        - Speed at exit: vx is within 1.5x v_ref (didn't slow excessively)

    Parameters
    ----------
    was_in_corner : bool  — previous step was in a curve
    is_in_corner  : bool  — current step is in a curve
    vx            : float — current speed [m/s]
    v_ref         : float — reference speed for the corner just exited [m/s]

    Returns
    -------
    float — ALPHA_CORNER if corner exit detected, else 0.0
    """
    exiting = was_in_corner and not is_in_corner
    speed_ok = vx > 0.5 * v_ref   # didn't stall
    return ALPHA_CORNER if (exiting and speed_ok) else 0.0


def corner_survival_bonus(is_in_corner: bool, radius: float) -> float:
    """
    Small per-step reward for surviving inside a corner section.

    Provides a dense positive signal at tight corners where the agent
    otherwise only receives -100 for dying.  Weighted by 1/sqrt(radius)
    so tighter corners (chicanes) are more rewarding per step survived.

    Parameters
    ----------
    is_in_corner : bool   — car is currently in a curve section
    radius       : float  — radius of the current corner [m]

    Returns
    -------
    float — ALPHA_SURVIVAL * weight if in corner, else 0.0
    """
    if not is_in_corner or radius <= 0:
        return 0.0
    # weight: tight corners (r=15) get ~0.26, wide corners (r=200) get ~0.07
    weight = 1.0 / np.sqrt(radius)
    return ALPHA_SURVIVAL * weight


if __name__ == '__main__':
    print('corner_physics.py — self test')
    print(f'  v_ref T1 (r=15m):         {v_ref_for_radius(15):.1f} m/s')
    print(f'  v_ref Roggia (r=22m):     {v_ref_for_radius(22):.1f} m/s')
    print(f'  v_ref Parabolica (r=192m):{v_ref_for_radius(192):.1f} m/s')
    print()
    print('Dynamic preview distance examples:')
    for vx in [30.0, 60.0, 85.0]:
        print(f'  vx={vx:4.1f} m/s  T1={corner_preview_lookahead(vx, 15):6.1f} m  '
              f'Roggia={corner_preview_lookahead(vx, 22):6.1f} m  '
              f'Parabolica={corner_preview_lookahead(vx, 192):6.1f} m')
    print()
    print('Grip utilisation bonus curve:')
    for g in [0.0, 0.5, 0.65, 0.80, 0.90, 0.92, 0.96, 1.0]:
        print(f'  util={g:.2f}  bonus={grip_utilisation_bonus(g):.3f}')
