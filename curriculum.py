"""
curriculum.py
=============
Shared corner-curriculum spawn logic for both SimpleTrack and Monza.

Why this exists
---------------
Both policies were stuck because the replay buffer was poisoned:
  - FIXED episodes crash at the first corner -> Q(corner) = -100 -> permanent trap
  - CORNER spawn 33m before entry still fails because the policy throttles
    into the corner during the run-up, arriving well above v_max

The correct curriculum strategy:
  1. Spawn AT the corner entry (3m before), not 33m before
  2. Spawn speed = 55% of v_max for that corner (always below the limit)
  3. Cycle through ALL corners, not just T1
  4. Drop FIXED episodes from the rotation while stuck - they actively harm training
  5. Include RAND for general coverage
  6. Re-add FIXED once the policy is consistently past 30% lap from corner spawns

Spawn mode rotation (CORNER_COUNT corners + 1 RAND block, then repeat)
  block 0..N-1 : one block per corner (in circuit order)
  block N      : RAND (general coverage)
  -> total cycle length = (N+1) * SPAWN_BLOCK episodes

Fine-tune rotation
  After the corner curriculum is stable, switch to a start-line-biased phase:
    - mostly FIXED start episodes (real start/finish transfer)
    - one long approach spawn for the first hard braking zone
    - one RAND block for generalisation
  This keeps the early-curriculum benefits while teaching the policy to brake
  correctly from the real launch straight.

Physics verification per corner
  v_max = sqrt(mu * g * r)   (conservative: no aero, worst case)
  spawn_spd = 0.55 * v_max   (55% of limit, well within physics)
  spawn_trip = corner_entry - 3  (arrive at apex with minimal run-up)
"""

import numpy as np


# SimpleTrack corner table
# Each entry: (label, trip_at_curve_entry, corner_radius_m)
# Trips estimated from TrackUnit geometry (straight_len + prior arcs).
SIMPLE_CORNERS = [
    # label            entry_trip   radius
    ('T1 unit0',         105.3,     30),
    ('T2 unit1',         302.4,     15),
    ('T3 unit2',         461.7,     20),
    ('T4 unit3',         600.5,     20),
    ('T5 unit4',         813.3,     30),
]

# Monza corner table
# Each entry stores the true ARC START trip from the current MonzaTrack
# geometry. corner_spawn_params() later applies the run-up offset:
#     spawn_trip = entry_trip - run_up
MONZA_CORNERS = [
    # label            entry_trip   radius
    ('T1 Rettifilo',      903.515,    15),
    ('T2 Exit',           939.695,    15),
    ('Curva Grande',     1217.969,   450),
    ('Roggia L',         2444.098,    22),
    ('Roggia R',         2496.735,    22),
    ('Lesmo 1',          2948.221,    85),
    ('Lesmo 2',          3269.101,    60),
    ('Ascari L',         4354.971,    50),
    ('Ascari M',         4428.039,    65),
    ('Ascari X',         4548.754,   110),
    ('Parabolica',       5437.786,   192),
]

# Start-line fine-tune approach spawns.
# Each entry: (label, corner_entry_trip, corner_radius_m, approach_distance_m)
# Spawn speed is computed from a conservative braking-distance model so the
# agent must brake meaningfully, but the task is still learnable.
SIMPLE_FINE_TUNE_APPROACHES = [
    ('T1 approach',       105.3,      30,   70.0),
]

MONZA_FINE_TUNE_APPROACHES = [
    ('T1 approach',       903.515,    15,  450.0),
]

MU, G = 1.4, 9.81
COMPLEX_GAP_DEFAULT = 80.0


def corner_spawn_params(corners, run_up=3.0, speed_fraction=0.55,
                        max_spd=12.0, min_spd=10.0):
    """
    Compute (trip, speed, label) for each corner.

    Parameters
    ----------
    corners       : list of (label, entry_trip, radius)
    run_up        : metres before corner entry to place spawn point [m]
    speed_fraction: fraction of v_max to use as spawn speed
    max_spd       : hard cap on spawn speed [m/s]
    min_spd       : floor on spawn speed [m/s] - must be > _STOP_SPEED

    Returns
    -------
    list of (spawn_trip, spawn_spd, label, v_max)
    """
    result = []
    for label, entry_trip, r in corners:
        v_max = np.sqrt(MU * G * r)
        spawn_spd = np.clip(speed_fraction * v_max, min_spd, max_spd)
        spawn_trip = max(0.0, entry_trip - run_up)
        result.append((spawn_trip, spawn_spd, label, v_max))
    return result


def build_spawn_schedule(corners, spawn_block):
    """
    Build the full episode -> spawn mapping.

    Returns a function: episode_number -> (mode, trip, spd, label)
    where mode is 'corner' or 'rand'.
    """
    params = corner_spawn_params(corners)
    n_corners = len(params)
    cycle_len = (n_corners + 1) * spawn_block   # +1 for RAND block

    def get_spawn(episode):
        block_idx = (episode // spawn_block) % (n_corners + 1)
        if block_idx < n_corners:
            trip, spd, label, vmax = params[block_idx]
            return 'corner', trip, spd, f'CRN:{label[:8]}'
        return 'rand', None, None, 'RAND    '

    return get_spawn, params


def approach_spawn_params(approaches, brake_decel=4.0,
                          min_spd=18.0, max_spd=75.0):
    """
    Compute approach-spawn trip/speed pairs for fixed-start fine-tuning.

    Spawn speed is chosen so the car must brake into the corner over the
    provided approach distance using a conservative planning deceleration.

    Parameters
    ----------
    approaches   : list of (label, entry_trip, radius, approach_distance)
    brake_decel  : assumed braking decel used to size the spawn speed [m/s^2]
    min_spd      : lower clamp on spawn speed [m/s]
    max_spd      : upper clamp on spawn speed [m/s]

    Returns
    -------
    list of (spawn_trip, spawn_spd, label, v_max, approach_distance)
    """
    result = []
    for label, entry_trip, r, approach_dist in approaches:
        v_max = np.sqrt(MU * G * r)
        spawn_spd = np.sqrt(v_max * v_max + 2.0 * brake_decel * approach_dist)
        spawn_spd = float(np.clip(spawn_spd, min_spd, max_spd))
        spawn_trip = max(0.0, entry_trip - approach_dist)
        result.append((spawn_trip, spawn_spd, label, v_max, approach_dist))
    return result


def bridge_spawn_params(approaches, distance_fraction=0.20,
                        min_distance=35.0, max_distance=80.0,
                        speed_fraction=0.85, min_spd=10.0, max_spd=22.0):
    """
    Build short "bridge" spawns close to the corner entry.

    These are meant to teach the second half of the braking/turn-in task
    without starting all the way from the long straight. The construction is
    track-agnostic: every long approach gets a shorter derivative spawn.

    Parameters
    ----------
    approaches        : list of (label, entry_trip, radius, approach_distance)
    distance_fraction : fraction of long-approach distance to keep
    min_distance      : lower clamp on bridge run-up [m]
    max_distance      : upper clamp on bridge run-up [m]
    speed_fraction    : spawn speed as fraction of corner v_max
    min_spd/max_spd   : speed clamps [m/s]

    Returns
    -------
    list of (spawn_trip, spawn_spd, label, v_max, bridge_distance)
    """
    result = []
    for label, entry_trip, r, approach_dist in approaches:
        v_max = np.sqrt(MU * G * r)
        bridge_dist = float(np.clip(
            distance_fraction * approach_dist, min_distance, max_distance
        ))
        spawn_spd = float(np.clip(speed_fraction * v_max, min_spd, max_spd))
        spawn_trip = max(0.0, entry_trip - bridge_dist)
        result.append((spawn_trip, spawn_spd, label, v_max, bridge_dist))
    return result


def build_finetune_schedule(approaches, spawn_block,
                            fixed_blocks=4, bridge_blocks=1, rand_blocks=1):
    """
    Build a start-line transfer schedule with FIXED, APPROACH, and RAND blocks.

    Returns a function: phase_episode -> (mode, trip, spd, label)
    where mode is 'fixed', 'approach', or 'rand'.
    """
    approach_params = approach_spawn_params(approaches)
    bridge_params = bridge_spawn_params(approaches)
    blocks = [('fixed', None, None, 'FIXED   ')] * fixed_blocks
    blocks += [('approach', trip, spd, f'APP:{label[:8]}')
               for trip, spd, label, _, _ in approach_params]
    blocks += [('bridge', trip, spd, f'BRG:{label[:8]}')
               for _ in range(bridge_blocks)
               for trip, spd, label, _, _ in bridge_params]
    blocks += [('rand', None, None, 'RAND    ')] * rand_blocks

    def get_spawn(phase_episode):
        block_idx = (phase_episode // spawn_block) % len(blocks)
        return blocks[block_idx]

    params = [('approach',) + p for p in approach_params]
    params += [('bridge',) + p for p in bridge_params]
    return get_spawn, params


def wrapped_trip_distance(total_trip, start_trip, end_trip):
    """Positive modular distance from start_trip to end_trip."""
    return float((end_trip - start_trip) % total_trip)


def next_corner_complex_exit(track, trip, gap_threshold=COMPLEX_GAP_DEFAULT):
    """
    Return the exit trip of the next corner complex after `trip`.

    A "corner complex" is one or more consecutive corner units separated by
    straights shorter than `gap_threshold`. This generalises across tracks and
    naturally groups chicanes such as Monza T1/T2 into a single local target.

    Returns
    -------
    (target_trip, first_idx, last_idx)
    """
    total = track.total_trip
    best_idx, best_dist = None, None

    for idx, unit in enumerate(track.unit_list):
        if abs(unit.angle) < 0.01:
            continue
        arc_start = unit.start_trip + unit.len1
        dist = (arc_start - trip) % total
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_idx = idx

    if best_idx is None:
        return float(trip), None, None

    last_idx = best_idx
    while True:
        next_idx = (last_idx + 1) % len(track.unit_list)
        if next_idx == best_idx:
            break

        current = track.unit_list[last_idx]
        nxt = track.unit_list[next_idx]
        next_entry = nxt.start_trip + nxt.len1
        gap = (next_entry - current.end_trip) % total
        if gap > gap_threshold:
            break
        last_idx = next_idx

    return float(track.unit_list[last_idx].end_trip), best_idx, last_idx


if __name__ == '__main__':
    print('=== SimpleTrack corner spawn schedule ===')
    params = corner_spawn_params(SIMPLE_CORNERS)
    for trip, spd, label, vmax in params:
        print(f'  {label:<12}  entry~{trip+3:.0f}m  spawn@{trip:.0f}m  '
              f'spd={spd:.1f} m/s  ({spd/vmax*100:.0f}% of v_max={vmax:.1f})')

    print()
    print('=== Monza corner spawn schedule ===')
    params = corner_spawn_params(MONZA_CORNERS)
    for trip, spd, label, vmax in params:
        print(f'  {label:<16}  entry~{trip+3:.0f}m  spawn@{trip:.0f}m  '
              f'spd={spd:.1f} m/s  ({spd/vmax*100:.0f}% of v_max={vmax:.1f})')

    print()
    print('=== Fine-tune approach spawns ===')
    for name, approaches in [('SimpleTrack', SIMPLE_FINE_TUNE_APPROACHES),
                             ('Monza', MONZA_FINE_TUNE_APPROACHES)]:
        print(f'  {name}:')
        approach_params = approach_spawn_params(approaches)
        for trip, spd, label, vmax, dist in approach_params:
            print(f'    APP {label:<8}  approach={dist:.0f}m  spawn@{trip:.0f}m  '
                  f'spd={spd:.1f} m/s  (corner v_max={vmax:.1f})')
        bridge_params = bridge_spawn_params(approaches)
        for trip, spd, label, vmax, dist in bridge_params:
            print(f'    BRG {label:<8}  bridge={dist:.0f}m   spawn@{trip:.0f}m  '
                  f'spd={spd:.1f} m/s  (corner v_max={vmax:.1f})')
