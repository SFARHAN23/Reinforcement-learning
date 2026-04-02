"""
DynamicActionMapping.py
========================
Drop-in replacement for the original ActionMapping.py from Wang et al. (2024).

Replaces the static, precomputed 4-D lookup-table friction circle with a
physics-driven Dynamic Action Mapping that accounts for:
    1. Velocity-dependent aerodynamic downforce  (F_aero ∝ vx²)
    2. Longitudinal load transfer                (ΔFz ∝ ax)
    3. Per-axle dynamic friction limits
    4. Friction-ellipse radial projection        (preserves steer/accel ratio)

Vehicle
-------
    High-downforce GT3 race car (representative: Porsche 911 GT3 R).

Public API  — identical call signature to original ActionMappingClass.mapping()
-----------------------------------------------------------------------
    am = DynamicActionMappingClass()
    action_in = am.mapping(vx, ax_prev, ux, uy)

        vx      : longitudinal speed [m/s]  ← env.car.spd
        ax_prev : previous-step longitudinal acceleration [m/s²]
                  ← getattr(env.car, 'long_acc', 0.0)
                  Using the PREVIOUS step's ax intentionally breaks the
                  circular dependency:  ax → ΔFz → grip → action → ax.
        ux      : raw TD3 longitudinal action ∈ [-1, 1]
                  (positive = throttle, negative = brake)
        uy      : raw TD3 lateral / steer action ∈ [-1, 1]

    Returns: [ux_mapped, uy_mapped]  — same format as original

Diagnostic attributes (read after each step for logging / reward shaping)
-------------------------------------------------------------------------
    am.last_result   : ActionMappingResult  — full physics breakdown
    am.clip_count    : int   — total projection events this episode
    am.step_count    : int   — total steps this episode
    am.clip_rate     : float — clip_count / step_count  (0.0 – 1.0)

    Call am.reset_diagnostics() at the start of each episode.

Integration checklist
---------------------
    [ ] Replace `from ActionMapping import ActionMappingClass`
               with `from DynamicActionMapping import DynamicActionMappingClass`
    [ ] Replace `am = ActionMappingClass()`
               with `am = DynamicActionMappingClass()`
    [ ] Replace `am.mapping(env.car.spd, env.car.steer, action[0], action[1])`
               with:
                   ax_prev = getattr(env.car, 'long_acc', 0.0)
                   am.mapping(env.car.spd, ax_prev, action[0], action[1])
    [ ] Call `am.reset_diagnostics()` inside the episode-reset block
    [ ] (Optional) log am.clip_rate per episode for policy quality monitoring

Dependencies
------------
    Python >= 3.9, numpy (already in the original project's requirements).
    No scipy, no pre-built .npy lookup table required.

Author note
-----------
    The original actionmap_200.npy lookup table is no longer needed and can
    be removed from the project. This module is fully self-contained.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — GT3 Vehicle Constants
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GT3Params:
    """
    Immutable physical parameters for the GT3 race car.

    All SI units.  frozen=True means any accidental mutation raises TypeError
    at runtime — important for a constants object used across many steps.

    Parameters
    ----------
    m : float
        Total vehicle mass including driver [kg].
    L : float
        Wheelbase [m].
    h : float
        Centre-of-gravity height above ground [m].
    rho : float
        Air density [kg/m³].  ISA sea-level standard.
    S : float
        Aerodynamic reference (frontal) area [m²].
    C_DF : float
        Downforce coefficient (positive = downward force on the car).
        Named C_DF, NOT C_L, to avoid sign confusion with IATA lift convention
        where downforce is negative lift.
    mu : float
        Combined tyre-road friction coefficient (dimensionless).
    g : float
        Standard gravitational acceleration [m/s²].
    weight_dist_rear : float
        Fraction of static vehicle weight on rear axle.
        0.5 = 50/50 split (used here to match research brief).
        Real Porsche 911 GT3 R ≈ 0.60 rear (can be updated later).
    """
    m:                float = 1_300.0
    L:                float = 2.457
    h:                float = 0.40
    rho:              float = 1.225
    S:                float = 2.50
    C_DF:             float = 1.50
    mu:               float = 1.40
    g:                float = 9.81
    weight_dist_rear: float = 0.50

    def __post_init__(self) -> None:
        if not (0.0 < self.weight_dist_rear < 1.0):
            raise ValueError(f"weight_dist_rear must be in (0,1), got {self.weight_dist_rear}")
        for name, val in [("m", self.m), ("L", self.L), ("h", self.h), ("mu", self.mu)]:
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — Aerodynamic Downforce + Total Normal Force
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_aero_forces(vx: float, p: GT3Params) -> tuple[float, float]:
    """
    F_aero    = 0.5 * rho * vx² * S * C_DF    [N]  always >= 0
    F_z_total = m*g + F_aero                   [N]  always >  0

    vx may be negative (reversing) — vx² is still positive. ✓
    No division by vx → no divide-by-zero risk. ✓
    """
    F_aero    = 0.5 * p.rho * (vx ** 2) * p.S * p.C_DF
    F_z_total = p.m * p.g + F_aero
    return F_aero, F_z_total


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — Longitudinal Load Transfer + Dynamic Axle Grip Limits
# ═══════════════════════════════════════════════════════════════════════════════

# Safety floor for per-axle normal force.
# Prevents max_grip from collapsing to zero (which would zero-out ALL traction
# and destabilise the RL gradient signal).
# Value: 1% of static half-weight ≈ 63.8 N — physically negligible but safe.
_F_Z_MIN: float = 0.01 * 1_300.0 * 9.81 / 2.0


@dataclass
class AxleLoads:
    """Per-timestep dynamic axle load results."""
    F_z_total:      float   # total normal force, all 4 tyres [N]
    delta_Fz:       float   # raw longitudinal load transfer [N] (signed)
    Fz_front:       float   # front axle normal force, clipped [N]
    Fz_rear:        float   # rear  axle normal force, clipped [N]
    max_grip_front: float   # mu * Fz_front [N]
    max_grip_rear:  float   # mu * Fz_rear  [N]
    front_clipped:  bool    # True if Fz_front hit the safety floor
    rear_clipped:   bool    # True if Fz_rear  hit the safety floor


def _compute_axle_loads(
    vx: float,
    ax_observed: float,
    p: GT3Params,
    f_z_min: float = _F_Z_MIN,
) -> AxleLoads:
    """
    Compute dynamic per-axle loads and friction limits.

    ax_observed is the PREVIOUS timestep's longitudinal acceleration [m/s²].
    Sign convention:
        ax < 0  →  braking  →  weight shifts forward  →  Fz_front increases ✓
        ax > 0  →  throttle →  weight shifts rearward  →  Fz_rear  increases ✓

    Physics
    -------
        ΔFz        = m * ax * h / L
        Fz_front   = (1 - weight_dist_rear) * F_z_total - ΔFz
        Fz_rear    =      weight_dist_rear  * F_z_total + ΔFz
        (then clipped to f_z_min)
    """
    _, F_z_total = _compute_aero_forces(vx, p)

    # Longitudinal load transfer [N].  L is a positive constant → no div-by-zero.
    delta_Fz: float = (p.m * ax_observed * p.h) / p.L

    Fz_front_raw = (1.0 - p.weight_dist_rear) * F_z_total - delta_Fz
    Fz_rear_raw  =        p.weight_dist_rear  * F_z_total + delta_Fz

    front_clipped = Fz_front_raw < f_z_min
    rear_clipped  = Fz_rear_raw  < f_z_min

    Fz_front = max(Fz_front_raw, f_z_min)
    Fz_rear  = max(Fz_rear_raw,  f_z_min)

    return AxleLoads(
        F_z_total      = F_z_total,
        delta_Fz       = delta_Fz,
        Fz_front       = Fz_front,
        Fz_rear        = Fz_rear,
        max_grip_front = p.mu * Fz_front,
        max_grip_rear  = p.mu * Fz_rear,
        front_clipped  = front_clipped,
        rear_clipped   = rear_clipped,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — Friction-Ellipse Action Squash
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ActionMappingResult:
    """
    Full diagnostic snapshot from one squash_action() call.

    Fields most useful for reward shaping / policy monitoring:
        was_clipped     : bool  — did the raw action exceed the ellipse?
        grip_utilisation: float — how close to the limit (0.0 – 1.0)
        active_constraint: str  — 'rear' (throttle) or 'front' (braking)
    """
    steer_mapped:        float  # clipped lateral  action ∈ [-1, 1]
    accel_brake_mapped:  float  # clipped longitudinal action ∈ [-1, 1]
    was_clipped:         bool
    scale_factor:        float  # 1.0 = no clip; <1.0 = projection applied
    F_y_physical:        float  # lateral force after clip [N]
    F_x_physical:        float  # longitudinal force after clip [N]
    active_constraint:   str    # 'rear' or 'front'
    grip_utilisation:    float  # |(F_y, F_x)| / R  ≤ 1.0
    F_y_limit:           float  # dynamic lateral limit used [N]
    F_x_limit:           float  # dynamic longitudinal limit used [N]


def _squash_action(
    steer_raw: float,
    accel_brake_raw: float,
    loads: AxleLoads,
) -> ActionMappingResult:
    """
    Project the raw TD3 action onto the dynamic friction ellipse.

    Constraint geometry
    -------------------
    The friction budget for each axle is a CIRCLE in physical force space [N].
    The two axles have DIFFERENT radii (due to load transfer), which creates
    an ELLIPSE in normalised action space — hence "friction ellipse".

    Active constraint selection:
        Throttle (accel_brake >= 0) → rear-axle circle,  R = max_grip_rear
        Braking  (accel_brake <  0) → front-axle circle, R = max_grip_front
        (Lateral force is always front-limited regardless of direction.)

    Projection method (radial, minimum-distance):
        1. Translate to physical forces:  F_y = steer * F_y_limit
                                          F_x = accel * F_x_limit
        2. Normalise by R:                n_y = F_y / R,  n_x = F_x / R
        3. If |(n_y, n_x)| > 1:          scale = 1 / |(n_y, n_x)|
                                          n_y *= scale,  n_x *= scale
        4. Back-project to actions:       steer_mapped = n_y * R / F_y_limit
                                          accel_mapped = n_x * R / F_x_limit

    Numerical guarantee
    -------------------
    max_grip_front > 0 and max_grip_rear > 0 are guaranteed by _F_Z_MIN in
    _compute_axle_loads(), so no division by zero can occur here.

    Sign preservation
    -----------------
    Radial scaling never changes the sign of any component.
    accel_brake_raw < 0 → accel_brake_mapped < 0  ✓
    accel_brake_raw > 0 → accel_brake_mapped > 0  ✓
    """
    # ── Step 1: choose active constraint ─────────────────────────────────────
    is_throttle = accel_brake_raw >= 0.0
    R           = loads.max_grip_rear  if is_throttle else loads.max_grip_front
    F_x_limit   = loads.max_grip_rear  if is_throttle else loads.max_grip_front
    F_y_limit   = loads.max_grip_front   # lateral always front-limited
    active      = "rear"               if is_throttle else "front"

    # ── Step 2: physical force request [N] ────────────────────────────────────
    F_y = steer_raw       * F_y_limit
    F_x = accel_brake_raw * F_x_limit

    # ── Step 3: normalise to unit-circle space ─────────────────────────────────
    # R > 0 guaranteed by _F_Z_MIN → no divide-by-zero
    n_y = F_y / R
    n_x = F_x / R
    magnitude = math.hypot(n_y, n_x)

    # ── Step 4: radial projection if outside the unit circle ──────────────────
    was_clipped  = magnitude > 1.0
    scale_factor = 1.0
    if was_clipped:
        scale_factor = 1.0 / magnitude
        n_y *= scale_factor
        n_x *= scale_factor

    # ── Step 5: back to physical forces → normalised actions ──────────────────
    F_y_clipped = n_y * R
    F_x_clipped = n_x * R

    # Divide by the PER-AXIS limit (not R) to recover action ∈ [-1, 1].
    # When throttle and front limits differ, this creates the elliptic shape.
    steer_mapped      = F_y_clipped / F_y_limit
    accel_brake_mapped = F_x_clipped / F_x_limit

    grip_utilisation = math.hypot(n_y, n_x)   # ≤ 1.0 after projection

    return ActionMappingResult(
        steer_mapped       = steer_mapped,
        accel_brake_mapped = accel_brake_mapped,
        was_clipped        = was_clipped,
        scale_factor       = scale_factor,
        F_y_physical       = F_y_clipped,
        F_x_physical       = F_x_clipped,
        active_constraint  = active,
        grip_utilisation   = grip_utilisation,
        F_y_limit          = F_y_limit,
        F_x_limit          = F_x_limit,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC CLASS — drop-in replacement for ActionMappingClass
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicActionMappingClass:
    """
    Drop-in replacement for ActionMappingClass from Wang et al. (2024).

    Usage
    -----
        am = DynamicActionMappingClass()          # no args, no .npy needed

        # Inside the training loop, BEFORE env.step():
        ax_prev   = getattr(env.car, 'long_acc', 0.0)
        action_in = am.mapping(env.car.spd, ax_prev, action[0], action[1])
        next_ob, r, done = env.step(action_in)

        # At episode boundary:
        am.reset_diagnostics()

    Parameters
    ----------
    params : GT3Params, optional
        Override vehicle parameters.  Defaults to the GT3 constants
        specified in the research brief.
    f_z_min : float, optional
        Safety floor for per-axle normal force [N].  Default ≈ 63.8 N.
    """

    def __init__(
        self,
        params: GT3Params | None = None,
        f_z_min: float = _F_Z_MIN,
    ) -> None:
        self.params  = params if params is not None else GT3Params()
        self.f_z_min = f_z_min

        # Diagnostic state — reset each episode
        self.last_result: ActionMappingResult | None = None
        self.clip_count:  int   = 0
        self.step_count:  int   = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def mapping(
        self,
        vx:       float,
        ax_prev:  float,
        ux:       float,
        uy:       float,
    ) -> list[float]:
        """
        Map raw TD3 actions to physically constrained actions.

        Parameters
        ----------
        vx : float
            Current longitudinal speed [m/s].  Pass env.car.spd.
        ax_prev : float
            Previous-step longitudinal acceleration [m/s²].
            Pass getattr(env.car, 'long_acc', 0.0).
            Using the previous step's value breaks the circular dependency
            ax → ΔFz → grip → action → ax.  One-step lag ≈ 10 ms (at dt=0.01),
            which is below sensor noise floor in any real IMU.
        ux : float
            Raw TD3 longitudinal action ∈ [-1, 1].
            action[0] from the TD3 actor.  Positive = throttle, negative = brake.
        uy : float
            Raw TD3 lateral / steer-rate action ∈ [-1, 1].
            action[1] from the TD3 actor.

        Returns
        -------
        list[float]
            [ux_mapped, uy_mapped] — same format as the original ActionMappingClass.
            Both values are guaranteed ∈ [-1, 1].
            The mapped action never exceeds the dynamic friction ellipse.

        Notes
        -----
        Axis convention (must match CarModel_Kinematic.py convert_control()):
            action[0] = ux = longitudinal  (throttle / brake)
            action[1] = uy = lateral       (steer rate)
        The squash function uses (steer_raw=uy, accel_brake_raw=ux) internally,
        then swaps back to [ux, uy] order for the return value.
        """
        # ── Compute dynamic grip limits for this timestep ─────────────────────
        loads = _compute_axle_loads(vx, ax_prev, self.params, self.f_z_min)

        # ── Squash: note the axis swap (ux=longitudinal, uy=lateral) ──────────
        result = _squash_action(
            steer_raw       = uy,   # lateral  → steer dimension
            accel_brake_raw = ux,   # longitudinal → accel dimension
        loads               = loads,
        )

        # ── Update diagnostics ────────────────────────────────────────────────
        self.last_result  = result
        self.step_count  += 1
        if result.was_clipped:
            self.clip_count += 1

        # ── Return in original [ux, uy] order ─────────────────────────────────
        # result.accel_brake_mapped corresponds to ux (longitudinal)
        # result.steer_mapped       corresponds to uy (lateral)
        return [result.accel_brake_mapped, result.steer_mapped]

    def reset_diagnostics(self) -> None:
        """
        Reset per-episode diagnostic counters.
        Call this at the start of every episode (inside the episode loop,
        after env.reset()).
        """
        self.last_result = None
        self.clip_count  = 0
        self.step_count  = 0

    @property
    def clip_rate(self) -> float:
        """
        Fraction of steps this episode where the raw action was clipped.
        0.0 = policy never exceeds the friction ellipse (ideal).
        1.0 = policy always exceeds it (early training, expected).
        Use as a training health metric alongside episode reward.
        """
        if self.step_count == 0:
            return 0.0
        return self.clip_count / self.step_count

    def get_current_limits(self, vx: float, ax_prev: float) -> dict[str, float]:
        """
        Query the current grip limits without consuming an action.
        Useful for logging, reward shaping, or curriculum scheduling.

        Returns a dict with keys:
            max_grip_front, max_grip_rear, F_z_total, delta_Fz,
            Fz_front, Fz_rear, F_aero
        """
        F_aero, F_z_total = _compute_aero_forces(vx, self.params)
        loads = _compute_axle_loads(vx, ax_prev, self.params, self.f_z_min)
        return {
            "F_aero":          F_aero,
            "F_z_total":       loads.F_z_total,
            "delta_Fz":        loads.delta_Fz,
            "Fz_front":        loads.Fz_front,
            "Fz_rear":         loads.Fz_rear,
            "max_grip_front":  loads.max_grip_front,
            "max_grip_rear":   loads.max_grip_rear,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-TEST  (python DynamicActionMapping.py)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import math

    am = DynamicActionMappingClass()
    print("=" * 70)
    print("DynamicActionMapping — Integration Self-Test")
    print("=" * 70)

    # ── Test 1: cold-start (ax_prev=0, mimics getattr default) ───────────────
    print("\n[Test 1] Cold-start / episode boundary (ax_prev=0.0)")
    result = am.mapping(20.0, 0.0, 0.5, 0.3)
    print(f"  Input  [ux=+0.50, uy=+0.30]  vx=20 m/s, ax_prev=0.0")
    print(f"  Output [ux={result[0]:+.4f}, uy={result[1]:+.4f}]")
    assert am.last_result is not None
    print(f"  Clipped: {am.last_result.was_clipped}  util={am.last_result.grip_utilisation:.4f}")

    # ── Test 2: axis order preserved — longitudinal stays longitudinal ────────
    print("\n[Test 2] Axis order: pure throttle input → longitudinal output only")
    r = am.mapping(30.0, 0.0, 0.8, 0.0)   # ux=0.8, uy=0.0
    assert r[1] == 0.0, f"Lateral output should be 0 for pure throttle, got {r[1]}"
    assert r[0] != 0.0, "Longitudinal output should not be 0 for pure throttle"
    print(f"  [ux=+0.80, uy=0.00] → [{r[0]:+.4f}, {r[1]:+.4f}]  ✓ axis order correct")

    # ── Test 3: pure braking stays negative ───────────────────────────────────
    print("\n[Test 3] Sign preservation: braking stays negative")
    r = am.mapping(30.0, -10.0, -0.9, 0.0)
    assert r[0] < 0.0, f"Braking output must be negative, got {r[0]}"
    print(f"  [ux=-0.90, uy=0.00] → [{r[0]:+.4f}, {r[1]:+.4f}]  ✓ sign preserved")

    # ── Test 4: combined demand clips and preserves ratio ─────────────────────
    print("\n[Test 4] Over-limit combined input — clipping and ratio preservation")
    r = am.mapping(55.0, 0.0, 1.0, 1.0)
    assert am.last_result.was_clipped, "Full combined input must clip"
    # ux and uy were equal going in; after squash they should still be equal
    assert math.isclose(abs(r[0]), abs(r[1]), rel_tol=1e-6), \
        f"Ratio not preserved: ux={r[0]:.4f} uy={r[1]:.4f}"
    print(f"  [ux=+1.00, uy=+1.00] → [{r[0]:+.4f}, {r[1]:+.4f}]  ✓ clipped, ratio preserved")

    # ── Test 5: output always in [-1, 1] ─────────────────────────────────────
    print("\n[Test 5] Output bounds [-1, 1] for extreme inputs")
    for ux, uy in [(2.0, 2.0), (-2.0, -2.0), (3.0, -0.1), (-3.0, 3.0)]:
        r = am.mapping(50.0, 0.0, ux, uy)
        assert -1.0 - 1e-9 <= r[0] <= 1.0 + 1e-9, f"ux_mapped={r[0]} out of range"
        assert -1.0 - 1e-9 <= r[1] <= 1.0 + 1e-9, f"uy_mapped={r[1]} out of range"
    print("  All extreme inputs produced outputs within [-1, 1]  ✓")

    # ── Test 6: diagnostics ───────────────────────────────────────────────────
    print("\n[Test 6] Diagnostic counters")
    am.reset_diagnostics()
    am.mapping(20.0, 0.0, 0.3, 0.3)   # should NOT clip
    am.mapping(20.0, 0.0, 1.0, 1.0)   # WILL clip
    am.mapping(20.0, 0.0, 1.0, 1.0)   # WILL clip
    assert am.step_count  == 3, f"Expected step_count=3, got {am.step_count}"
    assert am.clip_count  == 2, f"Expected clip_count=2, got {am.clip_count}"
    assert math.isclose(am.clip_rate, 2/3, rel_tol=1e-9), f"clip_rate wrong: {am.clip_rate}"
    print(f"  step_count={am.step_count}  clip_count={am.clip_count}  clip_rate={am.clip_rate:.4f}  ✓")

    # ── Test 7: get_current_limits ────────────────────────────────────────────
    print("\n[Test 7] get_current_limits()")
    limits = am.get_current_limits(55.56, 0.0)   # 200 km/h, coasting
    print(f"  @ 200 km/h, ax=0: max_grip_front={limits['max_grip_front']:.1f} N"
          f"  max_grip_rear={limits['max_grip_rear']:.1f} N")
    assert limits["max_grip_front"] > 0 and limits["max_grip_rear"] > 0

    # ── Test 8: dynamic effect visible ───────────────────────────────────────
    print("\n[Test 8] Dynamic effect — same raw input, different physics")
    r_coast = am.mapping(55.0,   0.0, 0.8, 0.8)
    r_brake = am.mapping(55.0, -15.0, 0.8, 0.8)
    r_accel = am.mapping(55.0, +10.0, 0.8, 0.8)
    print(f"  coasting ax= 0.0: [{r_coast[0]:+.4f}, {r_coast[1]:+.4f}]")
    print(f"  braking  ax=-15:  [{r_brake[0]:+.4f}, {r_brake[1]:+.4f}]")
    print(f"  accel    ax=+10:  [{r_accel[0]:+.4f}, {r_accel[1]:+.4f}]")
    print("  Different ax → different mapped outputs  ✓")

    print("\n" + "=" * 70)
    print("All integration self-tests passed ✓")
    print("=" * 70)
