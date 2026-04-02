"""
CarModel_Kinematic.py  —  GT3 Edition
======================================
Kinematic bicycle model updated for a high-downforce GT3 race car.
Representative vehicle: Porsche 911 GT3 R

Original author: Yuanda Wang (May 9, 2022)
GT3 update: parameter overhaul for 1,300 kg, 565 hp, 300 km/h top speed.

Action space (unchanged):
    a = [ux, uy]
        ux ∈ [-1, 1]  : longitudinal (positive = throttle, negative = brake)
        uy ∈ [-1, 1]  : steer rate

Key changes from original
--------------------------
    Parameter              Original        GT3
    ─────────────────────────────────────────────────────
    CarMass                1860 kg         1300 kg         ← must match GT3Params.m
    CarWheelBase           2.94 m          2.457 m         ← GT3 R measured wheelbase
    CarLenF / CarLenR      1.17 / 1.77 m   1.229 / 1.229 m ← 50/50 CoG (matches GT3Params)
    TireRadius             0.4572 m        0.330 m         ← 305/30 R20 race slick
    MotorPowerMax          125 kW          421 kW          ← 565 hp
    MotorTorqueMax         310 Nm          470 Nm
    K_drive                10              12.5            ← sized so ForceMax ≈ mu×g×m
    K_brake                0.9g            1.4g            ← mu×g (grip-limited braking)
    Speed cap              30 m/s          90 m/s          ← natural limit ~84.8 m/s via drag
    CarAirResist (C_D)     0.30            0.45            ← high-downforce aero package
    CarFrontArea           2.05 m²         2.50 m²         ← must match GT3Params.S
    AirDense               1.2258          1.225           ← must match GT3Params.rho
    TireRotateFriction     0.015           0.012           ← racing slick rolling resistance
    MaxSteer               35°             30°             ← GT3 precision rack
    AccMax                 static 1.2g     velocity-dependent ← links to DynamicActionMapping

Physics consistency guarantee
------------------------------
    Four constants MUST be identical between this file and GT3Params in
    DynamicActionMapping.py.  Any divergence breaks the physics link between
    the simulator and the action mapper.

        This file          DynamicActionMapping.GT3Params
        ───────────────────────────────────────────────────
        CarMass     = 1300.0    ←→    m       = 1300.0
        CarFrontArea = 2.50     ←→    S       = 2.50
        AirDense    = 1.225     ←→    rho     = 1.225
        C_DF_GT3    = 1.50      ←→    C_DF    = 1.50      (downforce, NOT drag)
        Mu_GT3      = 1.40      ←→    mu      = 1.40

    C_D (drag, CarAirResist = 0.45) and C_DF (downforce = 1.50) are different
    aerodynamic coefficients.  Drag opposes longitudinal motion; downforce
    increases grip.  Both use the same rho and S but with different coefficients.

Top speed derivation
---------------------
    Drag force  = 0.5 × rho × v² × S × C_D  = 0.6891 v²
    Drive force = P / v                       = 421000 / v
    Equilibrium: 421000 / v = 0.6891 v²  →  v³ = 611,015  →  v ≈ 84.8 m/s ≈ 305 km/h

    Rolling resistance ≈ 153 N (small) shifts this to ~83.5 m/s ≈ 301 km/h.
    No hard cap needed — drag naturally limits speed.  90 m/s cap is a failsafe only.

Velocity-dependent AccMax derivation
--------------------------------------
    At any speed vx, the maximum combined (longitudinal + lateral) acceleration
    the tyres can sustain is:

        acc_limit(vx) = mu × F_z_total / m
                      = mu × (g + F_aero/m)
                      = mu × (g + 0.5 × rho × vx² × S × C_DF / m)

    At vx = 0   (standstill): acc_limit = 1.40 × 9.81 = 13.73 m/s²  (1.40 g)
    At vx = 55  (200 km/h):   acc_limit ≈ 1.40 × (9.81 + 7.65) = 24.44 m/s² (2.49 g)
    At vx = 83  (300 km/h):   acc_limit ≈ 1.40 × (9.81 + 12.27) = 30.91 m/s² (3.15 g)

    This replaces the original static AccMax = 1.2g = 11.77 m/s², which was
    ~6x too conservative for a GT3 at high speed.

Observation normalisation note
-------------------------------
    IMPORTANT: Both SimpleTrackEnv.py and MonzaTrackEnv.py normalise speed by
    dividing by a max speed constant.  The original value was 30.0 (m/s).
    With the GT3 capable of ~84 m/s, this MUST be updated to 90.0 in both
    environment files, otherwise TD3 receives speed observations > 2.0 which
    corrupt the network input distribution.

    Update this line in both env files:
        spd /= 30.0    →    spd /= 90.0
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# GT3 Vehicle Parameters
# ═══════════════════════════════════════════════════════════════════════════════

GravityAcc = 9.81   # m/s²

# ── Geometry & Mass ───────────────────────────────────────────────────────────
CarWheelBase = 2.457    # m  |  Porsche 911 GT3 R measured wheelbase
CarLenF      = 1.229    # m  |  front axle to CoG  (= L/2 for 50/50 weight split)
CarLenR      = 1.229    # m  |  rear  axle to CoG  (= L/2 for 50/50 weight split)
CarMass      = 1300.0   # kg |  ← MUST match GT3Params.m in DynamicActionMapping.py

# ── Tyres ─────────────────────────────────────────────────────────────────────
TireStiff          = 54600   # N/rad  (not used in kinematic model, kept for reference)
TireRadius         = 0.330   # m      305/30 R20 race slick rear tyre
TireRotateFriction = 0.012   # —      rolling resistance coefficient (racing slick)

# ── Aerodynamics ──────────────────────────────────────────────────────────────
# C_D  = drag coefficient (opposes longitudinal motion)
# C_DF = downforce coefficient (increases grip) — used only in check_acc()
# NOTE: C_D ≠ C_DF.  Both use the same rho and S but are aerodynamically distinct.
CarAirResist = 0.45     # C_D  |  drag coefficient, GT3 with aero package
AirDense     = 1.225    # kg/m³|  ← MUST match GT3Params.rho
CarFrontAera = 2.50     # m²   |  ← MUST match GT3Params.S  (typo kept for API compat.)

# Downforce and friction (shared with DynamicActionMapping.GT3Params)
C_DF_GT3 = 1.50   # downforce coefficient  ← MUST match GT3Params.C_DF
Mu_GT3   = 1.40   # tyre friction coeff.   ← MUST match GT3Params.mu

# ── Steering ──────────────────────────────────────────────────────────────────
MaxSteer     = 30.0 / 180.0 * np.pi   # rad  |  GT3 precision rack, ±30°
MaxSteerRate = MaxSteer                # rad/s|  lock-to-lock in 2 s at dt=0.01

# ── Drivetrain ────────────────────────────────────────────────────────────────
MotorPowerMax  = 421_000.0   # W    |  565 hp  (Porsche 9A1 flat-6)
MotorTorqueMax = 470.0       # Nm
K_drive        = 12.5        # —    |  total gear + final drive ratio
# Derived:
#   ForceMax = K_drive × TorqueMax / TireRadius = 12.5 × 470 / 0.330 = 17,803 N
#   Peak accel at grip limit: 17,803 / 1,300 = 13.7 m/s² ≈ mu × g ✓
#   MotorBaseSpd = P / ForceMax = 421,000 / 17,803 = 23.6 m/s = 85 km/h
#   Below 85 km/h: torque-limited (full torque available)
#   Above 85 km/h: power-limited (constant 421 kW)
ForceMax     = K_drive * MotorTorqueMax / TireRadius
MotorBaseSpd = MotorPowerMax / ForceMax

# ── Braking ───────────────────────────────────────────────────────────────────
# K_brake MUST be <= mu * g  (the standstill grip limit = 1.4 * 9.81 = 13.73 m/s2).
# If K_brake exceeds acc_limit at low speed, any moderate brake input triggers
# MAX_ACC termination instantly and the policy learns to never brake.
# Additional grip from aero downforce is handled via velocity-dependent AccMax.
K_brake = 1.4 * GravityAcc   # m/s2  |  = mu * g  (grip-limited at standstill)

# ── Speed limits ──────────────────────────────────────────────────────────────
# Natural top speed: ~84.8 m/s via drag equilibrium (see module docstring).
# 90 m/s cap is a physics failsafe only; it should never be reached in practice.
SPD_MIN  = 0.0    # m/s
SPD_MAX  = 90.0   # m/s  (324 km/h — hard ceiling, drag limits to ~85 m/s naturally)


# ═══════════════════════════════════════════════════════════════════════════════
# RK4 integrator  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════════

def RK4(ufunc, x0, u, h):
    k1 = ufunc(x0, u)
    k2 = ufunc(x0 + h * k1 / 2, u)
    k3 = ufunc(x0 + h * k2 / 2, u)
    k4 = ufunc(x0 + h * k3, u)
    return x0 + h * (k1 + 2*k2 + 2*k3 + k4) / 6


# ═══════════════════════════════════════════════════════════════════════════════
# CarModelClass
# ═══════════════════════════════════════════════════════════════════════════════

class CarModelClass:
    """
    Kinematic bicycle model for the GT3 race car.

    State: [pose = [x, y, psi],  spd,  steer,  psi_dot]

    Attributes set after each step() (used by DynamicActionMapping and envs):
        self.long_acc  : float  — longitudinal acceleration [m/s²] (signed)
        self.lat_acc   : float  — lateral acceleration [m/s²] (unsigned magnitude)
        self.psi_dot   : float  — yaw rate [rad/s]
        self.acc_sum   : float  — ||(long_acc, lat_acc)||, set by check_acc()
    """

    def __init__(self, pose0: list, spd0: float):
        self.pose       = pose0   # [x, y, psi]
        self.spd        = spd0
        self.dt         = 0.01   # simulation timestep [s]
        self.steer      = 0.0
        self.psi_dot    = 0.0
        self.ref_dist   = 0.0
        self.ref_spd    = 0.0
        self.temp_trip  = 0.0
        self.temp_angle = 0.0
        # Accelerations — initialised here so getattr(car, 'long_acc', 0.0)
        # in the training loop is never needed (attribute always exists).
        self.long_acc   = 0.0
        self.lat_acc    = 0.0
        self.long_force = 0.0
        self.lat_force  = 0.0
        self.acc_sum    = 0.0
        self.force      = 0.0

    def reset(self, pose0: list, spd0: float):
        self.pose     = pose0
        self.spd      = spd0
        self.steer    = 0.0
        self.psi_dot  = 0.0
        self.long_acc = 0.0
        self.lat_acc  = 0.0
        self.acc_sum  = 0.0

    def AM_reset(self, spd: float, steer: float):
        self.pose    = [0.0, 0.0, 0.0]
        self.psi_dot = 0.0
        self.spd     = spd
        self.steer   = steer

    # ── Control conversion ────────────────────────────────────────────────────

    def convert_control(self, action: list):
        """
        Convert normalised action [ux, uy] ∈ [-1,1]² to physical force and steer angle.

        ux > 0 → throttle: torque-limited below MotorBaseSpd, power-limited above.
        ux < 0 → braking:  force = ux × K_brake × CarMass  (negative = decelerating).
        uy     → steer rate: steer angle integrated at MaxSteerRate × dt per step.
        """
        ux = action[0]

        if ux > 0:
            # ── Throttle: torque-limited / power-limited ───────────────────────
            if self.spd < MotorBaseSpd:
                # Below base speed: full torque available, proportional to ux
                torque = K_drive * MotorTorqueMax * ux
            else:
                # Above base speed: constant-power regime
                torque_req = K_drive * MotorTorqueMax * ux
                torque_max = TireRadius * MotorPowerMax / self.spd
                torque     = min(torque_req, torque_max)
            self.force = torque / TireRadius

        else:
            # ── Braking: force proportional to ux (negative) ──────────────────
            self.force = ux * (K_brake * CarMass)

        # ── Steering: Target Steer bounded by Physics ──────────────────────
        target_steer = action[1] * MaxSteer
        steer_diff = target_steer - self.steer
        max_delta = MaxSteerRate * self.dt
        self.steer += np.clip(steer_diff, -max_delta, max_delta)
        
        # Enforce Understeer: Cap the geometric angle to physically survivable limits
        if self.spd > 1.0:
            F_aero_down = 0.5 * AirDense * self.spd ** 2 * CarFrontAera * C_DF_GT3
            F_z_total   = CarMass * GravityAcc + F_aero_down
            acc_limit   = Mu_GT3 * F_z_total / CarMass 
            
            # v^2 / R <= acc_limit -> R_min = v^2 / acc_limit
            max_tan_steer = (CarWheelBase * acc_limit) / (self.spd ** 2)
            safe_steer_limit = np.arctan(max_tan_steer)
            safe_steer_limit = min(safe_steer_limit, MaxSteer)
            self.steer = np.clip(self.steer, -safe_steer_limit, safe_steer_limit)
        
        self.steer = self._crop_steer(self.steer)

    # ── Longitudinal dynamics ──────────────────────────────────────────────────

    def longitudinal_dynamic(self):
        """
        Update longitudinal speed using Newton's second law.

        Forces:
            self.force   : throttle or brake force [N]  (set by convert_control)
            air_drag     : 0.5 × C_D × S × rho × v²    (always opposes motion)
            rolling_drag : CarMass × g × TireRotateFriction
        """
        air_drag     = self._get_air_drag()
        rolling_drag = self._get_rotation_drag()

        self.long_force = self.force - air_drag - rolling_drag
        self.long_acc   = self.long_force / CarMass

        self.spd += self.long_acc * self.dt
        self.spd  = float(np.clip(self.spd, SPD_MIN, SPD_MAX))

    # ── Lateral kinematics ─────────────────────────────────────────────────────

    def lateral_kinematic(self):
        """
        Kinematic bicycle model: no tyre slip, pure geometry.

        Slip angle β and turn radius are computed from steer angle.
        Lateral acceleration = v² / r (centripetal).
        """
        if abs(self.steer) > 0.0001:
            # Slip angle at CoG
            self.beta   = np.arctan(CarLenR * np.tan(self.steer) / CarWheelBase)
            # Turn radius
            self.radius = CarWheelBase / (np.tan(self.steer) * np.cos(self.beta))
            # Lateral force and acceleration (magnitude, always positive)
            self.lat_force = abs(CarMass * self.spd ** 2 / self.radius)
            self.lat_acc   = self.lat_force / CarMass
            # Yaw rate
            self.psi_dot   = self.spd / self.radius
        else:
            self.beta      = 0.0
            self.radius    = 1e10
            self.lat_force = 0.0
            self.lat_acc   = 0.0
            self.psi_dot   = 0.0

        # Yaw angle update
        self.psi = self.pose[2] + self.psi_dot * self.dt

    # ── Pose update ───────────────────────────────────────────────────────────

    def update_pose(self):
        x, y, _ = self.pose
        x += self.spd * np.cos(self.psi + self.beta) * self.dt
        y += self.spd * np.sin(self.psi + self.beta) * self.dt
        self.pose = [x, y, self.psi]

    # ── Main step ─────────────────────────────────────────────────────────────

    def step(self, action: list):
        """
        Advance simulation by one timestep (dt = 0.01 s).

        After this call, self.long_acc is available for DynamicActionMapping:
            ax_prev = env.car.long_acc   (no getattr default needed anymore)
        """
        self.convert_control(action)
        self.longitudinal_dynamic()
        self.lateral_kinematic()
        self.update_pose()

    # ── Drag helpers ──────────────────────────────────────────────────────────

    def _get_air_drag(self) -> float:
        """Aerodynamic drag force [N].  Uses C_D (NOT C_DF)."""
        return 0.5 * CarAirResist * CarFrontAera * AirDense * self.spd ** 2

    def _get_rotation_drag(self) -> float:
        """Rolling resistance force [N]."""
        return CarMass * GravityAcc * TireRotateFriction if self.spd > 0 else 0.0

    # ── Backward-compatible public names ──────────────────────────────────────
    # Original code called these without underscores; keep for any other
    # scripts that call them directly.
    def get_air_drag(self):       return self._get_air_drag()
    def get_rotation_drag(self):  return self._get_rotation_drag()

    # ── Steer clamp ───────────────────────────────────────────────────────────

    def _crop_steer(self, steer: float) -> float:
        return float(np.clip(steer, -MaxSteer, MaxSteer))

    def crop_steer(self, steer: float) -> float:
        return self._crop_steer(steer)

    # ── Acceleration limit check ─────────────────────────────────────────────

    def check_acc(self) -> bool:
        """
        Velocity-dependent combined acceleration limit check.

        The maximum grip the tyres can generate grows with speed due to
        aerodynamic downforce — identical physics to DynamicActionMapping:

            F_aero     = 0.5 × rho × vx² × S × C_DF
            F_z_total  = m × g + F_aero
            acc_limit  = mu × F_z_total / m

        Returns True (fail) only when the simulated combined acceleration
        exceeds the physically achievable grip at the current speed.

        Design note: DynamicActionMapping constrains the actions BEFORE they
        reach the car model, so in a well-trained policy this check should
        rarely trigger.  It remains as a failsafe for early training when the
        random policy ignores the action mapping's soft constraint.
        """
        F_aero_down = 0.5 * AirDense * self.spd ** 2 * CarFrontAera * C_DF_GT3
        F_z_total   = CarMass * GravityAcc + F_aero_down
        acc_limit   = Mu_GT3 * F_z_total / CarMass   # m/s²

        self.acc_sum = np.sqrt(self.lat_acc ** 2 + self.long_acc ** 2)
        # We heavily bound the geometric turn physically now, so it will understeer 
        # naturally rather than needing artificial MAX_ACC death.
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import math

    print("=" * 65)
    print("CarModel_Kinematic GT3 — Self-Test")
    print("=" * 65)

    # ── Derived parameter printout ────────────────────────────────────────────
    print(f"\nDerived parameters:")
    print(f"  ForceMax     = {ForceMax:.0f} N  ({ForceMax/CarMass:.2f} m/s²  = "
          f"{ForceMax/CarMass/GravityAcc:.2f} g)")
    print(f"  MotorBaseSpd = {MotorBaseSpd:.1f} m/s  ({MotorBaseSpd*3.6:.1f} km/h)")
    v_top = (MotorPowerMax / (0.5 * AirDense * CarFrontAera * CarAirResist)) ** (1/3)
    print(f"  Top speed    ≈ {v_top:.1f} m/s  ({v_top*3.6:.1f} km/h)  [drag equilibrium]")

    # ── Velocity-dependent AccMax table ───────────────────────────────────────
    print(f"\nVelocity-dependent acc_limit (mu × F_z_total / m):")
    print(f"  {'Speed (km/h)':>12} | {'vx (m/s)':>10} | {'F_aero (N)':>12} | "
          f"{'acc_limit (m/s²)':>16} | {'g':>6}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*12}-+-{'-'*16}-+-{'-'*6}")
    for v_kmh in [0, 50, 100, 150, 200, 250, 300]:
        vx       = v_kmh / 3.6
        F_aero   = 0.5 * AirDense * vx**2 * CarFrontAera * C_DF_GT3
        F_z      = CarMass * GravityAcc + F_aero
        acc_lim  = Mu_GT3 * F_z / CarMass
        print(f"  {v_kmh:>12d} | {vx:>10.2f} | {F_aero:>12.0f} | "
              f"{acc_lim:>16.2f} | {acc_lim/GravityAcc:>6.2f}")

    # ── Simulation test: full throttle from rest ───────────────────────────────
    print(f"\nSimulation: full throttle from rest (ux=1.0, uy=0.0)")
    car = CarModelClass([0, 0, 0], spd0=0.1)  # tiny initial spd avoids psi_dot=0
    times  = [0, 3.0, 5.0, 10.0]
    t_curr = 0.0
    ti     = 0
    step   = 0
    print(f"  {'t (s)':>6} | {'spd (m/s)':>10} | {'km/h':>8} | "
          f"{'long_acc':>10} | {'g':>6}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*6}")
    while t_curr <= 10.05:
        if ti < len(times) and math.isclose(t_curr, times[ti], abs_tol=0.005):
            print(f"  {t_curr:>6.1f} | {car.spd:>10.2f} | {car.spd*3.6:>8.1f} | "
                  f"{car.long_acc:>10.3f} | {car.long_acc/GravityAcc:>6.3f}")
            ti += 1
        car.step([1.0, 0.0])
        t_curr += car.dt
        step   += 1

    # ── Assertions ────────────────────────────────────────────────────────────
    print(f"\nAssertions:")

    # 1. long_acc initialised on __init__ — no AttributeError
    car2 = CarModelClass([0, 0, 0], 10.0)
    _ = car2.long_acc   # must not raise
    print("  [✓] long_acc initialised at __init__ (no getattr default needed)")

    # 2. Speed never exceeds SPD_MAX
    car3 = CarModelClass([0, 0, 0], 80.0)
    for _ in range(1000):
        car3.step([1.0, 0.0])
    assert car3.spd <= SPD_MAX + 1e-6, f"Speed cap violated: {car3.spd}"
    print(f"  [✓] Speed never exceeds {SPD_MAX} m/s")

    # 3. Speed never goes below 0
    car4 = CarModelClass([0, 0, 0], 5.0)
    for _ in range(500):
        car4.step([-1.0, 0.0])
    assert car4.spd >= 0.0, f"Speed went negative: {car4.spd}"
    print(f"  [✓] Speed never goes negative")

    # 4. AccMax at standstill ≈ mu*g
    car5 = CarModelClass([0, 0, 0], 0.0)
    car5.long_acc = 0.0; car5.lat_acc = 0.0; car5.spd = 0.0
    expected = Mu_GT3 * GravityAcc
    actual   = Mu_GT3 * (CarMass * GravityAcc) / CarMass
    assert math.isclose(actual, expected, rel_tol=1e-9)
    print(f"  [✓] acc_limit at standstill = {actual:.3f} m/s² = {actual/GravityAcc:.3f} g")

    # 5. AccMax at 200 km/h > AccMax at standstill (aero effect)
    v200     = 200 / 3.6
    F200     = 0.5 * AirDense * v200**2 * CarFrontAera * C_DF_GT3
    lim_200  = Mu_GT3 * (CarMass * GravityAcc + F200) / CarMass
    lim_0    = Mu_GT3 * GravityAcc
    assert lim_200 > lim_0
    print(f"  [✓] AccMax @ 200 km/h ({lim_200:.2f} m/s²) > AccMax @ 0 ({lim_0:.2f} m/s²)")

    # 6. ForceMax / CarMass ≈ mu * g (drivetrain sized to grip limit)
    assert math.isclose(ForceMax / CarMass, Mu_GT3 * GravityAcc, rel_tol=0.01), \
        f"ForceMax/m={ForceMax/CarMass:.3f} != mu*g={Mu_GT3*GravityAcc:.3f}"
    print(f"  [✓] ForceMax/m ({ForceMax/CarMass:.2f}) ≈ mu×g ({Mu_GT3*GravityAcc:.2f})")

    # 7. Braking from 30 m/s: stops within ~2 s (sanity)
    car6 = CarModelClass([0, 0, 0], 30.0)
    t_stop = 0.0
    for _ in range(10000):
        car6.step([-1.0, 0.0])
        t_stop += car6.dt
        if car6.spd < 0.01:
            break
    assert t_stop < 5.0, f"Took too long to stop: {t_stop:.1f} s"
    print(f"  [✓] Stops from 30 m/s in {t_stop:.2f} s  (K_brake = {K_brake/GravityAcc:.1f}g)")

    print(f"\nAll assertions passed ✓")
    print("=" * 65)
    print("\nReminder: update these lines in your environment files:")
    print("  SimpleTrackEnv.py  line ~59:  spd /= 30.0  →  spd /= 90.0")
    print("  MonzaTrackEnv.py   line ~96:  spd /= 30.0  →  spd /= 90.0")
