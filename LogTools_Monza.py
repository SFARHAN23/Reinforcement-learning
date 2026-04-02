"""
LogTools_Monza.py
==================
Data logger and plot tools for the Monza GT3 evaluation run.

Updated from LogTools.py (Wang et al., 2024) for:
    - Speed range 0–320 km/h  (was 20–110 km/h, SimpleTrack passenger car)
    - Acceleration range 0–3.5g  (was 0–1.3g; GT3 dynamic limit peaks at 3.15g)
    - Steer angle range ±35°  (was ±10°; GT3 MaxSteer = 30°)
    - Track distance x-axis from env.track.total_trip  (was hardcoded 860 m)
    - Speed colormap on trajectory: jet 0→300 km/h  (was 20→110 km/h)
    - All output filenames prefixed 'Monza_'
    - Grip utilisation panel added: shows dynamic acc_limit(vx) alongside acc_sum

Public API (identical to original CarStateLogClass):
    logger = MonzaCarStateLogClass(env)
    logger.log_data(step, lap1done, action, action_in)
    logger.show_trajectory(lap='lap1')
    logger.show_states_controls(lap='lap1')
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# ── Smoothing helper (unchanged) ──────────────────────────────────────────────

def smooth(data, window: int = 20) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        return arr.copy()

    out = np.empty(arr.size, dtype=float)
    for i in range(arr.size):
        start = max(0, i - window + 1)
        segment = arr[start:i + 1]
        if np.isnan(segment).all():
            out[i] = np.nan
        else:
            out[i] = np.nanmean(segment)
    return out


# ── Per-lap data container (unchanged) ───────────────────────────────────────

class LaplogClass:
    def __init__(self):
        self.posx   = []
        self.posy   = []
        self.psi    = []
        self.spd    = []
        self.steer  = []
        self.ax     = []
        self.ay     = []
        self.ux     = []
        self.uy     = []
        self.T      = []
        self.acc    = []
        self.acc_limit = []   # NEW: velocity-dependent grip limit [m/s²]
        self.trip   = []
        self.amflag = []


# ── Main logger class ─────────────────────────────────────────────────────────

class MonzaCarStateLogClass:
    """
    Logs car state and control data across up to two laps.

    Parameters
    ----------
    env : MonzaTrackEnvClass
        The active environment instance.  Used for track geometry in plots.
    am : DynamicActionMappingClass, optional
        If provided, logs the dynamic acc_limit each step for the grip
        utilisation panel.  Pass None to skip that panel.
    """

    def __init__(self, env, am=None):
        self.env        = env
        self.am         = am          # optional: DynamicActionMappingClass
        self.lap_flag   = []
        self.lap_index  = 1
        self.lap1_log   = LaplogClass()
        self.lap2_log   = LaplogClass()
        self.lap1_time  = None
        self.lap2_time  = None

    def log_data(self, step: int, lap1done: bool,
                 a_v: list, a_r: list):
        """
        Record one timestep of data.

        Parameters
        ----------
        step    : int   — simulation step counter
        lap1done: bool  — True once the first lap has been completed
        a_v     : list  — raw policy action [ux, uy] before mapping
        a_r     : list  — mapped action [ux_mapped, uy_mapped] after mapping
        """
        env = self.env

        # Lap boundary switch
        if lap1done and (not self.lap_flag or not self.lap_flag[-1]):
            self.lap_index = 2
            self.lap1_time = len(self.lap_flag)
        self.lap_flag.append(lap1done)

        log = self.lap1_log if self.lap_index == 1 else self.lap2_log

        log.T.append(step * 0.01)
        log.posx.append(env.car.pose[0])
        log.posy.append(env.car.pose[1])
        log.psi.append(env.car.pose[2])
        log.spd.append(env.car.spd)
        log.steer.append(env.car.steer)
        log.trip.append(env.track.car_trip)
        log.acc.append(env.car.acc_sum)

        log.ax.append(a_v[0])
        log.ay.append(a_v[1])
        log.ux.append(a_r[0])
        log.uy.append(a_r[1])

        # AM active flag: action was modified
        log.amflag.append(a_v[0] != a_r[0] or a_v[1] != a_r[1])

        # Dynamic grip limit for this step (if am is available)
        if self.am is not None and self.am.last_result is not None:
            # Recompute acc_limit from current speed (same formula as check_acc)
            from CarModel_Kinematic import AirDense, CarFrontAera, C_DF_GT3, Mu_GT3, GravityAcc
            from CarModel_Kinematic import CarMass
            vx      = env.car.spd
            F_aero  = 0.5 * AirDense * vx**2 * CarFrontAera * C_DF_GT3
            F_z     = CarMass * GravityAcc + F_aero
            log.acc_limit.append(Mu_GT3 * F_z / CarMass)
        else:
            log.acc_limit.append(None)

    # ── Trajectory plot ────────────────────────────────────────────────────────

    def show_trajectory(self, lap: str = 'lap1'):
        """
        Overlay car trajectory on the Monza track, coloured by speed in km/h.
        """
        laplog, lap_time = self._get_lap(lap)

        fig1 = self.env.track.show(
            fig_name='MonzaTrajectory',
            figsize=(14, 8),
            label_corners=False,
            show_start_marker=False,
            title=None,
            show_legend=False
        )
        ax = fig1.axes[0]

        # Speed colourmap: 0–300 km/h
        norm = matplotlib.colors.Normalize(vmin=0, vmax=300)
        speed_kmh = np.array(laplog.spd) * 3.6
        if len(laplog.posx) >= 2:
            points = np.column_stack([laplog.posx, laplog.posy]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            sc = LineCollection(
                segments, cmap='turbo', norm=norm,
                linewidths=4.0, zorder=5
            )
            sc.set_array(speed_kmh[:-1])
            ax.add_collection(sc)
        else:
            sc = ax.scatter(
                laplog.posx, laplog.posy,
                c=speed_kmh, cmap='turbo', s=12, norm=norm, zorder=5
            )
        fig1.colorbar(sc, ax=ax, orientation='horizontal', shrink=0.55, pad=0.02,
                      label='Speed (km/h)')

        # Start/finish line marker
        ax.plot([-15, 15], [0, 0], lw=3, c='grey', solid_capstyle='round', zorder=4)

        ax.axis('equal')
        ax.axis('off')

        if lap_time is not None:
            lap_time_str = f"{lap_time / 100:.2f} s"
        else:
            lap_time_str = 'DNF'
        ax.set_title(f'Monza Trajectory — {lap}  (Lap Time = {lap_time_str})', size=12)
        fig1.tight_layout()

        path = f'results/Monza_Trajectory_{lap}.png'
        fig1.savefig(path, dpi=300)
        print(f'Trajectory figure saved → {path}')
        plt.close(fig1)

    # ── States and controls plot ───────────────────────────────────────────────

    def show_states_controls(self, lap: str = 'lap1'):
        """
        Four-panel plot: Speed / Throttle-Brake / Steer / Acceleration (g).
        If dynamic acc_limit data is available, overlays it on the acc panel.
        """
        laplog, _ = self._get_lap(lap)
        total_trip = self.env.track.total_trip

        c_blue  = '#006699'
        c_red   = '#CC3300'
        c_grey  = '#AAAAAA'

        has_lim = any(v is not None for v in laplog.acc_limit)
        n_panels = 5 if has_lim else 4
        fig, axes = plt.subplots(n_panels, 1, sharex=True,
                                 figsize=(12, 3 * n_panels))
        ax1, ax2, ax3, ax4 = axes[0], axes[1], axes[2], axes[3]

        # ── Panel 1: Speed ────────────────────────────────────────────────────
        ax1.plot(laplog.trip, np.array(laplog.spd) * 3.6, c=c_blue, lw=1.2)
        ax1.set_ylim([0, 340])
        ax1.set_yticks(range(0, 340, 50))
        ax1.set_ylabel('Speed (km/h)')
        ax1.grid(axis='y', ls=':', alpha=0.5)
        ax1.set_title(f'Monza — States & Controls — {lap}', size=12)

        # ── Panel 2: Throttle / Brake ─────────────────────────────────────────
        ux_arr = np.array(laplog.ux)
        ax2.plot(laplog.trip, smooth(ux_arr), c=c_blue, lw=1.2)
        ax2.axhline(0, color='k', lw=0.5)
        ax2.set_ylim([-1.25, 1.25])
        ax2.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax2.set_ylabel('Throttle / Brake')
        ax2.grid(axis='y', ls=':', alpha=0.5)

        # ── Panel 3: Steer angle ──────────────────────────────────────────────
        steer_deg = np.array(laplog.steer) / np.pi * 180
        ax3.plot(laplog.trip, steer_deg, c=c_blue, lw=1.0)
        ax3.axhline(0, color='k', lw=0.5)
        ax3.set_ylim([-35, 35])
        ax3.set_yticks([-30, -15, 0, 15, 30])
        ax3.set_ylabel('Steer Angle (°)')
        ax3.grid(axis='y', ls=':', alpha=0.5)

        # ── Panel 4: Combined acceleration (g) ────────────────────────────────
        acc_g = smooth(np.array(laplog.acc)) / 9.81
        ax4.plot(laplog.trip, acc_g, c=c_blue, lw=1.2, label='acc_sum')
        if has_lim:
            lim_g = [v / 9.81 if v is not None else np.nan
                     for v in laplog.acc_limit]
            ax4.plot(laplog.trip, smooth(lim_g), c=c_red, lw=1.0,
                     ls='--', label='dyn. limit')
            ax4.legend(fontsize=8, loc='upper right')
        ax4.set_ylim([0, 3.5])
        ax4.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.15])
        ax4.set_ylabel('Acceleration (g)')
        ax4.grid(axis='y', ls=':', alpha=0.5)

        # ── Panel 5 (optional): Grip utilisation % ────────────────────────────
        if has_lim and n_panels == 5:
            ax5 = axes[4]
            acc_arr = np.array(laplog.acc)
            lim_arr = np.array([v if v is not None else np.nan
                                 for v in laplog.acc_limit])
            util_pct = np.clip(acc_arr / lim_arr * 100, 0, 105)
            ax5.plot(laplog.trip, smooth(util_pct), c=c_red, lw=1.2)
            ax5.axhline(100, color='k', lw=0.7, ls='--', alpha=0.5)
            ax5.set_ylim([0, 110])
            ax5.set_yticks([0, 25, 50, 75, 100])
            ax5.set_ylabel('Grip util. (%)')
            ax5.set_xlabel('Track Distance (m)')
            ax5.set_xlim([0, total_trip])
            ax5.grid(axis='y', ls=':', alpha=0.5)
        else:
            ax4.set_xlabel('Track Distance (m)')
            ax4.set_xlim([0, total_trip])

        plt.tight_layout()
        path = f'results/Monza_States_{lap}.png'
        fig.savefig(path, dpi=300)
        print(f'States figure saved → {path}')
        plt.cla()
        plt.close()

    # ── Internal helper ───────────────────────────────────────────────────────

    def _get_lap(self, lap: str):
        if lap == 'lap1':
            return self.lap1_log, self.lap1_time
        elif lap == 'lap2':
            if self.lap_index == 1:
                raise RuntimeError('No lap2 log data — car did not complete lap 1')
            lap_time = len(self.lap_flag) - self.lap1_time
            return self.lap2_log, lap_time
        else:
            raise ValueError("lap must be 'lap1' or 'lap2'")
