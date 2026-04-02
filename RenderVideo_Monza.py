"""
RenderVideo_Monza.py
====================
Animation renderer for the Monza GT3 evaluation run.

Updated from RenderVideo.py (Wang et al., 2024) for:

    1. HUD text positions recalculated for Monza coordinate space
       (track spans x:[-10,1560], y:[-10,1850]; info panel sits above
       the start/finish area in the upper-left corner)

    2. Speed display in km/h (was implicit m/s), with a 0–300 km/h bar

    3. Dynamic AccMax overlay in the HUD: shows the current velocity-
       dependent grip limit alongside the measured combined acceleration

    4. AM flag labelled "DAM" (Dynamic Action Mapping) for clarity

    5. Car rectangle scaled proportionally to Monza (track is ~8× larger
       than SimpleTrack so a 1.5× car was invisible; scale raised to 3×)

    6. Animation interval tuned: step-skip 5 at 30 fps gives
       5 × 0.01 s / frame × 30 fps ≈ real-time playback

Coordinate reference (from MonzaTrack geometry run):
    Start/finish : (0, 0)
    Track extents: x ∈ [-10, 1560]  y ∈ [-10, 1850]
    Info panel   : x ∈ [-10, 300]   y ∈ [1880, 1980]   (above track)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.animation as animation

from CarModel_Kinematic import (
    AirDense, CarFrontAera, C_DF_GT3, Mu_GT3, GravityAcc, CarMass
)


def _acc_limit(vx: float) -> float:
    """Dynamic combined acceleration limit [m/s²] at speed vx."""
    F_aero = 0.5 * AirDense * vx**2 * CarFrontAera * C_DF_GT3
    F_z    = CarMass * GravityAcc + F_aero
    return Mu_GT3 * F_z / CarMass


def show_animi(env, logger, Tmax: int, save: bool = True,
               path: str = 'results/Monza_Video.mp4'):
    """
    Render and optionally save the Monza race animation.

    Parameters
    ----------
    env    : MonzaTrackEnvClass
    logger : MonzaCarStateLogClass
    Tmax   : int   — total steps recorded (used to size the frame range)
    save   : bool  — if True, write to `path` via ffmpeg
    path   : str   — output file path

    Notes
    -----
    Requires ffmpeg on PATH.  Install with: conda install -c conda-forge ffmpeg
    """

    # ── Merge lap data ────────────────────────────────────────────────────────
    l1, l2   = logger.lap1_log, logger.lap2_log
    posx_list  = l1.posx  + l2.posx
    posy_list  = l1.posy  + l2.posy
    psi_list   = l1.psi   + l2.psi
    spd_list   = l1.spd   + l2.spd
    steer_list   = l1.steer + l2.steer
    acc_list     = l1.acc   + l2.acc
    acc_lim_list = l1.acc_limit + l2.acc_limit
    ux_list      = l1.ux    + l2.ux
    T_list       = l1.T     + l2.T
    am_list      = l1.amflag+ l2.amflag
    trip_list    = l1.trip  + l2.trip

    if not posx_list:
        print('No logged frames available; skipping video rendering.')
        return

    # ── Build track figure ────────────────────────────────────────────────────
    trackfig = env.track.show(
        fig_name='monza_video',
        figsize=(14, 8),
        label_corners=False,
        show_start_marker=False,
        title=None,
        show_legend=False
    )
    ax = trackfig.axes[0]
    ax.axis('equal')
    ax.plot([-15, 15], [0, 0], lw=3, c='grey', solid_capstyle='round', zorder=2)

    # ── Car rectangle (3× real size for visibility on large track) ────────────
    amp        = 3.0
    car_width  = 1.95 * amp   # GT3 R actual width ~1.95 m
    car_length = 4.56 * amp   # GT3 R actual length ~4.56 m

    def build_rect_patch(posx, posy, psi, color='r'):
        rect = Rectangle(xy=(posx, posy), height=car_width,
                         width=car_length, color=color, zorder=5)
        vx1 = (car_length/2)*np.cos(psi) - (car_width/2)*np.sin(psi)
        vy1 = (car_length/2)*np.sin(psi) + (car_width/2)*np.cos(psi)
        rect.set_angle(psi / np.pi * 180)
        rect.set_xy((posx - vx1, posy - vy1))
        return rect

    def update_rect_patch(rect, posx, posy, psi):
        vx1 = (car_length/2)*np.cos(psi) - (car_width/2)*np.sin(psi)
        vy1 = (car_length/2)*np.sin(psi) + (car_width/2)*np.cos(psi)
        rect.set_angle(psi / np.pi * 180)
        rect.set_xy((posx - vx1, posy - vy1))

    car_patch = build_rect_patch(
        posx_list[0], posy_list[0], psi_list[0], color='r'
    )
    ax.add_patch(car_patch)

    panel_x, panel_y = -12, 1868
    panel_w, panel_h = 665, 300
    hud_patch = FancyBboxPatch(
        (panel_x, panel_y), panel_w, panel_h,
        boxstyle='round,pad=0.8,rounding_size=8',
        linewidth=1.0, edgecolor='#666666',
        facecolor='white', alpha=0.86, zorder=1
    )
    ax.add_patch(hud_patch)

    T_COL1, T_COL2 = 8, 368
    T_ROW = [2050, 2000, 1950, 1900]

    text_time   = ax.text(T_COL1, T_ROW[0], ' ', fontsize=9)
    text_lappct = ax.text(T_COL2, T_ROW[0], ' ', fontsize=9)
    text_spd    = ax.text(T_COL1, T_ROW[1], ' ', fontsize=9)
    text_trip   = ax.text(T_COL2, T_ROW[1], ' ', fontsize=9)
    text_steer  = ax.text(T_COL1, T_ROW[2], ' ', fontsize=9)
    text_acc    = ax.text(T_COL2, T_ROW[2], ' ', fontsize=9)
    text_lim    = ax.text(T_COL1, T_ROW[3], ' ', fontsize=9)
    text_action = ax.text(T_COL2, T_ROW[3], ' ', fontsize=9)
    text_dam = ax.text(652, T_ROW[0], ' ', fontsize=11, fontweight='bold')

    BAR_X = 652
    BAR_YC = 1946
    BAR_HALF = 48
    ax.plot([BAR_X, BAR_X], [BAR_YC - BAR_HALF, BAR_YC + BAR_HALF],
            c='silver', lw=3, zorder=2)
    ax.plot([BAR_X - 7, BAR_X + 7], [BAR_YC, BAR_YC], c='silver', lw=2, zorder=2)
    ax.text(BAR_X - 24, BAR_YC + BAR_HALF + 4, 'Thr', fontsize=7, color='grey')
    ax.text(BAR_X - 24, BAR_YC - BAR_HALF - 10, 'Brk', fontsize=7, color='grey')
    gasline, = ax.plot([BAR_X, BAR_X], [BAR_YC, BAR_YC], 'g-', linewidth=8, solid_capstyle='round')
    brkline, = ax.plot([BAR_X, BAR_X], [BAR_YC, BAR_YC], 'r-', linewidth=8, solid_capstyle='round')

    # ── Update function called per frame ──────────────────────────────────────
    def update_scene(i):
        vx     = spd_list[i]
        vx_kmh = vx * 3.6
        steer_deg   = steer_list[i] / np.pi * 180
        acc_g = (acc_list[i] / 9.81) if i < len(acc_list) and acc_list[i] is not None else 0.0
        acc_lim_g = (
            acc_lim_list[i] / 9.81
            if i < len(acc_lim_list) and acc_lim_list[i] is not None
            else _acc_limit(vx) / 9.81
        )

        # Car rectangle
        update_rect_patch(car_patch, posx_list[i], posy_list[i], psi_list[i])

        # HUD text
        text_time.set_text(f'Time: {T_list[i]:6.1f} s')
        text_lappct.set_text(f'Lap: {trip_list[i]/env.track.total_trip*100:5.1f}%')
        text_spd.set_text(f'Speed: {vx_kmh:5.0f} km/h')
        text_trip.set_text(f'Trip: {trip_list[i]:6.0f} m')
        text_steer.set_text(f'Steer: {steer_deg:+5.1f}°')
        text_acc.set_text(f'Acc: {acc_g:4.2f} g')
        text_lim.set_text(f'Grip Limit: {acc_lim_g:4.2f} g')
        text_action.set_text(f'Long Cmd: {ux_list[i]:+4.2f}')

        # DAM (Dynamic Action Mapping) indicator
        if am_list[i]:
            text_dam.set_text('DAM')
            text_dam.set_color('red')
        else:
            text_dam.set_text('   ')

        # Vertical throttle/brake bar
        if ux_list[i] >= 0:
            brkline.set_data([BAR_X, BAR_X], [BAR_YC, BAR_YC])
            gasline.set_data(
                [BAR_X, BAR_X],
                [BAR_YC, BAR_YC + ux_list[i] * BAR_HALF]
            )
        else:
            gasline.set_data([BAR_X, BAR_X], [BAR_YC, BAR_YC])
            brkline.set_data(
                [BAR_X, BAR_X],
                [BAR_YC, BAR_YC + ux_list[i] * BAR_HALF]
            )

        return ([car_patch, text_time, text_lappct, text_spd, text_trip,
                 text_steer, text_acc, text_lim, text_action, text_dam,
                 gasline, brkline])

    # ── Build animation ───────────────────────────────────────────────────────
    # Step-skip = 5 at dt=0.01 → 50 ms/frame → 20 fps native.
    # Saving at fps=30 plays back at 1.5× real time (nice for race reviews).
    frame_count = min(Tmax + 1, len(posx_list))
    frame_indices = np.arange(0, frame_count, 5)
    if frame_indices.size == 0:
        frame_indices = np.array([0])
    ani = animation.FuncAnimation(
        trackfig, update_scene, frame_indices,
        interval=20, blit=True
    )

    if save:
        ani.save(path, writer='ffmpeg', fps=30)
        print(f'Video saved → {path}')

    plt.show()


def smooth(data, window: int = 20) -> np.ndarray:
    """Rolling-window smoother (kept here for standalone use in RenderVideo)."""
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
