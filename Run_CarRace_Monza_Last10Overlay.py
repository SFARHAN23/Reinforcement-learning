import glob
import io
import os
import re
from contextlib import redirect_stdout

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

import TD3
from DynamicActionMapping import DynamicActionMappingClass
from LogTools_Monza import MonzaCarStateLogClass


MODEL_PREFIX = 'monza_gt3_model_'
MODEL_COUNT = 10
MAX_STEPS = 100000
OUTPUT_PATH = 'results/Monza_Last10_SpeedMap_Overlay.png'


def process_state(state):
    return np.reshape(state, [1, -1])


def resolve_recent_models(prefix, count):
    candidates = []
    pattern = os.path.join('models', f'{prefix}*_actor')
    for actor_path in glob.glob(pattern):
        base_path = actor_path[:-len('_actor')]
        base_name = os.path.basename(base_path)
        match = re.fullmatch(rf'{re.escape(prefix)}(\d+)', base_name)
        if match:
            candidates.append((int(match.group(1)), base_path))

    if not candidates:
        raise FileNotFoundError(f'No matching model checkpoints found for prefix {prefix}')

    candidates.sort(key=lambda item: item[0])
    return candidates[-count:]


def load_policy_checkpoint(policy, model_path):
    try:
        policy.load_gpu2cpu(model_path)
        print(f'Model loaded (GPU->CPU) from {model_path}')
    except Exception:
        policy.load(model_path)
        print(f'Model loaded from {model_path}')


def detect_state_dim(model_path):
    import torch

    actor_file = model_path + '_actor'
    sd = torch.load(actor_file, map_location='cpu', weights_only=True)
    return sd['l1.weight'].shape[1]


def evaluate_model(model_path):
    import MonzaTrackEnv as _mte

    detected_dim = detect_state_dim(model_path)
    _mte.ENHANCED_OBS = detected_dim == 34
    _mte.STATE_DIM = detected_dim

    policy = TD3.TD3(state_dim=detected_dim, action_dim=2, max_action=1)
    load_policy_checkpoint(policy, model_path)

    am = DynamicActionMappingClass()
    with redirect_stdout(io.StringIO()):
        env = _mte.MonzaTrackEnvClass(am=am)
    logger = MonzaCarStateLogClass(env, am=am)

    episode_reward = 0.0
    ob = process_state(env.test_reset()[:detected_dim])
    pre_trip = 0.0
    lap1_done = False
    lap1_time = None

    for step in range(MAX_STEPS):
        action = policy.select_action(ob)
        action_in = am.mapping(
            env.car.spd,
            env.car.long_acc,
            action[0],
            action[1]
        )

        next_ob, reward, done = env.step(action_in)
        ob = process_state(next_ob[:detected_dim])
        episode_reward += reward

        if done:
            logger.log_data(step, lap1_done, list(action), list(action_in))
            break

        current_trip = env.track.car_trip

        if current_trip < pre_trip and not lap1_done:
            lap1_time = step
            lap1_done = True
        elif current_trip < pre_trip and lap1_done:
            lap2_time = step - lap1_time
            if lap2_time > 1000:
                logger.log_data(step, lap1_done, list(action), list(action_in))
                break

        pre_trip = current_trip
        logger.log_data(step, lap1_done, list(action), list(action_in))

    fail_reason = env.query_fail_reason()
    max_pct = env.track.car_trip / env.track.total_trip * 100.0 if not lap1_done else 100.0

    return {
        'model_path': model_path,
        'model_name': os.path.basename(model_path),
        'state_dim': detected_dim,
        'reward': episode_reward,
        'max_pct': max_pct,
        'fail_reason': fail_reason,
        'lap1_done': lap1_done,
        'lap1_time': lap1_time,
        'track': env.track,
        'laplog': logger.lap1_log,
    }


def add_speed_line(ax, posx, posy, spd_kmh, norm, alpha, linewidth):
    if len(posx) >= 2:
        points = np.column_stack([posx, posy]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        line = LineCollection(
            segments,
            cmap='turbo',
            norm=norm,
            linewidths=linewidth,
            alpha=alpha,
            zorder=5
        )
        line.set_array(spd_kmh[:-1])
        ax.add_collection(line)
        return line

    return ax.scatter(
        posx, posy,
        c=spd_kmh, cmap='turbo', norm=norm,
        s=20, alpha=alpha, zorder=5
    )


def plot_overlay(records, output_path):
    newest_track = records[-1]['track']
    fig = newest_track.show(
        fig_name='MonzaLast10Overlay',
        figsize=(14, 8),
        label_corners=False,
        show_start_marker=False,
        title=None,
        show_legend=False
    )
    ax = fig.axes[0]
    ax.plot([-15, 15], [0, 0], lw=3, c='grey', solid_capstyle='round', zorder=3)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=300)
    color_ref = None
    handles = []
    total = len(records)

    for idx, rec in enumerate(records):
        age = idx / max(total - 1, 1)
        alpha = 0.18 + 0.77 * age
        linewidth = 1.5 + 2.5 * age
        speed_kmh = np.array(rec['laplog'].spd) * 3.6
        color_ref = add_speed_line(
            ax, rec['laplog'].posx, rec['laplog'].posy,
            speed_kmh, norm, alpha, linewidth
        )
        if rec['laplog'].posx:
            ax.scatter(
                rec['laplog'].posx[-1], rec['laplog'].posy[-1],
                s=18 + 20 * age, c='white', edgecolors='black',
                linewidths=0.4, alpha=min(1.0, alpha + 0.1), zorder=6
            )

        short_name = rec['model_name'].replace('monza_gt3_model_', '#')
        handles.append(
            Line2D(
                [0], [0], color='black', lw=linewidth, alpha=alpha,
                label=f'{short_name}  {rec["max_pct"]:.1f}%  {rec["fail_reason"]}'
            )
        )

    fig.colorbar(color_ref, ax=ax, orientation='horizontal', shrink=0.55, pad=0.02,
                 label='Speed (km/h)')
    ax.legend(handles=handles, loc='upper right', fontsize=8,
              title='Older → Newer', framealpha=0.92)
    ax.set_title('Monza Last 10 Checkpoints — Speedmap Overlay', size=13)
    ax.axis('equal')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f'Overlay figure saved → {output_path}')
    plt.close(fig)


def main():
    os.makedirs('results', exist_ok=True)
    recent_models = resolve_recent_models(MODEL_PREFIX, MODEL_COUNT)
    print('Evaluating checkpoints:')
    for model_num, model_path in recent_models:
        print(f'  {model_num:>3d}  {model_path}')

    records = []
    for _, model_path in recent_models:
        record = evaluate_model(model_path)
        records.append(record)
        print(
            f"{record['model_name']}: max={record['max_pct']:.1f}%  "
            f"reason={record['fail_reason']}  state_dim={record['state_dim']}"
        )

    plot_overlay(records, OUTPUT_PATH)


if __name__ == '__main__':
    main()
