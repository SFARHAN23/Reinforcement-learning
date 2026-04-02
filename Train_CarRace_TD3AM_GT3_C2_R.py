"""
Train_CarRace_TD3AM_GT3.py  —  Monza with corner spawn fix
"""

import os
import glob
import numpy as np
import TD3
import utils
from DynamicActionMapping import DynamicActionMappingClass
from MonzaTrackEnv import MonzaTrackEnvClass
from CarModel_Kinematic import CarModelClass

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
RESUME_FROM = None  # or None for fresh start
SPAWN_BLOCK = 50
LOG_PREFIX  = 'monza_gt3'

# T1 corner spawn: 33m before T1 entry (903m), v_max at r=15m = 14.4 m/s
# 8 m/s = 55% of corner limit → policy has realistic chance of surviving
CORNER_TRIP = 870.0
CORNER_SPD  = 8.0
# ═══════════════════════════════════════════════════════════════════════════


def process_state(s):
    return np.reshape(s, [1, -1])


state_dim  = 34   # ENHANCED_OBS=True: 31 base + 3 extended (v_ref, long_acc, grip_util)
action_dim = 2
max_action = 1

args = {
    'start_timesteps': 1e4,
    'expl_noise':      0.1,
    'batch_size':      256,
    'discount':        0.99,
    'tau':             0.005,
    'policy_noise':    0.2,
    'noise_clip':      0.5,
    'policy_freq':     2,
}

kwargs = {
    "state_dim":    state_dim,
    "action_dim":   action_dim,
    "max_action":   max_action,
    "discount":     args['discount'],
    "tau":          args['tau'],
}
kwargs["policy_noise"] = args['policy_noise'] * max_action
kwargs["noise_clip"]   = args['noise_clip']   * max_action
kwargs["policy_freq"]  = args['policy_freq']

os.makedirs('models',  exist_ok=True)
os.makedirs('results', exist_ok=True)

policy        = TD3.TD3(**kwargs)
replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=int(5e6))
am            = DynamicActionMappingClass()
env           = MonzaTrackEnvClass(am=am)

# ── Resume logic ───────────────────────────────────────────────────────────
trainlog = []
resuming = False

if RESUME_FROM is not None:
    actor_file = RESUME_FROM + '_actor'
    if not os.path.exists(actor_file):
        raise FileNotFoundError(
            f"Cannot resume: '{actor_file}' not found.\n"
            f"Available: {sorted(glob.glob('models/monza_gt3_model_*_actor'))}"
        )
    try:
        policy.load(RESUME_FROM)
        print(f"  ✓ Loaded actor weights from '{RESUME_FROM}'")
    except Exception:
        policy.load_gpu2cpu(RESUME_FROM)
        print(f"  ✓ Loaded actor weights (GPU→CPU) from '{RESUME_FROM}'")

    existing = glob.glob(f'models/{LOG_PREFIX}_model_*_actor')
    nums = []
    for f in existing:
        try:
            nums.append(int(f.split('_model_')[1].replace('_actor', '')))
        except ValueError:
            pass
    savecounter = max(nums) + 1 if nums else 1
    print(f"  ✓ Next checkpoint → model_{savecounter}")

    log_path = f'results/trainlog_{LOG_PREFIX}.npy'
    if os.path.exists(log_path):
        old = np.load(log_path, allow_pickle=True).tolist()
        trainlog = [list(row) for row in old]
        print(f"  ✓ Loaded trainlog ({len(trainlog)} episodes)")
    resuming = True
else:
    savecounter = 1

print(f"\n{'='*62}")
print(f"  Training : TD3-AM GT3  |  Track : MONZA ({env.track.total_trip:.0f} m)")
print(f"  Resume   : {RESUME_FROM or 'No — starting from scratch'}")
print(f"  Spawn    : FIXED→CORNER→RAND rotation, {SPAWN_BLOCK} eps each")
print(f"  T1 spawn : trip={CORNER_TRIP}m, spd={CORNER_SPD} m/s  (v_max=14.4 m/s at r=15m)")
print(f"  Next save: model_{savecounter}")
print(f"{'='*62}\n")

stepcounter  = 0
traincounter = 0
WINDOW         = 20
reward_window  = []
lap_pct_window = []
SPAWN_LABELS   = ['FIXED ', 'CORNER', 'RAND  ']

for episode in range(100_000):

    spawn_mode  = (episode // SPAWN_BLOCK) % 3
    spawn_label = SPAWN_LABELS[spawn_mode]

    if spawn_mode == 0:
        ob = env.test_reset()

    elif spawn_mode == 1:
        pose     = env.track.custom_car_pose(CORNER_TRIP, 0)
        env.car  = CarModelClass(pose, CORNER_SPD)
        env.reset_flags()
        ob = env.observe()
        env._last_ob = ob

    else:
        ob = env.reset()

    ob   = process_state(ob)
    done = False
    am.reset_diagnostics()
    episode_reward    = 0
    episode_timesteps = 0
    saved             = False
    max_lap_pct_ep    = 0.0

    for step in range(10000):
        stepcounter += 1

        if stepcounter < args['start_timesteps']:
            if resuming:
                noise  = np.random.normal(0, max_action * 0.3, size=action_dim)
                action = (policy.select_action(ob) + noise).clip(-max_action, max_action)
            else:
                action = np.random.uniform(-1, 1, action_dim)
        else:
            noise  = np.random.normal(0, max_action * args['expl_noise'], size=action_dim)
            action = (policy.select_action(ob) + noise).clip(-max_action, max_action)

        ax_prev   = env.car.long_acc
        action_in = am.mapping(env.car.spd, ax_prev, action[0], action[1])

        next_ob, r, done = env.step(action_in)
        next_ob = process_state(next_ob)

        if env.lap_pct > max_lap_pct_ep:
            max_lap_pct_ep = env.lap_pct

        replay_buffer.add(ob, action, next_ob, r, done)
        ob              = next_ob
        episode_reward += r
        episode_timesteps += 1

        if done:
            break

        if stepcounter > args['start_timesteps']:
            policy.train(replay_buffer, args['batch_size'])
            traincounter += 1

        if traincounter % 100000 == 0 and not saved:
            name = f'models/{LOG_PREFIX}_model_{savecounter}'
            policy.save(name)
            savecounter += 1
            saved = True
            print(f'  ✓ Saved → {name}')

    fail_reason    = env.query_fail_reason()
    max_lap_pct_ep = env.max_lap_pct

    reward_window.append(episode_reward)
    lap_pct_window.append(max_lap_pct_ep)
    if len(reward_window)  > WINDOW: reward_window.pop(0)
    if len(lap_pct_window) > WINDOW: lap_pct_window.pop(0)
    avg_reward  = float(np.mean(reward_window))
    avg_lap_pct = float(np.mean(lap_pct_window))

    phase = '[FILL ]' if stepcounter <= args['start_timesteps'] else '[TRAIN]'

    print(
        f"ep:{episode:<5d} "
        f"rew:{episode_reward:>8.1f}  "
        f"steps:{step:<5d}  "
        f"ctr:{traincounter:<7d}  "
        f"reason:{fail_reason:<12s} "
        f"{phase} [{spawn_label}]"
    )
    print(
        f"        lap%:{max_lap_pct_ep:>5.1f}  "
        f"avg_lap%:{avg_lap_pct:>5.1f}  "
        f"clip:{am.clip_rate:.3f}  "
        f"avg_rew:{avg_reward:>8.1f}"
    )

    trainlog.append([
        episode, episode_reward, step, traincounter,
        max_lap_pct_ep, avg_lap_pct, am.clip_rate, avg_reward,
    ])

    if saved:
        np.save(f'results/trainlog_{LOG_PREFIX}.npy', np.array(trainlog))
        print(f'  ✓ Log → results/trainlog_{LOG_PREFIX}.npy')
