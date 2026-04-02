"""
Train_CarRace_TD3AM_GT3.py
===========================
GT3 Dynamic Action Mapping — TD3-AM training script.

Supports both circuits via TRACK_SELECT at the top of the file.
Enhanced episode output includes lap completion %, peak lap %, and clip rate.

Changes from original Train_CarRace_TD3AM.py
---------------------------------------------
    #GT3   — imports / uses DynamicActionMappingClass
    #MONZA  — imports / uses MonzaTrackEnvClass  (when TRACK_SELECT='monza')
    #LOG    — enhanced logging fields
    #LAP    — lap progress tracking
"""

import numpy as np
import os
import TD3
import utils
from DynamicActionMapping import DynamicActionMappingClass           # GT3

# ── Track selector ─────────────────────────────────────────────────────────────
# Set to 'simple' for the original 5-unit track (fast iteration).
# Set to 'monza'  for the 11-unit Monza GP circuit (full challenge).
TRACK_SELECT = 'simple'   # 'simple' | 'monza'

if TRACK_SELECT == 'monza':
    from MonzaTrackEnv import MonzaTrackEnvClass as EnvClass         # MONZA
    LOG_PREFIX = 'monza_gt3'
else:
    from SimpleTrackEnv import SimpleTrackEnvClass as EnvClass
    LOG_PREFIX = 'simple_gt3'
# ──────────────────────────────────────────────────────────────────────────────


def process_state(s):
    return np.reshape(s, [1, -1])


# ── Hyperparameters (unchanged from original) ──────────────────────────────────
state_dim  = 34   # ENHANCED_OBS=True: 31 base + 3 extended (v_ref, long_acc, grip_util)
action_dim = 2
max_action = 1
dt         = 0.01

args = {
    'start_timesteps': 1e4,
    'eval_freq':       5e3,
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

# ── Policy, buffer, action mapping, environment ────────────────────────────────
policy        = TD3.TD3(**kwargs)
replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=int(5e6))
am            = DynamicActionMappingClass()                         # GT3
env           = EnvClass(am=am)

print(f"\n{'='*60}")
print(f"  Training : TD3-AM  |  Track : {TRACK_SELECT.upper()}")
print(f"  Circuit  : {env.track.total_trip:.0f} m total")
print(f"  Action   : GT3 Dynamic Action Mapping")
print(f"{'='*60}\n")

# ── Training counters ──────────────────────────────────────────────────────────
stepcounter  = 0
traincounter = 0
savecounter  = 1
trainlog     = []

WINDOW         = 20   # rolling average window size
reward_window  = []
lap_pct_window = []

for episode in range(10000):

    ob   = env.reset()
    ob   = process_state(ob)
    done = False
    am.reset_diagnostics()                                          # GT3 / LOG

    episode_reward    = 0
    episode_timesteps = 0
    saved             = False
    max_lap_pct_ep    = 0.0                                        # LAP

    for step in range(10000):
        stepcounter += 1

        # ── Action selection ───────────────────────────────────────────────────
        if stepcounter < args['start_timesteps']:
            action = np.random.uniform(-1, 1, action_dim)
        else:
            noise  = np.random.normal(0, max_action * args['expl_noise'],
                                      size=action_dim)
            action = (policy.select_action(ob) + noise).clip(-max_action, max_action)

        # ── GT3 Dynamic Action Mapping ─────────────────────────────────────────
        ax_prev   = getattr(env.car, 'long_acc', 0.0)              # GT3
        action_in = am.mapping(                                     # GT3
            env.car.spd,
            ax_prev,
            action[0],
            action[1],
        )

        # ── Environment step ───────────────────────────────────────────────────
        next_ob, r, done = env.step(action_in)
        next_ob = process_state(next_ob)

        # ── Lap progress ───────────────────────────────────────────────────────
        if hasattr(env, 'lap_pct') and env.lap_pct > max_lap_pct_ep:  # LAP
            max_lap_pct_ep = env.lap_pct                           # LAP

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

    # ── End-of-episode ─────────────────────────────────────────────────────────
    fail_reason = env.query_fail_reason()

    if hasattr(env, 'max_lap_pct'):                                # LAP
        max_lap_pct_ep = env.max_lap_pct                          # LAP

    reward_window.append(episode_reward)
    lap_pct_window.append(max_lap_pct_ep)
    if len(reward_window)  > WINDOW: reward_window.pop(0)
    if len(lap_pct_window) > WINDOW: lap_pct_window.pop(0)
    avg_reward  = float(np.mean(reward_window))
    avg_lap_pct = float(np.mean(lap_pct_window))

    # ── Enhanced print ─────────────────────────────────────────────────────────
    print(
        f"ep:{episode:<5d} "
        f"rew:{episode_reward:>8.1f}  "
        f"steps:{step:<5d}  "
        f"ctr:{traincounter:<7d}  "
        f"reason:{fail_reason:<12s}"
    )
    print(
        f"        lap%:{max_lap_pct_ep:>5.1f}  "
        f"avg_lap%:{avg_lap_pct:>5.1f}  "
        f"clip:{am.clip_rate:.3f}  "
        f"avg_rew:{avg_reward:>8.1f}"
    )

    trainlog.append([
        episode,
        episode_reward,
        step,
        traincounter,
        max_lap_pct_ep,
        avg_lap_pct,
        am.clip_rate,
        avg_reward,
    ])

    if saved:
        np.save(f'results/trainlog_{LOG_PREFIX}.npy', np.array(trainlog))
        print(f'  ✓ Log → results/trainlog_{LOG_PREFIX}.npy')
