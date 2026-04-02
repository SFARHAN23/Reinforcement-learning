"""
Train_CarRace_TD3AM_GT3.py
===========================
GT3 Dynamic Action Mapping — TD3-AM training script.
Supports both circuits and resume-from-checkpoint.

── RESUME TRAINING ──────────────────────────────────────────────────────────
To continue from a saved checkpoint, set RESUME_FROM to the model path:

    RESUME_FROM = 'models/monza_gt3_model_3'   # resume from checkpoint 3
    RESUME_FROM = None                          # start fresh

What happens on resume:
    1. Actor weights are loaded from the checkpoint file.
    2. savecounter auto-advances past the highest existing checkpoint so
       the next save does not overwrite previous models.
    3. The existing trainlog is loaded and appended to (if it exists).
    4. The replay buffer starts EMPTY (it is not saved to disk).
       start_timesteps random-ish exploration runs again to fill it before
       TD3 critic updates begin.  With loaded actor weights this exploration
       is guided (actor + noise) rather than pure random — better coverage.

Why the critic isn't loaded:
    TD3.save() saves only the actor (critic save is commented out upstream).
    The critic must relearn from the fresh replay buffer, which it does
    quickly (~10k steps) since the actor already produces useful trajectories.

── TRACK SELECT ─────────────────────────────────────────────────────────────
    TRACK_SELECT = 'monza'    # 11-unit Monza GP (~5,987 m)
    TRACK_SELECT = 'simple'   # 5-unit oval-ish track (fast iteration)
"""

import os
import glob
import numpy as np
import TD3
import utils
from DynamicActionMapping import DynamicActionMappingClass

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION  ← edit these two lines
# ═══════════════════════════════════════════════════════════════════════════
TRACK_SELECT = 'monza'                       # 'monza' | 'simple'
RESUME_FROM  = None                          # path string, or None to start fresh
# ═══════════════════════════════════════════════════════════════════════════

if TRACK_SELECT == 'monza':
    from MonzaTrackEnv import MonzaTrackEnvClass as EnvClass
    LOG_PREFIX = 'monza_gt3'
else:
    from SimpleTrackEnv import SimpleTrackEnvClass as EnvClass
    LOG_PREFIX = 'simple_gt3'


def process_state(s):
    return np.reshape(s, [1, -1])


# ── Hyperparameters ────────────────────────────────────────────────────────
state_dim  = 34   # ENHANCED_OBS=True: 31 base + 3 extended (v_ref, long_acc, grip_util)
action_dim = 2
max_action = 1
dt         = 0.01

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

# ── Build objects ──────────────────────────────────────────────────────────
policy        = TD3.TD3(**kwargs)
replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=int(5e6))
am            = DynamicActionMappingClass()
env           = EnvClass()

# ── Resume logic ───────────────────────────────────────────────────────────
trainlog = []

if RESUME_FROM is not None:
    actor_file = RESUME_FROM + '_actor'
    if not os.path.exists(actor_file):
        raise FileNotFoundError(
            f"Cannot resume: actor file not found at '{actor_file}'.\n"
            f"Models in ./models/: {sorted(os.listdir('models/'))}"
        )

    # Load actor weights (try GPU-saved model on CPU if needed)
    try:
        policy.load(RESUME_FROM)
        print(f"  ✓ Loaded actor weights from '{RESUME_FROM}'")
    except Exception:
        policy.load_gpu2cpu(RESUME_FROM)
        print(f"  ✓ Loaded actor weights (GPU→CPU) from '{RESUME_FROM}'")

    # Auto-detect next save number by scanning existing checkpoints
    existing = glob.glob(f'models/{LOG_PREFIX}_model_*_actor')
    if existing:
        nums = []
        for f in existing:
            try:
                n = int(f.split('_model_')[1].replace('_actor', ''))
                nums.append(n)
            except ValueError:
                pass
        savecounter = max(nums) + 1
    else:
        savecounter = 1
    print(f"  ✓ Next checkpoint will be saved as model_{savecounter}")

    # Load existing trainlog if present (append new episodes to it)
    log_path = f'results/trainlog_{LOG_PREFIX}.npy'
    if os.path.exists(log_path):
        old_log = np.load(log_path, allow_pickle=True).tolist()
        trainlog = [list(row) for row in old_log]
        print(f"  ✓ Loaded existing trainlog ({len(trainlog)} episodes)")
    else:
        print(f"  ℹ  No existing trainlog found — starting fresh log")

    resuming = True

else:
    savecounter = 1
    resuming    = False

print(f"\n{'='*60}")
print(f"  Training : TD3-AM  |  Track : {TRACK_SELECT.upper()}")
print(f"  Circuit  : {env.track.total_trip:.0f} m")
print(f"  Resume   : {RESUME_FROM if RESUME_FROM else 'No — starting from scratch'}")
print(f"  Next save: model_{savecounter}")
print(f"{'='*60}\n")

# ── Training counters ──────────────────────────────────────────────────────
# stepcounter always starts from 0 because the replay buffer is empty.
# start_timesteps must run to fill the buffer before TD3 updates begin.
# This is correct even on resume — the critic starts fresh every time.
stepcounter  = 0
traincounter = 0
saved        = False

WINDOW         = 20
reward_window  = []
lap_pct_window = []

# ── Spawn strategy ────────────────────────────────────────────────────────
# Alternate between fixed start/finish spawning and random spawning in
# blocks of SPAWN_BLOCK episodes.
#
# Why this matters:
#   random_car_pose() weights spawn probability by straight length.
#   T1-T2 chicane has only a 10 m straight → almost never spawned.
#   The policy never encounters the first 100 m of the lap during training,
#   so it always fails immediately when evaluated from the start line.
#
# Alternating blocks give the policy equal exposure to:
#   - Fixed (start line): learns T1-T2 chicane, Curva Grande entry
#   - Random:             learns the rest of the circuit
#
# SPAWN_BLOCK = 50 means episodes 0-49 → fixed, 50-99 → random, etc.
SPAWN_BLOCK = 50   # number of consecutive episodes per spawn mode

# ── Training loop ──────────────────────────────────────────────────────────
for episode in range(100_000):   # large upper bound; Ctrl-C to stop

    # Determine spawn mode for this episode
    use_fixed_spawn = (episode // SPAWN_BLOCK) % 2 == 0
    spawn_label     = 'FIXED' if use_fixed_spawn else 'RAND '

    if use_fixed_spawn:
        ob = env.test_reset()   # always starts at position (0,0), heading north
    else:
        ob = env.reset()        # weighted random spawn across all straights

    ob   = process_state(ob)
    done = False
    am.reset_diagnostics()

    episode_reward    = 0
    episode_timesteps = 0
    saved             = False
    max_lap_pct_ep    = 0.0

    for step in range(10000):
        stepcounter += 1

        # Action selection
        if stepcounter < args['start_timesteps']:
            if resuming:
                # Use loaded actor + larger noise during buffer-fill phase.
                # Much better than pure random — actor already has some sense
                # of direction after training, so transitions are meaningful.
                noise  = np.random.normal(0, max_action * 0.3, size=action_dim)
                action = (policy.select_action(ob) + noise).clip(-max_action, max_action)
            else:
                action = np.random.uniform(-1, 1, action_dim)
        else:
            noise  = np.random.normal(0, max_action * args['expl_noise'],
                                      size=action_dim)
            action = (policy.select_action(ob) + noise).clip(-max_action, max_action)

        # Dynamic action mapping
        ax_prev   = env.car.long_acc
        action_in = am.mapping(env.car.spd, ax_prev, action[0], action[1])

        # Environment step
        next_ob, r, done = env.step(action_in)
        next_ob = process_state(next_ob)

        # Lap progress
        if hasattr(env, 'lap_pct') and env.lap_pct > max_lap_pct_ep:
            max_lap_pct_ep = env.lap_pct

        replay_buffer.add(ob, action, next_ob, r, done)
        ob              = next_ob
        episode_reward += r
        episode_timesteps += 1

        if done:
            break

        # Policy update (only after replay buffer has enough data)
        if stepcounter > args['start_timesteps']:
            policy.train(replay_buffer, args['batch_size'])
            traincounter += 1

        # Checkpoint save
        if traincounter % 100000 == 0 and not saved:
            name = f'models/{LOG_PREFIX}_model_{savecounter}'
            policy.save(name)
            savecounter += 1
            saved = True
            print(f'  ✓ Saved → {name}')

    # ── End-of-episode logging ─────────────────────────────────────────────
    fail_reason    = env.query_fail_reason()
    max_lap_pct_ep = getattr(env, 'max_lap_pct', max_lap_pct_ep)

    reward_window.append(episode_reward)
    lap_pct_window.append(max_lap_pct_ep)
    if len(reward_window)  > WINDOW: reward_window.pop(0)
    if len(lap_pct_window) > WINDOW: lap_pct_window.pop(0)
    avg_reward  = float(np.mean(reward_window))
    avg_lap_pct = float(np.mean(lap_pct_window))

    # Status line — shows [FILL] during buffer-fill phase, [TRAIN] after
    phase = '[FILL]' if stepcounter <= args['start_timesteps'] else '[TRAIN]'

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
