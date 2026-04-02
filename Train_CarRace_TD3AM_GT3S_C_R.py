"""
Train_CarRace_TD3AM_GT3S.py
============================
GT3 Dynamic Action Mapping — TD3-AM training for the SimpleTrack circuit.

Features
--------
    - Resume from any saved checkpoint  (set RESUME_FROM below)
    - Alternating fixed/random spawn in SPAWN_BLOCK episode blocks
      so the policy learns the start chicane (fixed) AND the full circuit (random)
    - [FILL]/[TRAIN] phase tag per episode
    - [FIXED]/[RAND] spawn tag per episode
    - Rolling 20-episode averages for reward and lap%
    - Auto-advancing checkpoint numbering on resume

To resume training
------------------
    Set RESUME_FROM to the path of the checkpoint you want to continue from:
        RESUME_FROM = 'models/simple_gt3_model_3'

    Set RESUME_FROM = None to start fresh.

Spawn strategy
--------------
    Episodes 0–49   : FIXED  (start/finish line, heading north, spd=10 m/s)
    Episodes 50–99  : RAND   (weighted random across all straights)
    Episodes 100–149: FIXED  ... and so on.

    SPAWN_BLOCK = 50 controls the block size.
    The FIXED episodes give the policy honest lap% numbers directly comparable
    to evaluation run results (Run_CarRace_ExampleModel).

Required files in working directory
-------------------------------------
    CarModel_Kinematic.py   (GT3 edition)
    SimpleTrackEnv.py       (GT3 edition — 5 fixes applied)
    SimpleTrack.py          (original, unchanged)
    DynamicActionMapping.py (GT3 dynamic action mapping)
    TD3.py                  (original, unchanged)
    utils.py                (original, unchanged)
"""

import os
import glob
import numpy as np
import TD3
import utils
from DynamicActionMapping import DynamicActionMappingClass
from SimpleTrackEnv import SimpleTrackEnvClass

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION  ← edit these lines
# ═══════════════════════════════════════════════════════════════════════════
RESUME_FROM = None   # e.g. 'models/simple_gt3_model_3', or None for fresh start
SPAWN_BLOCK = 50     # episodes per spawn mode (fixed vs random alternating)
LOG_PREFIX  = 'simple_gt3'
# ═══════════════════════════════════════════════════════════════════════════


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
env           = SimpleTrackEnvClass(am=am)

# ── Resume logic ───────────────────────────────────────────────────────────
trainlog = []

if RESUME_FROM is not None:
    actor_file = RESUME_FROM + '_actor'
    if not os.path.exists(actor_file):
        raise FileNotFoundError(
            f"Cannot resume: '{actor_file}' not found.\n"
            f"Available: {sorted(glob.glob('models/simple_gt3_model_*_actor'))}"
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
    resuming    = False

print(f"\n{'='*60}")
print(f"  Training : TD3-AM GT3  |  Track : SIMPLE")
print(f"  Circuit  : {env.track.total_trip:.0f} m")
print(f"  Resume   : {RESUME_FROM or 'No — starting from scratch'}")
print(f"  Spawn    : alternating FIXED/RAND every {SPAWN_BLOCK} episodes")
print(f"  Next save: model_{savecounter}")
print(f"{'='*60}\n")

# ── Training counters ──────────────────────────────────────────────────────
stepcounter  = 0
traincounter = 0

WINDOW         = 20
reward_window  = []
lap_pct_window = []

# ── Training loop ──────────────────────────────────────────────────────────
for episode in range(100_000):

    # ── Spawn selection ────────────────────────────────────────────────────
    use_fixed   = (episode // SPAWN_BLOCK) % 2 == 0
    spawn_label = 'FIXED' if use_fixed else 'RAND '

    if use_fixed:
        ob = env.test_reset()   # start/finish line — honest lap% comparable to eval
    else:
        ob = env.reset()        # weighted random spawn

    ob   = process_state(ob)
    done = False
    am.reset_diagnostics()

    episode_reward    = 0
    episode_timesteps = 0
    saved             = False
    max_lap_pct_ep    = 0.0

    for step in range(10000):
        stepcounter += 1

        # ── Action selection ───────────────────────────────────────────────
        if stepcounter < args['start_timesteps']:
            if resuming:
                # Guided exploration: loaded actor + larger noise
                noise  = np.random.normal(0, max_action * 0.3, size=action_dim)
                action = (policy.select_action(ob) + noise).clip(-max_action, max_action)
            else:
                action = np.random.uniform(-1, 1, action_dim)
        else:
            noise  = np.random.normal(0, max_action * args['expl_noise'],
                                      size=action_dim)
            action = (policy.select_action(ob) + noise).clip(-max_action, max_action)

        # ── Dynamic action mapping ─────────────────────────────────────────
        ax_prev   = env.car.long_acc   # always initialised in GT3 CarModelClass
        action_in = am.mapping(env.car.spd, ax_prev, action[0], action[1])

        # ── Step ──────────────────────────────────────────────────────────
        next_ob, r, done = env.step(action_in)
        next_ob = process_state(next_ob)

        # Lap progress
        if env.lap_pct > max_lap_pct_ep:
            max_lap_pct_ep = env.lap_pct

        replay_buffer.add(ob, action, next_ob, r, done)
        ob              = next_ob
        episode_reward += r
        episode_timesteps += 1

        if done:
            break

        # ── Policy update ──────────────────────────────────────────────────
        if stepcounter > args['start_timesteps']:
            policy.train(replay_buffer, args['batch_size'])
            traincounter += 1

        # ── Checkpoint ────────────────────────────────────────────────────
        if traincounter % 100000 == 0 and not saved:
            name = f'models/{LOG_PREFIX}_model_{savecounter}'
            policy.save(name)
            savecounter += 1
            saved = True
            print(f'  ✓ Saved → {name}')

    # ── End-of-episode ─────────────────────────────────────────────────────
    fail_reason    = env.query_fail_reason()
    max_lap_pct_ep = env.max_lap_pct   # always exists now (FIX 4)

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
