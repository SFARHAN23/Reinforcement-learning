"""
Train_CarRace_TD3AM_GT3S.py  —  SimpleTrack, all-corner curriculum
====================================================================
RESUME_FROM : path to checkpoint, or None to start fresh
SPAWN_BLOCK : episodes per spawn mode

Spawn rotation (5 corners + 1 RAND, no FIXED):
    block 0  : T1  spawn@102m,  spd=11.2 m/s
    block 1  : T2  spawn@254m,  spd=7.9  m/s
    block 2  : T3  spawn@451m,  spd=9.1  m/s
    block 3  : T4  spawn@571m,  spd=9.1  m/s
    block 4  : T5  spawn@681m,  spd=11.2 m/s
    block 5  : RAND (general coverage)
    → cycle repeats every 6 × SPAWN_BLOCK episodes

Why no FIXED:
    FIXED episodes that crash at T1 inject Q(T1)≈-100 into the buffer,
    permanently suppressing exploration past the corner.  FIXED is removed
    from the rotation until corner spawns show lap% > 30%.  Add it back
    manually by changing the rotation below once that threshold is reached.

Why spawn 3m before entry at 55% v_max:
    The previous CORNER spawn (33m before, 8 m/s) still failed because the
    policy accelerates during the run-up, arriving at v >> v_max.
    3m run-up at 55% v_max gives almost no time to accelerate, so even a
    partially-random policy survives the corner.
"""

import os, glob
import numpy as np
import TD3, utils
from DynamicActionMapping import DynamicActionMappingClass
from SimpleTrackEnv import SimpleTrackEnvClass
from CarModel_Kinematic import CarModelClass
from corner_physics import scan_v_ref_info, VREF_PLANNING_DECEL
from curriculum import (
    build_spawn_schedule, build_finetune_schedule,
    SIMPLE_CORNERS, SIMPLE_FINE_TUNE_APPROACHES,
    next_corner_complex_exit, wrapped_trip_distance
)

# ═══════════════════════════════════════════════════════════════════════════
RESUME_FROM = None    # auto-resume from the latest checkpoint
SPAWN_BLOCK = 30      # episodes per corner block (shorter → faster curriculum cycling)
FINE_TUNE_BLOCK = 15
FINE_TUNE_AFTER = 300_000
FINE_TUNE_EXPL_NOISE = 0.05
SAVE_EVERY = 25_000
FT_ENV_REWARD_WEIGHT = 0.01
FT_PROGRESS_REWARD = 4.0
FT_MILESTONE_STEP = 10.0
FT_MILESTONE_BONUS = 25.0
FT_STEP_PENALTY = 0.10
FT_SUCCESS_BONUS = 2500.0
FT_FAIL_PENALTY = 300.0
FT_STALL_WINDOW = 80
FT_STALL_DELTA = 2.0
FT_MIN_TARGET_DIST = 40.0
FT_OVERSPEED_MARGIN = 1.0
FT_OVERSPEED_PENALTY = 6.0
FT_URGENCY_PENALTY = 10.0
FT_BRAKE_BONUS = 4.0
FT_THROTTLE_PENALTY = 4.0
FT_CLIP_PENALTY = 3.0
LOG_PREFIX  = 'simple_gt3'
# ═══════════════════════════════════════════════════════════════════════════

def process_state(s):
    return np.reshape(s, [1, -1])


def resolve_resume_from(log_prefix, explicit_path):
    if explicit_path is not None:
        return explicit_path

    nums = [int(f.split('_model_')[1].replace('_actor', ''))
            for f in glob.glob(f'models/{log_prefix}_model_*_actor')
            if f.split('_model_')[1].replace('_actor', '').isdigit()]
    if not nums:
        return None
    return f'models/{log_prefix}_model_{max(nums)}'

state_dim, action_dim, max_action = 34, 2, 1
args = dict(start_timesteps=1e4, expl_noise=0.1, batch_size=256,
            discount=0.99, tau=0.005, policy_noise=0.2,
            noise_clip=0.5, policy_freq=2)
kwargs = dict(state_dim=state_dim, action_dim=action_dim, max_action=max_action,
              discount=args['discount'], tau=args['tau'],
              policy_noise=args['policy_noise']*max_action,
              noise_clip=args['noise_clip']*max_action,
              policy_freq=args['policy_freq'])

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

policy        = TD3.TD3(**kwargs)
replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=int(5e6))
am            = DynamicActionMappingClass()
env           = SimpleTrackEnvClass(am=am)

# Build corner curriculum
curriculum_get_spawn, spawn_params = build_spawn_schedule(SIMPLE_CORNERS, SPAWN_BLOCK)
finetune_get_spawn, finetune_params = build_finetune_schedule(
    SIMPLE_FINE_TUNE_APPROACHES, FINE_TUNE_BLOCK,
    fixed_blocks=4, bridge_blocks=1, rand_blocks=1
)

# ── Resume ─────────────────────────────────────────────────────────────────
trainlog, resuming = [], False
resume_traincounter = 0
RESUME_FROM = resolve_resume_from(LOG_PREFIX, RESUME_FROM)

if RESUME_FROM is not None:
    if not os.path.exists(RESUME_FROM + '_actor'):
        raise FileNotFoundError(f"No model at '{RESUME_FROM}_actor'")
    try:
        full = policy.load(RESUME_FROM)
    except Exception:
        full = policy.load_gpu2cpu(RESUME_FROM)
    critic_loaded = full if isinstance(full, bool) else False
    print(f"  ✓ Loaded actor from '{RESUME_FROM}'  critic={'yes' if critic_loaded else 'no (old ckpt)'}")

    nums = [int(f.split('_model_')[1].replace('_actor',''))
            for f in glob.glob(f'models/{LOG_PREFIX}_model_*_actor')
            if f.split('_model_')[1].replace('_actor','').isdigit()]
    savecounter = max(nums)+1 if nums else 1

    lp = f'results/trainlog_{LOG_PREFIX}.npy'
    if os.path.exists(lp):
        trainlog = [list(r) for r in np.load(lp, allow_pickle=True).tolist()]
        print(f"  ✓ Trainlog loaded ({len(trainlog)} eps)")
        if trainlog:
            resume_traincounter = int(trainlog[-1][3])
    resuming = True
else:
    savecounter = 1

n_corners   = len(SIMPLE_CORNERS)
cycle_steps = (n_corners + 1) * SPAWN_BLOCK

print(f"\n{'='*62}")
print(f"  SimpleTrack GT3 — all-corner curriculum")
print(f"  Circuit : {env.track.total_trip:.0f} m  |  Corners : {n_corners}")
print(f"  Cycle   : {n_corners} corners + 1 RAND × {SPAWN_BLOCK} eps = {cycle_steps} eps/cycle")
print(f"  FineTune: start-line phase after ctr {FINE_TUNE_AFTER:,}  "
      f"(3×FIXED + T1 approach + RAND, block={FINE_TUNE_BLOCK})")
print(f"  Save    : every {SAVE_EVERY:,} train updates")
print(f"  Resume  : {RESUME_FROM or 'fresh start'}")
if resume_traincounter:
    print(f"  Ctr     : resume from {resume_traincounter:,}")
for trip, spd, label, vmax in spawn_params:
    print(f"    {label:<12} spawn@{trip:.0f}m  spd={spd:.1f}m/s  ({spd/vmax*100:.0f}% of v_max)")
for mode_name, trip, spd, label, vmax, dist in finetune_params:
    print(f"    FT {mode_name[:3].upper():<3} {label:<10} spawn@{trip:.0f}m  "
          f"spd={spd:.1f}m/s  ({dist:.0f}m run-up)")
print(f"{'='*62}\n")

stepcounter = 0
traincounter = resume_traincounter
saved = False
WINDOW = 20
rw, lw = [], []
schedule_mode = 'fine_tune' if traincounter >= FINE_TUNE_AFTER else 'curriculum'
phase_episode = 0

def schedule_tag(mode):
    return '[FTUNE]' if mode == 'fine_tune' else '[CURR ]'

for episode in range(100_000):
    active_mode = 'fine_tune' if traincounter >= FINE_TUNE_AFTER else 'curriculum'
    if active_mode != schedule_mode:
        schedule_mode = active_mode
        phase_episode = 0
        print(f"\n{'='*62}")
        print(f"  Switching to {'start-line fine-tune' if schedule_mode == 'fine_tune' else 'corner curriculum'}")
        print(f"{'='*62}\n")

    if schedule_mode == 'fine_tune':
        mode, trip, spd, label = finetune_get_spawn(phase_episode)
    else:
        mode, trip, spd, label = curriculum_get_spawn(phase_episode)
    phase_episode += 1

    if mode in ('corner', 'approach', 'bridge'):
        pose     = env.track.custom_car_pose(trip, 0)
        env.car  = CarModelClass(pose, spd)
        env.reset_flags()
        ob = env.observe()
        env._last_ob = ob
    elif mode == 'fixed':
        ob = env.test_reset()
    else:
        ob = env.reset()

    spawn_pct = env.track.car_trip / env.track.total_trip * 100.0
    spawn_trip = env.track.car_trip
    target_trip = None
    target_dist = 0.0
    prev_progress_m = 0.0
    best_progress_m = 0.0
    last_improve_progress_m = 0.0
    last_improve_step = 0
    local_done_reason = None
    use_local_goal = schedule_mode == 'fine_tune' and mode in ('fixed', 'approach', 'bridge')
    if use_local_goal:
        target_trip, _, _ = next_corner_complex_exit(env.track, spawn_trip)
        target_dist = max(
            FT_MIN_TARGET_DIST,
            wrapped_trip_distance(env.track.total_trip, spawn_trip, target_trip)
        )
    ob = process_state(ob)
    am.reset_diagnostics()
    episode_reward = max_lap = 0.0
    saved = False

    for step in range(10000):
        stepcounter += 1

        if stepcounter < args['start_timesteps']:
            if resuming:
                noise_std = 0.15 if schedule_mode == 'fine_tune' else 0.3
                noise  = np.random.normal(0, noise_std, size=action_dim)
                action = (policy.select_action(ob)+noise).clip(-1,1)
            else:
                action = np.random.uniform(-1, 1, action_dim)
        else:
            curr_base_noise = max(0.02, args['expl_noise'] - (args['expl_noise'] - 0.02) * (stepcounter / 1000000.0))
            expl_noise = FINE_TUNE_EXPL_NOISE if schedule_mode == 'fine_tune' else curr_base_noise
            noise  = np.random.normal(0, expl_noise, size=action_dim)
            action = (policy.select_action(ob)+noise).clip(-1,1)

        ax_prev   = env.car.long_acc
        action_in = am.mapping(env.car.spd, ax_prev, action[0], action[1])
        next_ob, r, done = env.step(action_in)
        next_ob = process_state(next_ob)

        if use_local_goal:
            curr_progress_m = min(
                wrapped_trip_distance(env.track.total_trip, spawn_trip, env.track.car_trip),
                target_dist
            )
            v_ref_ahead, dist_to_corner, _ = scan_v_ref_info(
                env.track, env.track.car_trip, vx=env.car.spd
            )
            overspeed = max(0.0, env.car.spd - v_ref_ahead - FT_OVERSPEED_MARGIN)
            overspeed_ratio = overspeed / max(v_ref_ahead, 1.0)
            braking_need = max(0.0, env.car.spd * env.car.spd - v_ref_ahead * v_ref_ahead)
            braking_need /= (2.0 * VREF_PLANNING_DECEL)
            late_ratio = max(0.0, braking_need - dist_to_corner) / max(dist_to_corner, 1.0)
            delta_progress_m = max(0.0, curr_progress_m - prev_progress_m)
            prev_best = best_progress_m
            best_progress_m = max(best_progress_m, curr_progress_m)
            prev_progress_m = curr_progress_m

            if best_progress_m >= last_improve_progress_m + FT_STALL_DELTA:
                last_improve_step = step
                last_improve_progress_m = best_progress_m

            prev_mark = int(prev_best // FT_MILESTONE_STEP)
            best_mark = int(best_progress_m // FT_MILESTONE_STEP)
            milestone_gain = max(0, best_mark - prev_mark)

            r_ft = FT_ENV_REWARD_WEIGHT * r
            r_ft += FT_PROGRESS_REWARD * delta_progress_m
            r_ft += FT_MILESTONE_BONUS * milestone_gain
            r_ft -= FT_STEP_PENALTY

            if overspeed_ratio > 0.0:
                r_ft -= FT_OVERSPEED_PENALTY * overspeed_ratio
                r_ft -= FT_URGENCY_PENALTY * min(1.0, late_ratio)
                if env.car.long_acc < 0.0:
                    brake_ratio = min(1.0, -env.car.long_acc / VREF_PLANNING_DECEL)
                    r_ft += FT_BRAKE_BONUS * overspeed_ratio * brake_ratio
                if action_in[0] > 0.0:
                    r_ft -= FT_THROTTLE_PENALTY * overspeed_ratio * action_in[0]
                if am.last_result is not None and am.last_result.was_clipped:
                    r_ft -= FT_CLIP_PENALTY * overspeed_ratio

            if best_progress_m >= target_dist - 1e-6:
                r_ft += FT_SUCCESS_BONUS
                done = True
                local_done_reason = 'TARGET'
            elif done:
                progress_frac = best_progress_m / max(target_dist, 1e-6)
                r_ft -= FT_FAIL_PENALTY * (1.0 - progress_frac)
            elif step - last_improve_step >= FT_STALL_WINDOW:
                progress_frac = best_progress_m / max(target_dist, 1e-6)
                r_ft -= FT_FAIL_PENALTY * (1.0 - progress_frac)
                done = True
                local_done_reason = 'STALL'

            r = r_ft

        if env.lap_pct > max_lap: max_lap = env.lap_pct
        replay_buffer.add(ob, action, next_ob, r, done)
        ob = next_ob; episode_reward += r

        if done: break

        if stepcounter > args['start_timesteps']:
            policy.train(replay_buffer, args['batch_size'])
            traincounter += 1

        if traincounter > 0 and traincounter % SAVE_EVERY == 0 and not saved:
            name = f'models/{LOG_PREFIX}_model_{savecounter}'
            policy.save(name); savecounter += 1; saved = True
            print(f'  ✓ Saved → {name}')

    fail  = local_done_reason or env.query_fail_reason()
    max_lap = env.max_lap_pct
    rw.append(episode_reward); lw.append(max_lap)
    if len(rw)>WINDOW: rw.pop(0)
    if len(lw)>WINDOW: lw.pop(0)
    phase = '[FILL ]' if stepcounter <= args['start_timesteps'] else '[TRAIN]'

    progress = max_lap
    print(f"ep:{episode:<5d} rew:{episode_reward:>8.1f}  steps:{step:<5d}  "
          f"ctr:{traincounter:<7d}  reason:{fail:<12s} {phase} {schedule_tag(schedule_mode)} [{label}]")
    extra = (f"  goal:{best_progress_m:>5.0f}/{target_dist:.0f}m"
             if use_local_goal else '')
    print(f"        lap%:{max_lap:>5.1f}  prog:{progress:>+6.1f}%  avg_lap%:{np.mean(lw):>5.1f}  "
          f"clip:{am.clip_rate:.3f}  avg_rew:{np.mean(rw):>8.1f}{extra}")

    trainlog.append([episode, episode_reward, step, traincounter,
                     max_lap, np.mean(lw), am.clip_rate, np.mean(rw)])
    if saved:
        np.save(f'results/trainlog_{LOG_PREFIX}.npy', np.array(trainlog))
