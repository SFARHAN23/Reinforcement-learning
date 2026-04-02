import glob
import os
import re

import numpy as np

import TD3
from DynamicActionMapping import DynamicActionMappingClass
from LogTools import CarStateLogClass
from RenderVideo import show_animi
from SimpleTrackEnv import SimpleTrackEnvClass


MODEL_PATH = None
RENDER_VIDEO = True


def process_state(state):
    return np.reshape(state, [1, -1])


def resolve_model_path(model_path, prefixes):
    if model_path:
        return model_path

    candidates = []
    for prefix in prefixes:
        pattern = os.path.join('models', f'{prefix}*_actor')
        for actor_path in glob.glob(pattern):
            base_path = actor_path[:-len('_actor')]
            base_name = os.path.basename(base_path)
            match = re.fullmatch(rf'{re.escape(prefix)}(\d+)', base_name)
            if match:
                candidates.append((int(match.group(1)), os.path.getmtime(actor_path), base_path))
        if candidates:
            break

    if not candidates:
        raise FileNotFoundError(
            f'No matching model checkpoints found in models/ for prefixes: {prefixes}'
        )

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2]


def load_policy_checkpoint(policy, model_path):
    try:
        policy.load_gpu2cpu(model_path)
        print(f'Model loaded (GPU->CPU) from {model_path}')
    except Exception:
        policy.load(model_path)
        print(f'Model loaded from {model_path}')


def detect_state_dim(model_path):
    """Detect state_dim from actor checkpoint's first layer weight shape."""
    import torch
    cpu = torch.device('cpu')
    actor_file = model_path + '_actor'
    sd = torch.load(actor_file, map_location=cpu, weights_only=True)
    # Actor.l1 is nn.Linear(state_dim, 256) → weight shape is (256, state_dim)
    detected = sd['l1.weight'].shape[1]
    return detected


kwargs = {
    "state_dim": 34,
    "action_dim": 2,
    "max_action": 1,
}

model_path = resolve_model_path(MODEL_PATH, prefixes=('simple_gt3_model_', 'model_'))

# Auto-detect state_dim from saved model weights
detected_dim = detect_state_dim(model_path)
if detected_dim != 34:
    print(f'  ⚠ Model was trained with state_dim={detected_dim} (current env uses 34)')
    print(f'    Loading in compatibility mode — results may differ from fresh training')
    kwargs['state_dim'] = detected_dim
    # Switch env to legacy 31-dim observation mode
    import SimpleTrackEnv as _ste
    _ste.ENHANCED_OBS = False
    _ste.STATE_DIM    = detected_dim

policy = TD3.TD3(**kwargs)
load_policy_checkpoint(policy, model_path)

am = DynamicActionMappingClass()
env = SimpleTrackEnvClass(am=am)
logger = CarStateLogClass(env)

os.makedirs('results', exist_ok=True)

episode_reward = 0.0
ob = process_state(env.test_reset()[:detected_dim])
pre_trip = 0.0
lap1_done = False
lap1_time = None

print(f'Running Track-A evaluation with {model_path}')

for step in range(10000):
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
        print(f'lap 1 finished!  lap time: {lap1_time / 100:.2f}')
        lap1_done = True

    elif current_trip < pre_trip and lap1_done:
        lap2_time = step - lap1_time
        if lap2_time > 1000:
            print(f'lap 2 finished!  lap time: {lap2_time / 100:.2f}')
            logger.log_data(step, lap1_done, list(action), list(action_in))
            break

    pre_trip = current_trip
    logger.log_data(step, lap1_done, list(action), list(action_in))

fail_reason = env.query_fail_reason()
print('Race Done! reward: %.1f  step: %d reason: %s' % (episode_reward, step, fail_reason))

if not lap1_done:
    max_pct = env.track.car_trip / env.track.total_trip * 100
    print('Lap 1 not completed. Maximum circuit progress: %.1f%%' % max_pct)

logger.show_trajectory(lap='lap1')
logger.show_states_controls(lap='lap1')

if lap1_done:
    logger.show_trajectory(lap='lap2')
    logger.show_states_controls(lap='lap2')

if RENDER_VIDEO:
    show_animi(env, logger, step, save=True, path='results/TrackA_Video2.mp4')
