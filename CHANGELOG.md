# Changelog

## [v26.4.2.1] - 2026-04-02
### GT3 RL Physics and Environment Major Fixes
- **Physics**: Reworked `CarModel_Kinematic.py` to map the RL actor's lateral output to a target steering geometry heavily bounded by physical limits (`v^2/R ≤ max_grip`). This replaces the fatal steering rate bug.
- **Understeer Mechanic**: Replaced artificial `MAX_ACC` terminations with gradual understeer slipping physics.
- **Metrics**: Modified `lap_pct` math inside `SimpleTrackEnv.py` and `MonzaTrackEnv.py` to track precise linear progress based on correct sequence wrapping relative to `spawn_trip`.
- **Completion States**: Added proper `env.FINISH = True` termination boundaries logic delivering sequence success (+1000 lap boundary).
- **Network Boundaries**: Normalized distances on extended lookahead blocks within `observe()` to remain between `[-1.0, 1.0]` preventing actor saturation limits.
- **Curriculum Training**: Injected chronological noise decay scaling (`expl_noise` fades from 0.1 to 0.02) to maintain high speed GT3 maneuverability.
