from gymnasium.envs.registration import register

# Dot Reacher Easy
register(
    id='dot_reacher_easy',
    entry_point='incremental_rl.envs.dot_reacher_env:DotReacherEasy',
    max_episode_steps=20000
)

# Dot Reacher Hard
register(
    id='dot_reacher_hard',
    entry_point='incremental_rl.envs.dot_reacher_env:DotReacherHard',
    max_episode_steps=20000
)

# Point Maze
register(
    id='point_maze_open',
    entry_point='incremental_rl.envs.point_maze:PointMaze',
    max_episode_steps=10000,
    kwargs={"reward_type": "sparse", "map_type": "open",
            "render_mode": None, "use_image": False,},
)

register(
    id='point_maze_mt1',
    entry_point='incremental_rl.envs.point_maze:PointMaze',
    max_episode_steps=10000,
    kwargs={"reward_type": "sparse", "map_type": "min_time",
            "render_mode": None, "use_image": False,},
)

register(
    id='acrobot',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={'domain': "acrobot", "task": "swingup"},
)

register(
    id='acrobot_sparse',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={'domain': "acrobot", "task": "swingup"},
)

register(
    id='ball_in_cup',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={'domain': "ball_in_cup", "task": "catch"},
)

register(
    id='cartpole_balance',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "cartpole", "task": "balance"},
)

register(
    id='cartpole_balance_sparse',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "cartpole", "task": "balance_sparse"},
)

register(
    id='cartpole_swingup',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "cartpole", "task": "swingup"},
)

register(
    id='cartpole_swingup_sparse',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "cartpole", "task": "swingup_sparse"},
)

register(
    id='cheetah_run',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "cheetah", "task": "run"},
)

register(
    id='dog_stand',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "dog", "task": "stand"},
)

register(
    id='dog_walk',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "dog", "task": "walk"},
)

register(
    id='dog_trot',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "dog", "task": "trot"},
)

register(
    id='dog_run',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "dog", "task": "run"},
)

register(
    id='dog_fetch',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "dog", "task": "fetch"},
)

register(
    id='finger_spin',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "finger", "task": "spin"},
)

register(
    id='finger_turn_easy',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "finger", "task": "turn_easy"},
)

register(
    id='finger_turn_hard',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "finger", "task": "turn_hard"},
)

register(
    id='fish_upright',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "fish", "task": "upright"},
)

register(
    id='fish_swim',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "fish", "task": "swim"},
)

register(
    id='hopper_stand',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "hopper", "task": "stand"},
)

register(
    id='hopper_hop',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "hopper", "task": "hop"},
)

register(
    id='humanoid_stand',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "humanoid", "task": "stand"},
)

register(
    id='humanoid_walk',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "humanoid", "task": "walk"},
)

register(
    id='humanoid_run',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "humanoid", "task": "run"},
)

register(
    id='manipulator_bring_ball',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "manipulator", "task": "bring_ball"},
)

register(
    id='pendulum_swingup',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "pendulum", "task": "swingup"},
)

register(
    id='point_mass_easy',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "point_mass", "task": "easy"},
)

register(
    id='quadruped_fetch',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "quadruped", "task": "fetch"},
)

register(
    id='quadruped_run',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "quadruped", "task": "run"},
)

register(
    id='quadruped_walk',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "quadruped", "task": "walk"},
)

register(
    id='reacher_easy',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "reacher", "task": "easy"},
)

register(
    id='reacher_hard',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "reacher", "task": "hard"},
)

register(
    id='swimmer6',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "swimmer", "task": "swimmer6"},
)

register(
    id='swimmer15',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "swimmer", "task": "swimmer15"},
)

register(
    id='walker_stand',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "walker", "task": "stand"},
)

register(
    id='walker_walk',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "walker", "task": "walk"},
)

register(
    id='walker_run',
    entry_point='incremental_rl.envs.dm_control_wrapper:DMControl',
    max_episode_steps=1000,
    kwargs={"domain": "walker", "task": "run"},
)
