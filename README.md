# Action Value Gradient Algorithm

This repo provides an implementation of the following incremental learning algorithms:
- Action Value Gradient (AVG)
- Incremental One-Step Actor-Critic (IAC)
- Incremental Soft Actor Critic (SAC-1)


```python
python avg.py --env "Humanoid-v4" --N 10001000
```

## Robot Tasks

| ![UR-Reacher-2](assets/UR-Reacher-2.gif) <br> UR-Reacher-2 | ![Create-Mover](assets/Create-Mover.gif) <br /> Create-Mover |
| --- | --- |

## Hyper-parameter search
*AVG*
```
cd incremental_rl
python hyp_sweep.py --algo "avg" --hyp_seed 122 --env "Hopper-v4" --N 10001000 --n_seeds 10
python replicate_run.py --algo "avg_norm_obs_scaled_td" --hyp_seed 129 --env "Ant-v4" --N 10001000
```

*Incremental Actor Critic*
```
cd incremental_rl
python hyp_sweep.py --algo "iac" --hyp_seed 122 --env "Hopper-v4" --N 10001000 --n_seeds 10
python replicate_run.py --algo "iac_all" --hyp_seed 294 --env "Hopper-v4" --N 10001000
```

*Incremental Soft Actor Critic*
```
cd incremental_rl
python hyp_sweep.py --algo "isac" --hyp_seed 146 --env "HalfCheetah-v4" --N 10001000 
```

## Cite
```bash
@inproceedings{vasan2024deep,
  title={Deep Policy Gradient Methods Without Batch Updates, Target Networks, or Replay Buffers},
  author={Vasan, Gautham and Elsayed, Mohamed and Azimi, Seyed Alireza and He, Jiamin and Shahriar, Fahim and Bellinger, Colin and White, Martha and Mahmood, A Rupam},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}

```