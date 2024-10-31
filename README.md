# Action Value Gradient Algorithm

This repo provides an implementation of the following incremental learning algorithms:
- Action Value Gradient (AVG)
- Incremental One-Step Actor-Critic (IAC)
- Incremental Soft Actor Critic (SAC-1)


```python
python avg.py --env "Humanoid-v4" --N 10001000
```

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
Vasan, G., Elsayed, M., Azimi, S. A., He, J., Shahriar, F., Bellinger, C., White, M., & Mahmood, A. R. (2024). Deep policy gradient methods without batch updates, target networks, or replay buffers. To appear in Neural Information Processing Systems.
```