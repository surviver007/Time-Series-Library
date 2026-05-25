"""
RL 期货交易训练脚本
用法:
    python train_rl.py --symbol RB --total_timesteps 1000000
    python train_rl.py --symbol RB --pretrained ./checkpoints/xxx/checkpoint.pth --freeze_encoder
"""

import os
import argparse
import torch
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from rl.config import RLConfig
from rl.env.futures_env import FuturesEnvTrain, FuturesEnvVal
from rl.policy import build_ppo_model


def make_train_env(config):
    def _init():
        return FuturesEnvTrain(config)
    return _init


def make_eval_env(config):
    def _init():
        return FuturesEnvVal(config)
    return _init


def main():
    parser = argparse.ArgumentParser(description="RL 期货交易训练")
    parser.add_argument("--symbol", type=str, default="RB")
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--features_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=4096)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--capital", type=float, default=1_000_000)
    args = parser.parse_args()

    # 构建配置
    config = RLConfig(
        symbol=args.symbol,
        lookback_window=args.lookback,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        features_extractor_dim=args.features_dim,
        learning_rate=args.lr,
        total_timesteps=args.total_timesteps,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        initial_capital=args.capital,
        pretrained_path=args.pretrained,
        freeze_encoder=args.freeze_encoder,
    )

    print(f"=== RL 训练: {config.symbol} ===")
    print(f"  lookback={config.lookback_window}, d_model={config.d_model}, "
          f"n_heads={config.n_heads}, e_layers={config.e_layers}")
    print(f"  lr={config.learning_rate}, total_timesteps={config.total_timesteps}")
    print(f"  n_envs={args.n_envs}, device={args.device}")

    # 创建并行环境
    train_envs = VecMonitor(DummyVecEnv([make_train_env(config) for _ in range(args.n_envs)]))
    eval_env = VecMonitor(DummyVecEnv([make_eval_env(config)]))

    # 验证环境
    obs = train_envs.envs[0].reset()
    print(f"  观测空间: {obs[0].shape if isinstance(obs, tuple) else obs.shape}")
    print(f"  动作空间: {train_envs.envs[0].action_space}")

    # 构建模型
    model = build_ppo_model(config, train_envs, seed=args.seed, device=args.device)

    # 加载预训练权重
    if config.pretrained_path:
        print(f"  加载预训练: {config.pretrained_path}")
        state_dict = torch.load(config.pretrained_path, map_location="cpu")
        model.policy.features_extractor.load_pretrained(state_dict, freeze=config.freeze_encoder)

    # Callbacks
    save_dir = os.path.join(config.save_path, config.symbol)
    callbacks = CallbackList([
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(save_dir, "best"),
            log_path=os.path.join(save_dir, "eval"),
            eval_freq=max(config.eval_freq // args.n_envs, 1),
            n_eval_episodes=5,
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=max(config.save_freq // args.n_envs, 1),
            save_path=os.path.join(save_dir, "checkpoints"),
            name_prefix=f"ppo_{config.symbol}",
        ),
    ])

    # 训练
    print("\n开始训练...")
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        log_interval=config.log_interval,
    )

    # 保存最终模型
    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    print(f"\n训练完成! 模型保存到: {final_path}")
    print(f"  TensorBoard 日志: {config.tensorboard_log}")


if __name__ == "__main__":
    main()
