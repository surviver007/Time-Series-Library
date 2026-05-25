"""
PPO 策略配置
利用 SB3 内置的 features_extractor_class 机制, 无需自定义 Policy 类
此文件提供策略构建辅助函数
"""

from stable_baselines3 import PPO
from rl.feature_extractor import iTransformerFeatureExtractor


def build_policy_kwargs(config) -> dict:
    """构建 SB3 PPO 所需的 policy_kwargs"""
    return {
        "features_extractor_class": iTransformerFeatureExtractor,
        "features_extractor_kwargs": {
            "features_dim": config.features_extractor_dim,
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "e_layers": config.e_layers,
            "d_ff": config.d_ff,
            "dropout": config.dropout,
            "activation": config.activation,
        },
        "net_arch": dict(pi=config.net_arch_pi, vf=config.net_arch_vf),
    }


def build_ppo_model(config, train_env, eval_env=None, seed=42, device="auto"):
    """构建 PPO 模型"""
    from stable_baselines3.common.callbacks import EvalCallback

    policy_kwargs = build_policy_kwargs(config)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        device=device,
        tensorboard_log=config.tensorboard_log,
    )

    return model
