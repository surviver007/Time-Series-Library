"""
RL 交易系统统一配置
"""

from dataclasses import dataclass, field


@dataclass
class RLConfig:
    # ---- 数据 ----
    symbol: str = "RB"
    data_dir: str = "./dataset/futures_rl"
    lookback_window: int = 120         # 回看窗口 (~2 小时分钟线)
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    # test_ratio = 0.2 (剩余)

    # ---- 环境 ----
    initial_capital: float = 1_000_000.0
    multiplier: int = 10               # 合约乘数 (RB=10 吨)
    margin_rate: float = 0.10          # 保证金率 10%
    commission_per_lot: float = 2.0    # 单手单边手续费 (元)
    slippage: float = 1.0              # 滑点 (元/吨, 单边)
    flat_bonus: float = 0.0001         # 空仓奖励 (原始)
    reward_scale: float = 10000.0       # 奖励缩放因子 (净值变化率 -> 有效RL信号)
    flat_bonus_scaled: float = 0.0     # 缩放后空仓奖励 (已禁用)
    direction_bonus: float = 0.15      # 方向正确奖励 (需 > 单步交易成本奖励 ~0.12)
    direction_penalty: float = 0.08    # 方向错误惩罚
    max_position: int = 1              # 最大持仓手数
    episode_bars: int = 0              # 0 = 自动按日切分
    stop_loss_pct: float = 0.5         # 净值止损线 (相对初始资金)

    # ---- 特征 ----
    num_market_features: int = 22      # 市场特征数 (来自预处理)
    num_account_features: int = 6      # 账户状态特征数
    num_features: int = 28             # 总观测特征数
    num_action_bins: int = 3           # 动作离散化: long/short/flat

    # ---- iTransformer ----
    d_model: int = 64
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    activation: str = "gelu"
    features_extractor_dim: int = 128  # 特征提取器输出维度

    # ---- PPO ----
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # ---- 训练 ----
    total_timesteps: int = 1_000_000
    log_interval: int = 10
    eval_freq: int = 10_000
    save_freq: int = 50_000
    tensorboard_log: str = "./rl_logs/"
    save_path: str = "./rl_checkpoints/"

    # ---- 预训练 ----
    pretrained_path: str = ""
    freeze_encoder: bool = False

    # ---- 网络结构 ----
    net_arch_pi: list = field(default_factory=lambda: [64, 32])
    net_arch_vf: list = field(default_factory=lambda: [256, 128, 64])
