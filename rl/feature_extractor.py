"""
iTransformer 特征提取器 — 用于 Stable-Baselines3 PPO
复用项目中 layers/ 的 DataEmbedding_inverted + Encoder
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class iTransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    使用 iTransformer encoder 从观测序列中提取特征

    输入: (B, num_features, lookback_window)
    处理: permute → instance norm → DataEmbedding_inverted → Encoder → flatten → Linear
    输出: (B, features_dim)
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128,
                 d_model: int = 64, n_heads: int = 4, e_layers: int = 2,
                 d_ff: int = 256, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__(observation_space, features_dim)

        num_features = observation_space.shape[0]   # 28
        lookback = observation_space.shape[1]        # 120

        # iTransformer embedding: 每个特征的时序 (length=lookback) 映射到 d_model
        self.enc_embedding = DataEmbedding_inverted(
            c_in=lookback,
            d_model=d_model,
            dropout=dropout,
        )

        # iTransformer encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,            # mask_flag
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        # 投影层
        self.flatten_dim = num_features * d_model
        self.projection = nn.Sequential(
            nn.Linear(self.flatten_dim, features_dim),
            nn.ReLU(),
        )

        self._d_model = d_model
        self._num_features = num_features

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: (B, num_features, lookback_window)
        Returns:
            features: (B, features_dim)
        """
        # 防护: 确保 input 无 NaN/inf
        observations = torch.nan_to_num(observations, nan=0.0, posinf=10.0, neginf=-10.0)

        # (B, num_features, lookback) → (B, lookback, num_features) iTransformer 输入格式
        x = observations.permute(0, 2, 1).contiguous()

        # Instance normalization (与 iTransformer.forecast() 一致)
        means = x.mean(dim=1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x = x / stdev

        # Embedding: (B, lookback, num_features) → DataEmbedding_inverted 内部 permute
        # → (B, num_features, lookback) → Linear(lookback, d_model) → (B, num_features, d_model)
        enc_out = self.enc_embedding(x, None)

        # Encoder: (B, num_features, d_model) → (B, num_features, d_model)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        # Flatten + 投影
        enc_out = enc_out.reshape(enc_out.shape[0], -1)  # (B, num_features * d_model)
        features = self.projection(enc_out)                # (B, features_dim)

        return features

    def load_pretrained(self, state_dict: dict, freeze: bool = False):
        """
        加载预训练的 iTransformer 权重 (部分匹配)

        Args:
            state_dict: 预训练模型的 state_dict
            freeze: 是否冻结 encoder 参数
        """
        own_keys = set(self.state_dict().keys())
        matched = {k: v for k, v in state_dict.items() if k in own_keys}
        self.load_state_dict(matched, strict=False)
        print(f"加载预训练权重: {len(matched)}/{len(own_keys)} 个参数匹配")

        if freeze:
            for name, param in self.named_parameters():
                if "projection" not in name:
                    param.requires_grad = False
            print("已冻结 encoder 参数")
