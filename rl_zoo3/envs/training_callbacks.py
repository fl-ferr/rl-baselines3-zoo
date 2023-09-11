from typing import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "gae_lambda": self.model.gae_lambda,
            "ent_coef": self.model.ent_coef,
            "n_epochs": self.model.n_epochs,
            "batch_size": self.model.batch_size,
            "n_steps": self.model.n_steps,
            "normalize_advantage": self.model.normalize_advantage,
            "max_grad_norm": self.model.max_grad_norm,
            "use_sde": self.model.use_sde,
            "target_kl": self.model.target_kl,
            "policy_kwargs": str(self.model.policy_kwargs),
        }
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
            "train/approx_kl": 0.0,
            "train/clip_fraction": 0.0,
            "train/explained_variance": 0.0,
        }

        self.logger.record(
            "hparams",
            HParam(hparam_dict=hparam_dict, metric_dict=metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True


class BatchNormExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space):
        super().__init__(observation_space, features_dim=observation_space.shape[0])
        self._batch_norm = nn.BatchNorm1d(
            num_features=observation_space.shape[0],
            momentum=0.95,
            eps=1e-5,
            affine=False,
            track_running_stats=True,
        )

    def forward(self, observations: np.ndarray) -> np.ndarray:
        return self._batch_norm(observations)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
        current learning rate depending on remaining progress
    """
    return lambda progress_remaining: progress_remaining * initial_value
