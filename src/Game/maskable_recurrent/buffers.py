from typing import Generator, Optional, Tuple, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecNormalize

from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer, create_sequencers
from sb3_contrib.common.recurrent.type_aliases import RNNStates

from .type_aliases import MaskableRecurrentRolloutBufferSamples


class MaskableRecurrentRolloutBuffer(RecurrentRolloutBuffer):
    """RecurrentRolloutBuffer that additionally stores per-step action masks."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_state_shape: Tuple[int, int, int, int],
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ) -> None:
        super().__init__(
            buffer_size, observation_space, action_space,
            hidden_state_shape, device, gae_lambda, gamma, n_envs,
        )

    def reset(self) -> None:
        super().reset()
        # action_masks: shape (buffer_size, n_envs, n_actions), default all-valid
        assert isinstance(self.action_space, spaces.Discrete), (
            "MaskableRecurrentRolloutBuffer only supports Discrete action spaces"
        )
        self.action_masks = np.ones(
            (self.buffer_size, self.n_envs, int(self.action_space.n)), dtype=np.float32
        )

    def add(
        self,
        *args: object,
        lstm_states: RNNStates,
        action_masks: Optional[np.ndarray] = None,
        **kwargs: object,
    ) -> None:
        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.reshape(
                (self.n_envs, int(self.action_space.n))
            )
        super().add(*args, lstm_states=lstm_states, **kwargs)

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[MaskableRecurrentRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"
        # Prepare action_masks before the parent's generator_ready block runs
        if not self.generator_ready:
            self.action_masks = self.swap_and_flatten(self.action_masks)
        yield from super().get(batch_size)

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> MaskableRecurrentRolloutBufferSamples:
        # parent sets self.seq_start_indices, self.pad, self.pad_and_flatten
        base = super()._get_samples(batch_inds, env_change, env)
        padded_batch_size = base.actions.shape[0]

        # Pad with 1.0 so padded timesteps allow all actions (avoids log(0) on padded steps)
        action_masks_padded = self.pad(self.action_masks[batch_inds], padding_value=1.0)
        action_masks_flat = action_masks_padded.reshape((padded_batch_size, -1))

        return MaskableRecurrentRolloutBufferSamples(
            observations=base.observations,
            actions=base.actions,
            old_values=base.old_values,
            old_log_prob=base.old_log_prob,
            advantages=base.advantages,
            returns=base.returns,
            lstm_states=base.lstm_states,
            episode_starts=base.episode_starts,
            mask=base.mask,
            action_masks=action_masks_flat,
        )
