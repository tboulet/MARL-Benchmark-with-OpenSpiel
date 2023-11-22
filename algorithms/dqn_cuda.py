import collections
import math
import sys
import numpy as np
from scipy import stats
import torch
from torch import nn
import torch.nn.functional as F

from open_spiel.python import rl_agent
from open_spiel.python.utils.replay_buffer import ReplayBuffer

Transition = collections.namedtuple(
    "Transition",
    "info_state action reward next_info_state is_final_step legal_actions_mask")

ILLEGAL_ACTION_LOGITS_PENALTY = sys.float_info.min





from open_spiel.python.pytorch.dqn import DQN


class DQN_Cuda(DQN):

  def __init__(self, *args, device : str = "cpu", **kwargs):
    self.device = device
    super().__init__(*args, **kwargs)
    self.to(self.device)

  def to(self, device):
    self._q_network.to(device)
    self._target_q_network.to(device)
    
  def get_device(self):
    return self.device
        

  def _epsilon_greedy(self, info_state, legal_actions, epsilon):
    """Returns a valid epsilon-greedy action and valid action probs.

    Action probabilities are given by a softmax over legal q-values.

    Args:
      info_state: hashable representation of the information state.
      legal_actions: list of legal actions at `info_state`.
      epsilon: float, probability of taking an exploratory action.

    Returns:
      A valid epsilon-greedy action and valid action probabilities.
    """
    probs = np.zeros(self._num_actions)
    if np.random.rand() < epsilon:
      action = np.random.choice(legal_actions)
      probs[legal_actions] = 1.0 / len(legal_actions)
    else:      
      info_state = torch.tensor(np.reshape(info_state, [1, -1]), device=self.device, dtype=torch.float32)
      q_values = self._q_network(info_state).detach()[0]
      legal_q_values = q_values[legal_actions]
      action = legal_actions[torch.argmax(legal_q_values)]
      probs[action] = 1.0
    return action, probs

  def _get_epsilon(self, is_evaluation, power=1.0):
    """Returns the evaluation or decayed epsilon value."""
    if is_evaluation:
      return 0.0
    decay_steps = min(self._step_counter, self._epsilon_decay_duration)
    decayed_epsilon = (
        self._epsilon_end + (self._epsilon_start - self._epsilon_end) *
        (1 - decay_steps / self._epsilon_decay_duration)**power)
    return decayed_epsilon

  def learn(self):
    """Compute the loss on sampled transitions and perform a Q-network update.

    If there are not enough elements in the buffer, no loss is computed and
    `None` is returned instead.

    Returns:
      The average loss obtained on this batch of transitions or `None`.
    """

    if (len(self._replay_buffer) < self._batch_size or
        len(self._replay_buffer) < self._min_buffer_size_to_learn):
      return None

    transitions = self._replay_buffer.sample(self._batch_size)
    info_states = torch.tensor([t.info_state for t in transitions], device=self.device)
    actions = torch.LongTensor([t.action for t in transitions])
    rewards = torch.tensor([t.reward for t in transitions], device=self.device)
    next_info_states = torch.tensor([t.next_info_state for t in transitions], device=self.device)
    are_final_steps = torch.tensor([t.is_final_step for t in transitions], device=self.device)
    legal_actions_mask = torch.tensor(np.array([t.legal_actions_mask for t in transitions]), device=self.device)

    self._q_values = self._q_network(info_states)
    self._target_q_values = self._target_q_network(next_info_states).detach()

    illegal_actions_mask = 1 - legal_actions_mask
    legal_target_q_values = self._target_q_values.masked_fill(
        illegal_actions_mask.bool(), ILLEGAL_ACTION_LOGITS_PENALTY
    )
    max_next_q = torch.max(legal_target_q_values, dim=1)[0]

    target = (
        rewards + (1 - are_final_steps) * self._discount_factor * max_next_q)
    action_indices = torch.stack([
        torch.arange(self._q_values.shape[0], dtype=torch.long), actions
    ],
                                 dim=0)
    predictions = self._q_values[list(action_indices)]

    loss = self.loss_class(predictions, target)

    self._optimizer.zero_grad()
    loss.backward()
    self._optimizer.step()

    return loss

  @property
  def q_values(self):
    return self._q_values

  @property
  def replay_buffer(self):
    return self._replay_buffer

  @property
  def loss(self):
    return self._last_loss_value

  @property
  def prev_timestep(self):
    return self._prev_timestep

  @property
  def prev_action(self):
    return self._prev_action

  @property
  def step_counter(self):
    return self._step_counter

  def get_weights(self):
    variables = [m.weight for m in self._q_network.model]
    variables.append([m.weight for m in self._target_q_network.model])
    return variables

  def copy_with_noise(self, sigma=0.0, copy_weights=True):
    """Copies the object and perturbates it with noise.

    Args:
      sigma: gaussian dropout variance term : Multiplicative noise following
        (1+sigma*epsilon), epsilon standard gaussian variable, multiplies each
        model weight. sigma=0 means no perturbation.
      copy_weights: Boolean determining whether to copy model weights (True) or
        just model hyperparameters.

    Returns:
      Perturbated copy of the model.
    """
    _ = self._kwargs.pop("self", None)
    copied_object = DQN(**self._kwargs)

    q_network = getattr(copied_object, "_q_network")
    target_q_network = getattr(copied_object, "_target_q_network")

    if copy_weights:
      with torch.no_grad():
        for q_model in q_network.model:
          q_model.weight *= (1 + sigma * torch.randn(q_model.weight.shape))
        for tq_model in target_q_network.model:
          tq_model.weight *= (1 + sigma * torch.randn(tq_model.weight.shape))
    return copied_object

  def save(self, data_path, optimizer_data_path=None):
    """Save checkpoint/trained model and optimizer.

    Args:
      data_path: Path for saving model. It can be relative or absolute but the
        filename should be included. For example: q_network.pt or
        /path/to/q_network.pt
      optimizer_data_path: Path for saving the optimizer states. It can be
        relative or absolute but the filename should be included. For example:
        optimizer.pt or /path/to/optimizer.pt
    """
    torch.save(self._q_network, data_path)
    if optimizer_data_path is not None:
      torch.save(self._optimizer, optimizer_data_path)

  def load(self, data_path, optimizer_data_path=None):
    """Load checkpoint/trained model and optimizer.

    Args:
      data_path: Path for loading model. It can be relative or absolute but the
        filename should be included. For example: q_network.pt or
        /path/to/q_network.pt
      optimizer_data_path: Path for loading the optimizer states. It can be
        relative or absolute but the filename should be included. For example:
        optimizer.pt or /path/to/optimizer.pt
    """
    torch.load(self._q_network, data_path)
    torch.load(self._target_q_network, data_path)
    if optimizer_data_path is not None:
      torch.load(self._optimizer, optimizer_data_path)
