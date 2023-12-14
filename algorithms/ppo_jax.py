# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import time

import numpy as np
import torch
from torch import nn
from torch import optim


from open_spiel.python.jax.policy_gradient import PolicyGradient


class PPO_Jax_Adapted(PolicyGradient):

  def __init__(
      self,
      state_representation_size : int,
      num_actions,
      player_id,
      **kwargs,
      ):
    
    super().__init__(
      player_id = player_id, 
      info_state_size = state_representation_size,
      num_actions = num_actions, 
      **kwargs,
      )