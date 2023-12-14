from typing import Dict, Type
from algorithms.ppo_jax import PPO_Jax_Adapted
from algorithms.random import Random
from open_spiel.python import rl_agent
from open_spiel.python.pytorch.dqn import DQN
from algorithms.ppo_adapted import PPO_Adapted
from algorithms.a2c_adapted import PolicyGradient_Adapted as A2C_Adapted
    
algo_name_to_algo_class : Dict[str, Type[rl_agent.AbstractAgent]] = {
    "random" : Random,
    "dqn" : DQN,
    "ppo" : PPO_Adapted,
    "a2c" : A2C_Adapted,
    "ppo_jax" : PPO_Jax_Adapted,
}