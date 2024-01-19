# Imports
import pyspiel

import datetime
from typing import Any, List, Callable, Tuple
from matplotlib import pyplot as plt
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

# Define some types for lisibility
Policy = List[float]
JointPolicy = List[Policy]   # if p is of type Policy, then p[i][a] = p_i(a)
Game = Any

from algorithms.nfg_algorithms import algo_name_to_nfg_solver

@hydra.main(config_path="configs_nfg", config_name="nfg_default.yaml")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    config = OmegaConf.to_container(config, resolve=True)
    
    # Get the config parameters
    game_name = config["game"]["game_name"]
    algo_name = config["algo"]["algo_name"]
    algo_config = config["algo"]["algo_config"]
    n_episodes_training = config["n_episodes_training"]
    
    # Get the game
    game = pyspiel.load_game(game_name)
    
    # Initialize the algorithm for learning in that game
    AlgoClass = algo_name_to_nfg_solver[algo_name]
    algo = AlgoClass(**algo_config)
    algo.initialize_algorithm(game)
    
    for idx_episode_training in range(n_episodes_training):
        
        # Initialize an episode
        state = game.new_initial_state()
        assert state.is_simultaneous_node(), "The game should be simultaneous"
        
        # Choose a joint action
        joint_action, probs = algo.choose_joint_action(game_state=state)
        
        # Play the joint action and get the rewards
        state.apply_actions(joint_action)
        assert state.is_terminal(), "The game should be over"
        rewards = state.returns()
        
        # Learn from the experience
        algo.learn(
            game_state=state, 
            joint_action=joint_action, 
            probs=probs, 
            rewards=rewards,
            )
        
        # Check if we should stop learning
        if algo.do_stop_learning():
            break
        

if __name__ == "__main__":
    main()