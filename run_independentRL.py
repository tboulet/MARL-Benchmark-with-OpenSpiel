# Utils import
from typing import Any, Callable, Dict, List, TypeVar
import numpy as np

from tqdm import tqdm
import wandb

from hydra import main
from omegaconf import DictConfig, OmegaConf



# Algorithms
from open_spiel.python import rl_agent

from algorithms.random import Random
from open_spiel.python.pytorch.dqn import DQN
from algorithms.ppo_adapted import PPO_Adapted
from algorithms.a2c_adapted import PolicyGradient_Adapted as A2C_Adapted
    
algo_name_to_algo_class : Dict[str, TypeVar("rl_agent.AbstractAgent")] = {
    "random" : Random,
    "dqn" : DQN,
    "ppo" : PPO_Adapted,
    "a2c" : A2C_Adapted,
}



# Metrics
from core.metrics import Metric, VsRandomMetric, VsHumanMetric
metric_name_to_metric_class : Dict[str, TypeVar("core.metrics.Metric")] = {
    "vs_random" : VsRandomMetric,
    "vs_human" : VsHumanMetric,
}



@main(version_base=None, config_path="configs", config_name="independentRL_default")
def main(config: DictConfig):
    
    # Initialize training constants
    config = OmegaConf.to_container(config, resolve=True)
    verbose : int = config["verbose"]
    do_wandb : bool = config["do_wandb"]
    seed : int = config["seed"]
    n_episodes_training : int = config["n_episodes_training"]
    n_episodes_evaluation : int = config["n_episodes_evaluation"]
    
    
    
    
    
    # Create the environment
    from open_spiel.python import rl_environment
    
    game_name = config["env"]["name"]
    env = rl_environment.Environment(game_name)

    n_players = env.num_players
    n_actions = env.action_spec()["num_actions"]
    n_observation_size = env.observation_spec()["info_state"][0]
    max_game_length = env.max_game_length
    min_utility = env.game.min_utility()
    max_utility = env.game.max_utility()

    if verbose >= 1:
        print(f"Number of players: {n_players}")
        print(f"Number of actions : {n_actions}")
        print(f"Observation size : {n_observation_size}")
        print(f"Max game length: {max_game_length}")
        print(f"Reward in [{min_utility}, {max_utility}]")



    # Create the agents
    algo_name = config["algo"]["name"]
    algo_class = algo_name_to_algo_class[algo_name]
    agents : List[rl_agent.AbstractAgent] = [
        algo_class(
            player_id = player_id,
            state_representation_size = n_observation_size,
            num_actions = n_actions,
            **config["algo"]["config"],
            # device = config.device,  # doesnt work yet
            )
        for player_id in range(n_players)
    ]
    
    
    
    # Initialize WandB
    if do_wandb:
        run_name = f"[{algo_name}]_[{game_name}]_{np.random.randint(0, 1000)}"
        run = wandb.init(
            project="IndependentRL Benchmark", 
            config=config,
            name=run_name,
            )
    
    
    
    # Initialize the metrics objects
    metric_name_to_metric = {}
    for metric_name, metric_config in config["metrics"].items():
        metric_class = metric_name_to_metric_class[metric_name]
        metric_name_to_metric[metric_name] = metric_class(config = config, training_env = env, **metric_config)
        
    
        
    # Train the agents in self-play.
    print("Training...")
    for episode_idx in tqdm(range(n_episodes_training)):
        # Evaluate the agents
        for metric_name in metric_name_to_metric:
            metric : Metric = metric_name_to_metric[metric_name]
            metric.evaluate(agents = agents, episode_idx = episode_idx)
            
        # Train for one episode
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_playing = agents[player_id]
            agent_output = agent_playing.step(time_step)
            time_step = env.step([agent_output.action])
        for agent in agents:
            agent.step(time_step) # Episode is over, step all agents with final info state.
            
    print("End of the run.")
    
    
if __name__ == "__main__":
    main()