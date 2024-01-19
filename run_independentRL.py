# Utils import
import datetime
from typing import Any, Callable, Dict, List, Type
from matplotlib.pylab import f
import numpy as np

from tqdm import tqdm

from hydra import main
from omegaconf import DictConfig, OmegaConf
import wandb



# Algorithms
from open_spiel.python import rl_agent
from algorithms import algo_name_to_algo_class

# Metrics
from metrics import Metric, metric_name_to_metric_class

# Env
from open_spiel.python import rl_environment




class IndependentRL_Algorithm:
    """A class to run a training of algorithms on a given game in an IndependentRL settings. 
    Specifically :
    - Parallel IndependentRL : we train k = self.n_algorithms group of agents in an Independent RL settings each
    - Intra-algorithm training : in each group, we have n = n_players agents, each following the same algorithm, that fight each other
    - Inter-algorithm evaluation : regularly, at evaluation phase, one of our metric is the inter-evaluation evaluation, i.e. our agents are pitted each against each other 
    - Other evaluation metric : vs random evaluation, exploitability
    """    

    def __init__(self, config : DictConfig) -> None:
        self.config = config
                
        
    def run(self):
        """Run the training."""
        
        
        # Initialize training parameters
        self.verbose : int = self.config["verbose"]
        self.tqdm_bar : bool = self.config["tqdm_bar"]
        self.do_wandb : bool = self.config["do_wandb"]
        self.do_tb : bool = self.config["do_tb"]
        self.do_cli : bool = self.config["do_cli"]
        self.seed : int = self.config["seed"] if self.config["seed"] is not None else np.random.randint(0, 1000)
        self.n_episodes_training : int = self.config["n_episodes_training"]
        
        self.group_names_to_grouped_agents_names : Dict[str, List[str]] = self.config["agents"]["group_names_to_grouped_agents_names"]
        self.n_groups : int = len(self.group_names_to_grouped_agents_names)
        self.agents_grouped_repr : str = self.config["agents"]["name"]
        
        # Create the k environments, k being the number of algorithms
        self.game_name = self.config["env"]["name"]
        self.game_config = self.config["env"]["config"]
        assert self.n_groups > 0, "No algorithm specified in the config file."
        group_names_to_envs = {group_name : rl_environment.Environment(self.game_name, **self.game_config) for group_name in self.group_names_to_grouped_agents_names}
        env = group_names_to_envs[list(group_names_to_envs.keys())[0]]
        
        self.n_players = env.num_players
        self.n_actions = env.action_spec()["num_actions"]
        self.n_observation_size = env.observation_spec()["info_state"][0]
        self.max_game_length = env.max_game_length
        self.min_utility = env.game.min_utility()
        self.max_utility = env.game.max_utility()

        if self.verbose >= 1:
            print(f"\nGame name : {self.game_name}")
            print(f"Number of algorithms (group of player with the same algo training together) : {self.n_groups}")
            print(f"Number of players: {self.n_players}")
            print(f"Number of actions : {self.n_actions}")
            print(f"Observation size : {self.n_observation_size}")
            print(f"Max game length: {self.max_game_length}")
            print(f"Reward in [{self.min_utility}, {self.max_utility}]")


        # Create the agents
        self.group_names_to_grouped_agents : Dict[str, List[rl_agent.AbstractAgent]] = {}
        for group_name, algo_names_list in self.group_names_to_grouped_agents_names.items():
            grouped_agents : List[rl_agent.AbstractAgent] = [
                algo_name_to_algo_class[algo_name](
                    player_id = player_id,
                    state_representation_size = self.n_observation_size,
                    num_actions = self.n_actions,
                    # **algo_config # TODO : add the possibility to specify algo config in the config file
                    )
                for player_id, algo_name in enumerate(algo_names_list)
            ]
            self.group_names_to_grouped_agents[group_name] = grouped_agents
        
        
        # Initialize loggers
        self.initialize_loggers()
        self.config["game_theory_algorithm"] = self
        
        
        # Initialize the metrics objects
        self.metric_name_to_metric = {}
        for metric_name, metric_dict in self.config["metric_list"]["metric"].items():
            metric_class = metric_name_to_metric_class[metric_name]
            self.metric_name_to_metric[metric_name] = metric_class(config = self.config, **metric_dict["config"])
        time_training_cumulative = 0
        time_evaluation_cumulative = 0
        
        # Train the agents in self-play.
        print(f"Training {self.agents_grouped_repr} agents in parallel IndependentRL on {self.game_name} for {self.n_episodes_training} episodes...")
        iterator = tqdm(range(self.n_episodes_training)) if self.tqdm_bar else range(self.n_episodes_training)
        for episode_idx in iterator:
            # Evaluate at each episode
            t0 = datetime.datetime.now()
            self.evaluate_grouped_agents(
                group_name_to_grouped_agents = self.group_names_to_grouped_agents,
                envs = group_names_to_envs,
                episode_idx = episode_idx,
                )
            time_evaluation_cumulative += (datetime.datetime.now() - t0).total_seconds()
                    
            # Train each group of algorithms in their corressponding environment
            t0 = datetime.datetime.now()
            self.train_grouped_agents_one_episode(
                group_names_to_grouped_agents = self.group_names_to_grouped_agents, 
                group_names_to_envs = group_names_to_envs,
                episode_idx = episode_idx,
                )
            time_training_cumulative += (datetime.datetime.now() - t0).total_seconds()
            
            # Log time metrics
            time_iteration_cumulative = time_training_cumulative + time_evaluation_cumulative
            training_proportion = time_training_cumulative / time_iteration_cumulative
            evaluation_proportion = time_evaluation_cumulative / time_iteration_cumulative
            time_metrics_dict = {
                "time/training cumulative" : time_training_cumulative,
                "time/evaluation cumulative" : time_evaluation_cumulative,
                "time/training proportion" : training_proportion,
                "time/evaluation proportion" : evaluation_proportion,
                }
            self.log_metrics(time_metrics_dict, episode_idx)
            
        # Do one step of evaluation at the end
        self.evaluate_grouped_agents(
            group_name_to_grouped_agents = self.group_names_to_grouped_agents,
            envs = group_names_to_envs,
            episode_idx = self.n_episodes_training,
            )
        
        # End the run        
        self.end_run()
    
    
    
    def train_grouped_agents_one_episode(self,
            group_names_to_grouped_agents : Dict[str, List[rl_agent.AbstractAgent]],
            group_names_to_envs : Dict[str, rl_environment.Environment],
            episode_idx : int,
    ) -> None:
        """Train each group of agents in their corressponding environment for one episode.

        Complexity : O(L * (K * C_one_step_env + sum_k(C_one_step_agent_k)))
        
        Args:
            group_names_to_grouped_agents (Dict[str, List[rl_agent.AbstractAgent]]): the agents to train, grouped by algorithm
            group_names_to_envs (Dict[str, rl_environment.Environment]): the environments to train the agents in
            episode_idx (int): the episode index
        """
        for group_name, grouped_agents in group_names_to_grouped_agents.items():
            env = group_names_to_envs[group_name]
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_playing = grouped_agents[player_id]
                agent_output = agent_playing.step(time_step)
                time_step = env.step([agent_output.action])
            for agent in grouped_agents:
                agent.step(time_step)
                    
                  
                    
    def evaluate_grouped_agents(self, 
            group_name_to_grouped_agents : Dict[str, List[rl_agent.AbstractAgent]],
            envs : Dict[str, rl_environment.Environment],
            episode_idx : int,
            ) -> None:
        """Evaluate the grouped agents using the metrics and log the results.

        Complexity : sum over all metrics j of O(C_metric_j)
        Args:
            group_name_to_grouped_agents : Dict[str, List[rl_agent.AbstractAgent]]: the agents to evaluate, grouped by algorithm
            envs (Dict[str, rl_environment.Environment]): the environments to evaluate the agents in
            episode_idx (int): the episode index
        """
        for metric_name in self.metric_name_to_metric:
            t0 = datetime.datetime.now()
            metric : Metric = self.metric_name_to_metric[metric_name]
            metrics_dict = metric.evaluate(
                group_names_to_grouped_agents = group_name_to_grouped_agents,
                group_names_to_envs = envs,
                episode_idx = episode_idx,
                )
            metrics_dict[f"time/compute metric '{metric_name}'"] = (datetime.datetime.now() - t0).total_seconds()
            self.log_metrics(metrics_dict, episode_idx)
            
                        
    
    def log_metrics(self, metrics_dict : Dict[str, float], episode_idx : int) -> None:
        """Log a dictionary of metrics to the different loggers.

        Args:
            metrics_dict (Dict[str, float]): the metrics to log, as a dictionary with the metric name as key and the metric value as value
            episode_idx (int): the training episode index
        """
        if self.do_wandb:
                wandb.log(metrics_dict, step = episode_idx)
        if self.do_tb:
            for metric_name in metrics_dict:
                self.tb_writer.add_scalar(f"{metric_name}", metrics_dict[metric_name], global_step = episode_idx)
        if self.do_cli:
            if len(metrics_dict) > 0:
                print(f"Metrics at episode {episode_idx} : {metrics_dict}")
                    
                    
                    
    def initialize_loggers(self):
        """Initialize the loggers (e.g. WandB, Tensorboard)
        """
        self.run_name = f"inRL_[{self.agents_grouped_repr}]_[{self.game_name}]_{datetime.datetime.now().strftime('%dth%mmo_%Hh%Mmin%Ss')}_seed{self.seed}"
        print(f"\nRun name : {self.run_name}")
        if self.do_wandb:
            import wandb
            self.wandb_run = wandb.init(
                project="IndependentRL Benchmark", 
                config=self.config,
                name=self.run_name,
                )
        if self.do_tb:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir = f"tensorboard/{self.run_name}")
            
                  
                   
    def end_run(self):
        """End the run, including for example by closing the loggers.
        """
        print("End of the run.")
        if self.do_wandb:
            self.wandb_run.finish()
        if self.do_tb:
            self.tb_writer.close()
                        
                        
                        
                        
@main(version_base=None, config_path="configs_independentRL", config_name="independentRL_default.yaml")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    config = OmegaConf.to_container(config, resolve=True)
    game_theory_algorithm = IndependentRL_Algorithm(config = config)
    game_theory_algorithm.run()



if __name__ == "__main__":
    main()