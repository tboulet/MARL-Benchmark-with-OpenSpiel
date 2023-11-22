# Utils import
from typing import List
from tqdm import tqdm


# Import the environment
from open_spiel.python import rl_environment

tic_tac_toe_string = "tic_tac_toe"
connect_four_string = "connect_four"
env = rl_environment.Environment(connect_four_string)

print(f"Number of players: {env.num_players}")
print(f"Number of actions : {env.action_spec()['num_actions']}")
print(f"Observation spec : {env.observation_spec()}")
print(f"Size of the observation space : {env.observation_spec()['info_state']}")
print(f"Max game length: {env.max_game_length}")
print(f"Reward in [{env.game.min_utility()}, {env.game.max_utility()}]")


# Define the agents
from _ALGOS.dqn_cuda import DQN_Cuda

    
    
        
agents : List[DQN_Cuda] = [
    DQN_Cuda(
        player_id = player_id,
        state_representation_size = env.observation_spec()['info_state'][0],
        num_actions = env.action_spec()['num_actions'],
        device = "cuda",
        batch_size = 1024,
        )
    for player_id in range(env.num_players)
]

# Display agent device
print(next(agents[0]._q_network.parameters()).device)

# Train the Q-learning agents in self-play.
print("Training...")
for cur_episode in tqdm(range(10000)):
  time_step = env.reset()
  while not time_step.last():
    player_id = time_step.observations["current_player"]
    agent_output = agents[player_id].step(time_step)
    time_step = env.step([agent_output.action])
  # Episode is over, step all agents with final info state.
  for agent in agents:
    agent.step(time_step)
print("Done!")


# Evaluation phase
print("Evaluation...")
from open_spiel.python.algorithms import random_agent

agents_to_eval = [agents[0], random_agent.RandomAgent(
    player_id=1, num_actions=env.action_spec()["num_actions"])]
agents_to_eval[0].eval_mode = True

list_final_rewards = []
for _ in tqdm(range(10000)):
    time_step = env.reset()
    while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = agents_to_eval[player_id].step(time_step, is_evaluation=True)
        # print(f"Player {player_id} chose action {agent_output.action}")
        time_step = env.step([agent_output.action])
        
    list_final_rewards.append(time_step.rewards[0])

import numpy as np
print(f"Average reward over {len(list_final_rewards)} episodes: {sum(list_final_rewards)/len(list_final_rewards)} +/- {np.std(list_final_rewards)}")