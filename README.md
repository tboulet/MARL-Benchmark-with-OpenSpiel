# MARL-Benchmark-with-OpenSpiel
A research project for MARL and Game Theory algorithms, using the DeepMind framework OpenSpiel.

The codebase is in Python, in particular we use Pytorch as our deep learning framework, Hydra as our configuration system and WandB and tensorboard for logging metrics.



# Goals

Currently, we have the following goals for this project :
- IndependentRL Benchmark : Implement a benchmark for IndependentRL algorithms (e.g. Q-Learning, DQN, etc.) on OpenSpiel games, and compare the different algorithms by having them play against each other. This was made as a course project for the [Deep Learning](https://www.master-mva.com/cours/cat-deep-learning/) course at the master MVA (Ens Paris-Saclay).
- Algorithms for Normal Form Games : Implement algorithms for normal form games (e.g. fictitious play, etc.) and compare them on OpenSpiel games.


# Installation

This is the method for installing on WSL. For other systems, this may vary but for installing OpenSpiel, the OpenSpiel repo may help.

### Clone this repo at the desired location

```bash
git clone git@github.com:tboulet/MARL-Benchmark-with-OpenSpiel.git
cd MARL-Benchmark-with-OpenSpiel
```

### Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install OpenSpiel from source

Install the OpenSpiel package from source as a sub repo. This is adapted from the WSL installation tutorial of OpenSpiel that you can find [here](https://github.com/google-deepmind/open_spiel/blob/master/docs/windows.md#option-2-windows-installation-using-windows-subsystem-for-linux-wsl).
```bash
git clone https://github.com/deepmind/open_spiel.git
cd open_spiel
./install.sh # you will be prompted for the password created at stage 3. Press Y to continue and install. During installation press Yes to restart services during package upgrades
pip install -U pip # Upgrade pip (required for TF >= 1.15)
pip3 install --upgrade -r requirements.txt # Install Python dependencies

mkdir build
cd build
CXX=clang++ cmake -DPython3_EXECUTABLE=$(which python3) -DCMAKE_CXX_COMPILER=clang++ ../open_spiel
make -j12 # The 12 here is the number of parallel processes used to build
ctest -j12 # Run the tests to verify that the installation succeeded
cd ..
```

One can run an example of a game running :

```bash
build/examples/example --game=tic_tac_toe
```

### Setting your PYTHONPATH environment variable :

We suggest including those lines at the end of your `venv/bin/activate` file :

```shell
# Add the new path
export OLD_PYTHONPATH=$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:<your repo path>/open_spiel
export PYTHONPATH=$PYTHONPATH:<your repo path>/open_spiel/build/python
```
Replace your repo path with the path to the folder `MARL-Benchmark-with-OpenSpiel`.

To check that it works, this command should return "`No module named open_spiel.__main__; 'open_spiel' is a package and cannot be directly executed`" :

```bash
python -m open_spiel
```

### Go back to the root of the repo and install the requirements

```
cd ..
pip install -r requirements.txt
```

### Install Jax
This step may be unecessary for running non-Jax algorithms. If you want to run Jax algorithms, you need to install Jax with CUDA support. For this, you can run the following command :
```
pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Set up WandB

If you want to use WandB for logging metrics, follow the [WandB quickstart](https://docs.wandb.ai/quickstart) and put `do_wandb : True` in the config file.


### Have Pylance detect OpenSpiel

Note : for Pylance to detect the open_spiel package, you can add ./open_spiel to your `./.vscode/settings.json` worskpace settings file :

```json
{
    "python.analysis.extraPaths": [
        "./open_spiel"
    ]
}
```

### Some OpenSpiel bugs and how to solve them
- For using the exploitability metric, you need to modify in ``open_spiel/open_spiel/python/algorithms/exploitability.py`` line 200 the ``on_policy_values`` into an array through ``on_policy_values = np.array(on_policy_values)``.



# Run the code
 
For training your algorithms on a IndependentRL settings and in parallel (e.g. `dqn`, `ppo`, `a2c`) on a certain environment/game (e.g. `tic_tac_toe`, `connect_four`), run the command :

```bash
python run_independentRL.py env=tic_tac_toe
```

The agents tag should correspond to a configuration in ``configs_independentRL/agents/`` where you can specify the group of agents. Each group of agents is trained separately of the other groups of agents, in an IndependentRL settings.

We use Hydra as our config system. The config folder is `./configs_independentRL`. You can modify the config (logging, metrics, number of training episodes) from the `independentRL_default.yaml` file. The available algorithms are in the `algos/algo` sub-folder and the available environments are in the `env` sub-folder.

You can specify the grouped agents configuration, the environment (game), the metrics (winrate against random, faceoff, exploitability...) and the logging (WandB, tensorboard) and other training parameters in the config file. We advise you to not use the exploitability metric on Connect4, as it is extremely slow to compute.

## Visualize the results

You can access the results both on tensorboard :

```bash
tensorboard --logdir ./tensorboard
```

And on your WandB account, where a project named "IndependentRL Benchmark" should have been created. 

Alternatively, you can also access our results on this Wandb project : https://wandb.ai/tboulet/IndependentRL%20Benchmark. We advise you to log to your own WandB account which will allow you to group wandb runs by ``env.name`` and then display only one game at a time in order to compare metrics game by game.