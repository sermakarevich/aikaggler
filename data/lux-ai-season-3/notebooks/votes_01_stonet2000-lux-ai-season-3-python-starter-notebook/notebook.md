# Lux AI Season 3 Python Starter Notebook

- **Author:** Stone Tao
- **Votes:** 548
- **Ref:** stonet2000/lux-ai-season-3-python-starter-notebook
- **URL:** https://www.kaggle.com/code/stonet2000/lux-ai-season-3-python-starter-notebook
- **Last run:** 2024-12-26 20:00:17.247000

---

# Lux AI Season 3 @NeurIPS'24 Tutorial - Python Kit

Welcome to Season 3!

This notebook is the basic setup to use Jupyter Notebooks and the kaggle-environments package to develop your bot. If you plan to not use Jupyter Notebooks or any other programming language, please see our Github. The following are some important links!

Competition Page: https://www.kaggle.com/competitions/lux-ai-season-3/

Online Visualizer: https://s3vis.lux-ai.org/

Specifications: https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/docs/specs.md

Github: https://github.com/Lux-AI-Challenge/Lux-Design-S3

Bot API: https://github.com/Lux-AI-Challenge/Lux-Design-S3/tree/main/kits

And if you haven't done so already, we highly recommend you join our Discord server at https://discord.gg/aWJt3UAcgn or at the minimum follow the kaggle forums at https://www.kaggle.com/c/lux-ai-season-3/discussion. We post important announcements there such as changes to rules, events, and opportunities from our sponsors!

Now let's get started!

## Prerequisites
We assume that you have a basic knowledge of Python and programming. It's okay if you don't know the game specifications yet! Feel free to always refer back to our [docs on Github](https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/docs/specs.md)

## Basic Setup
First thing to verify is that you have python 3.9 or above and have the [luxai_s3](https://pypi.org/project/luxai_s3/) package installed. Run the command below to do so. If you are using Kaggle Notebooks, **make sure to also click run-> restart and clear cell outputs** on the top right next to view and add-ons. (This fixes a bug where Kaggle Notebooks loads an incompatible package)

```python
# verify version
!python --version
!pip install --upgrade luxai-s3
!mkdir agent && cp -r ../input/lux-ai-season-3/* agent/
import sys
sys.path.insert(1, 'agent')
```

Then run the following command line code to verify your installation works. It will use the current starter kit agent and run one game and generate a replay file in the form of an openable HTML file. If you are using kaggle notebooks sometimes the html won't load, in which case you can download the HTML file and open it locally. The HTML viewer shown inline here can be used to upload json formatted replays by clicking the Home button or you can upload them to https://s3vis.lux-ai.org/.

```python
!luxai-s3 agent/main.py agent/main.py --output=replay.html
```

```python
import IPython # load the HTML replay
IPython.display.HTML(filename='replay.html')
```

## Building an Agent
Now we know what the environment looks like, let's try building a working agent. The goal of this environment is to win a best of 5 game, where each match in the 5-match sequence is won by who has the most relic points.

In our kit we provide a skeleton for building an agent. Avoid removing any function from the kit unless you know what you are doing as it may cause your agent to fail on the competition servers. This agent defintion should be stored in the `agent.py` file.

The agent will have `self.player, self.opp_player, self.env_cfg` populated with the correct values at each step of an environment during competition or when you use the CLI tool to run matches. 

`self.env_cfg` stores the curent environment's configurations, and `self.player, self.opp_player` stores the name of your player/team and the opposition respectively (will always be "player_0" or "player_1").

The Agent class below is a minimal code sample of something that takes zero actions (units do nothing).

```python
import numpy as np
class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit. 
        
        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        unit_mask = np.array(obs["units_mask"][self.team_id]) # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id]) # shape (max_units, 2)
        unit_energys = np.array(obs["units"]["energy"][self.team_id]) # shape (max_units, 1)
        observed_relic_node_positions = np.array(obs["relic_nodes"]) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )
        team_points = np.array(obs["team_points"]) # points of each team, team_points[self.team_id] is the points of the your team
        # ids of units you can control at this timestep
        available_unit_ids = np.where(unit_mask)[0]
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        return actions
```

The next code snippet provides a simple agent evaluation function that creates an initial environment and provides an initial seed. Then it runs games and automatically saves replays to the `replays` folder as it goes. The replays are auto saved after each environment reset or when the environment is closed and can be watched by uploading them to https://s3vis.lux-ai.org/

```python
import json
from IPython.display import display, Javascript
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

def render_episode(episode: RecordEpisode) -> None:
    data = json.dumps(episode.serialize_episode_data(), separators=(",", ":"))
    display(Javascript(f"""
var iframe = document.createElement('iframe');
iframe.src = 'https://s3vis.lux-ai.org/#/kaggle';
iframe.width = '100%';
iframe.scrolling = 'no';

iframe.addEventListener('load', event => {{
    event.target.contentWindow.postMessage({data}, 'https://s3vis.lux-ai.org');
}});

new ResizeObserver(entries => {{
    for (const entry of entries) {{
        entry.target.height = `${{Math.round(320 + 0.3 * entry.contentRect.width)}}px`;
    }}
}}).observe(iframe);

element.append(iframe);
    """))

def evaluate_agents(agent_1_cls, agent_2_cls, seed=42, games_to_play=3, replay_save_dir="replays"):
    env = RecordEpisode(
        LuxAIS3GymEnv(numpy_output=True), save_on_close=True, save_on_reset=True, save_dir=replay_save_dir
    )
    obs, info = env.reset(seed=seed)
    for i in range(games_to_play):
        obs, info = env.reset()
        env_cfg = info["params"] # only contains observable game parameters
        player_0 = agent_1_cls("player_0", env_cfg)
        player_1 = agent_2_cls("player_1", env_cfg)
    
        # main game loop
        game_done = False
        step = 0
        print(f"Running game {i}")
        while not game_done:
            actions = dict()
            for agent in [player_0, player_1]:
                actions[agent.player] = agent.act(step=step, obs=obs[agent.player])
            obs, reward, terminated, truncated, info = env.step(actions)
            # info["state"] is the environment state object, you can inspect/play around with it to e.g. print
            # unobservable game data that agents can't see
            dones = {k: terminated[k] | truncated[k] for k in terminated}
            if dones["player_0"] or dones["player_1"]:
                game_done = True
            step += 1
        render_episode(env)
    env.close() # free up resources and save final replay
```

```python
evaluate_agents(Agent, Agent) # here we evaluate our dummy agent against itself, it will auto render in the notebook
!ls replays # see what replays we have
```

Now that we have some agents that can be run, lets add some logic to the agent so that it can go and explore. We will use a basic strategy of sampling a random location on the game map, and having the unit move there until it reaches that location.

```python
from lux.utils import direction_to
import numpy as np
class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg
        
        self.unit_explore_locations = dict()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit. 
        
        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        unit_mask = np.array(obs["units_mask"][self.team_id]) # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id]) # shape (max_units, 2)
        unit_energys = np.array(obs["units"]["energy"][self.team_id]) # shape (max_units, 1)
        observed_relic_node_positions = np.array(obs["relic_nodes"]) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )
        team_points = np.array(obs["team_points"]) # points of each team, team_points[self.team_id] is the points of the your team
        # ids of units you can control at this timestep
        available_unit_ids = np.where(unit_mask)[0]
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        # unit ids range from 0 to max_units - 1
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            # every 20 steps or if a unit doesn't have an assigned location to explore
            if step % 20 == 0 or unit_id not in self.unit_explore_locations:
                # pick a random location on the map for the unit to explore
                rand_loc = (np.random.randint(0, self.env_cfg["map_width"]), np.random.randint(0, self.env_cfg["map_height"]))
                self.unit_explore_locations[unit_id] = rand_loc
            # using the direction_to tool we can generate a direction that makes the unit move to the saved location
            # note that the first index of each unit's action represents the type of action. See specs for more details
            actions[unit_id] = [direction_to(unit_pos, self.unit_explore_locations[unit_id]), 0, 0]
        return actions
```

```python
evaluate_agents(Agent, Agent)
```

Great now that we have randomly moving units, we now want to move to locations that give our team points, namely relic nodes. Recall that in this season's game, you can only see what your units see, so information about the dynamically changing map is crucical. For starters we will write some code to track every relic node location we find as we need to go there to get points. Because each game has 5 matches and the map is preserved in between matches, we can save map information between matches to improve our agent's gameplay. In the example code below, we implement a very basic strategy to leverage this information by simply making all units move towards a relic node. Since to gain points a unit must move on top of a hidden tile near the relic node (in a 5x5 square centered at the node) we add code to make units move randomly around relic nodes if they are close enough.

```python
from lux.utils import direction_to
import numpy as np
class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg
        
        self.unit_explore_locations = dict()
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit. 
        
        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        unit_mask = np.array(obs["units_mask"][self.team_id]) # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id]) # shape (max_units, 2)
        unit_energys = np.array(obs["units"]["energy"][self.team_id]) # shape (max_units, 1)
        observed_relic_node_positions = np.array(obs["relic_nodes"]) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )
        team_points = np.array(obs["team_points"]) # points of each team, team_points[self.team_id] is the points of the your team
        # ids of units you can control at this timestep
        available_unit_ids = np.where(unit_mask)[0]
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)


        # visible relic nodes
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
        # save any new relic nodes that we discover for the rest of the game.
        for id in visible_relic_node_ids:
            if id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(id)
                self.relic_node_positions.append(observed_relic_node_positions[id])

        # unit ids range from 0 to max_units - 1
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            # if we found at least one relic node
            if len(self.relic_node_positions) > 0:
                nearest_relic_node_position = self.relic_node_positions[0]
                manhattan_distance = abs(unit_pos[0] - nearest_relic_node_position[0]) + abs(unit_pos[1] - nearest_relic_node_position[1])
                
                # if close to the relic node we want to move randomly around it and hope to gain points
                if manhattan_distance <= 4:
                    random_direction = np.random.randint(0, 5)
                    actions[unit_id] = [random_direction, 0, 0]
                else:
                    # otherwise we want to move towards the relic node
                    actions[unit_id] = [direction_to(unit_pos, nearest_relic_node_position), 0, 0]
            # every 20 steps or if a unit doesn't have an assigned location to explore
            else:
                if step % 20 == 0 or unit_id not in self.unit_explore_locations:
                    # pick a random location on the map for the unit to explore
                    rand_loc = (np.random.randint(0, self.env_cfg["map_width"]), np.random.randint(0, self.env_cfg["map_height"]))
                    self.unit_explore_locations[unit_id] = rand_loc
                # using the direction_to tool we can generate a direction that makes the unit move to the saved location
                # note that the first index of each unit's action represents the type of action. See specs for more details
                actions[unit_id] = [direction_to(unit_pos, self.unit_explore_locations[unit_id]), 0, 0]
        return actions
```

```python
evaluate_agents(Agent, Agent)
```

## Create a submission
Now we need to create a .tar.gz file with main.py (and agent.py) at the top level. We can then upload this. In the notebook you can write to a local file with the %%writefile command, just copy your agent code below:

```python
%%writefile agent/agent.py
from lux.utils import direction_to
import numpy as np
class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg
        
        self.unit_explore_locations = dict()
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit. 
        
        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        unit_mask = np.array(obs["units_mask"][self.team_id]) # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id]) # shape (max_units, 2)
        unit_energys = np.array(obs["units"]["energy"][self.team_id]) # shape (max_units, 1)
        observed_relic_node_positions = np.array(obs["relic_nodes"]) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )
        team_points = np.array(obs["team_points"]) # points of each team, team_points[self.team_id] is the points of the your team
        # ids of units you can control at this timestep
        available_unit_ids = np.where(unit_mask)[0]
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)


        # visible relic nodes
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
        # save any new relic nodes that we discover for the rest of the game.
        for id in visible_relic_node_ids:
            if id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(id)
                self.relic_node_positions.append(observed_relic_node_positions[id])

        # unit ids range from 0 to max_units - 1
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            # if we found at least one relic node
            if len(self.relic_node_positions) > 0:
                nearest_relic_node_position = self.relic_node_positions[0]
                manhattan_distance = abs(unit_pos[0] - nearest_relic_node_position[0]) + abs(unit_pos[1] - nearest_relic_node_position[1])
                
                # if close to the relic node we want to move randomly around it and hope to gain points
                if manhattan_distance <= 4:
                    random_direction = np.random.randint(0, 5)
                    actions[unit_id] = [random_direction, 0, 0]
                else:
                    # otherwise we want to move towards the relic node
                    actions[unit_id] = [direction_to(unit_pos, nearest_relic_node_position), 0, 0]
            # every 20 steps or if a unit doesn't have an assigned location to explore
            else:
                if step % 20 == 0 or unit_id not in self.unit_explore_locations:
                    # pick a random location on the map for the unit to explore
                    rand_loc = (np.random.randint(0, self.env_cfg["map_width"]), np.random.randint(0, self.env_cfg["map_height"]))
                    self.unit_explore_locations[unit_id] = rand_loc
                # using the direction_to tool we can generate a direction that makes the unit move to the saved location
                # note that the first index of each unit's action represents the type of action. See specs for more details
                actions[unit_id] = [direction_to(unit_pos, self.unit_explore_locations[unit_id]), 0, 0]
        return actions
```

```python
!cd agent && tar -czf submission.tar.gz *
!mv agent/submission.tar.gz .
```

## Submit
Now open the /kaggle/working folder and find submission.tar.gz, download that file, navigate to the "MySubmissions" tab in https://www.kaggle.com/competitions/lux-ai-season-3/submissions and upload your submission! It should play a validation match against itself and once it succeeds it will be automatically matched against other players' submissions. Newer submissions will be prioritized for games over older ones. Your team is limited in the number of successful submissions per day so we highly recommend testing your bot locally before submitting.

## CLI Tool

To test your agent without using the python API you can also run

```python
!luxai-s3 agent/main.py agent/main.py --seed 101 -o replay.html
```

which uses a seed of 101 and generates a replay.html file that you can click and watch. Optionally if you specify `-o replay.json` you can upload replay.json to http://s3vis.lux-ai.org/. We **highly recommend** watching on a separate window instead of watching here on a notebook as the notebook screen width is quite small.

The CLI tool enables you to easily run episodes between any two agents (python or not) and provides a flexible tournament running tool to evaluate many agents together. Documentation on this tool can be found here: https://github.com/Lux-AI-Challenge/Lux-Design-S3/tree/main/luxai_runner/README.md