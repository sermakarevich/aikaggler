# NuralBrain v0.5.1 model | Train And Win

- **Author:** Sangram Patil
- **Votes:** 134
- **Ref:** sangrampatil5150/nuralbrain-v0-5-1-model-train-and-win
- **URL:** https://www.kaggle.com/code/sangrampatil5150/nuralbrain-v0-5-1-model-train-and-win
- **Last run:** 2025-02-18 12:39:18.197000

---

```python
# verify version
!python --version
!pip install --upgrade luxai-s3
!mkdir agent && cp -r ../input/lux-ai-season-3/* agent/
import sys
sys.path.insert(1, 'agent')
```

```python
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1  # Used for beta annealing

    def push(self, state, action, reward, next_state, done):
        # New experience: Max priority
        if self.priorities:
            max_priority = np.max([np.max(p) if isinstance(p, np.ndarray) else p for p in self.priorities])
        else:
            max_priority = 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size):
        # Calculate beta (annealing)
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        # Calculate probabilities
        priorities = np.array([np.max(p) if isinstance(p, np.ndarray) else p for p in self.priorities], dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize
        weights = torch.FloatTensor(weights).to(samples[0][0].device) # Move to device

        return samples, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha # Add small constant

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_channels, output_size, local_view_size, hidden_size=512):
        super(DQN, self).__init__()
        self.local_view_size = local_view_size

        # Convolutional Layers (Deeper and Wider) 
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate Conv Output Size 
        self._conv_output_size = self._get_conv_output_size(input_channels)

        # Connected Layers 
        self.fc1 = nn.Linear(self._conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def _get_conv_output_size(self, input_channels):
        _input = torch.zeros(1, input_channels, self.local_view_size, self.local_view_size)
        x = self.pool(F.relu(self.bn1(self.conv1(_input))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return int(np.prod(x.size()))

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

```python
import random
import numpy as np
import torch
import torch.optim as optim
#from DQN_ import DQN, ReplayBuffer  # Now properly imported
from enum import IntEnum
from scipy.signal import convolve2d
import copy
from collections import deque
from torch.optim.lr_scheduler import StepLR

SPACE_SIZE = 24  # Define as a constant

class Global:
    MAX_UNITS = 16
    RELIC_REWARD_RANGE = 5
    ALL_RELICS_FOUND = False
    ALL_REWARDS_FOUND = False
    LAST_MATCH_WHEN_RELIC_CAN_APPEAR = 10
    LAST_MATCH_STEP_WHEN_RELIC_CAN_APPEAR = 0
    REWARD_RESULTS = []
    OBSTACLES_MOVEMENT_STATUS = []
    OBSTACLE_MOVEMENT_DIRECTION_FOUND = False
    OBSTACLE_MOVEMENT_DIRECTION = (0, 0)
    OBSTACLE_MOVEMENT_PERIOD_FOUND = False
    OBSTACLE_MOVEMENT_PERIOD = 0

def get_opposite(x, y):
    """Returns the symmetrical opposite coordinates on the map."""
    return SPACE_SIZE - 1 - x, SPACE_SIZE - 1 - y

def warp_point(x, y):
    return x % SPACE_SIZE, y % SPACE_SIZE

def nearby_positions(x, y, radius):
    """Yields positions within a given radius of (x, y)."""
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx, ny = warp_point(x + dx, y + dy)
            yield nx, ny

def get_match_number(step):
    return step // 505

def get_match_step(step):
    return step % 505

def manhattan_distance(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

_DIRECTIONS = [
    (0, 0),  # center
    (0, -1),  # up
    (1, 0),  # right
    (0, 1),  # down
    (-1, 0),  # left
    (0, 0),  # sap
]

class ActionType(IntEnum):
    center = 0
    up = 1
    right = 2
    down = 3
    left = 4
    sap = 5

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @classmethod
    def from_coordinates(cls, current_position, next_position):
        dx = next_position[0] - current_position[0]
        dy = next_position[1] - current_position[1]
        if dx < 0:
            return ActionType.left
        elif dx > 0:
            return ActionType.right
        elif dy < 0:
            return ActionType.up
        elif dy > 0:
            return ActionType.down
        else:
            return ActionType.center

    def to_direction(self):
        return _DIRECTIONS[self]

class NodeType(IntEnum):
    unknown = -1
    empty = 0
    nebula = 1
    asteroid = 2

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = NodeType.unknown
        self.energy = None
        self.is_visible = False

        self._relic = False
        self._reward = False
        self._explored_for_relic = False
        self._explored_for_reward = False

    def __repr__(self):
        return f"Node({self.x}, {self.y}, {self.type})"

    def __hash__(self):
        return self.coordinates.__hash__()

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    @property
    def relic(self):
        return self._relic

    @property
    def reward(self):
        return self._reward

    @property
    def explored_for_relic(self):
        return self._explored_for_relic

    @property
    def explored_for_reward(self):
        return self._explored_for_reward

    def update_relic_status(self, status: None | bool):
        if self._explored_for_relic and self._relic and not status:
            raise ValueError(
                f"Can't change the relic status {self._relic}->{status} for {self}"
                ", the tile has already been explored"
            )
        if status is None:
            self._explored_for_relic = False
            return
        self._relic = status
        self._explored_for_relic = True

    def update_reward_status(self, status: None | bool):
        if self._explored_for_reward and self._reward and not status:
            self._explored_for_reward = False
        if status is None:
            self._explored_for_reward = False
            return
        self._reward = status
        self._explored_for_reward = True

    @property
    def is_unknown(self) -> bool:
        return self.type == NodeType.unknown

    @property
    def is_walkable(self) -> bool:
        return self.type != NodeType.asteroid

    @property
    def coordinates(self) -> tuple[int, int]:
        return self.x, self.y

    def manhattan_distance(self, other: "Node") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

class Space:
    def __init__(self, space_size):
        self.space_size = space_size
        self._nodes: list[list[Node]] = []
        for y in range(self.space_size):
            row = [Node(x, y) for x in range(self.space_size)]
            self._nodes.append(row)
        self._relic_nodes: set[Node] = set()
        self._reward_nodes: set[Node] = set()
        self.obstacle_history = deque(maxlen=100)
        self.best_direction = (0, 0)
        self.best_period = 40
        self.movement_counter = 0

    def __repr__(self) -> str:
        return f"Space({self.space_size}x{self.space_size})"

    def __iter__(self):
        for row in self._nodes:
            yield from row

    @property
    def relic_nodes(self) -> set[Node]:
        return self._relic_nodes

    @property
    def reward_nodes(self) -> set[Node]:
        return self._reward_nodes

    def get_node(self, x, y) -> Node:
        return self._nodes[y][x]

    def update(self, step, obs, team_id, team_reward):
        self.move_obstacles(step)
        self._update_map(obs)
        self._update_relic_map(step, obs, team_id, team_reward)

    def _update_relic_map(self, step, obs, team_id, team_reward):
        for mask, xy in zip(obs["relic_nodes_mask"], obs["relic_nodes"]):
            if mask and not self.get_node(*xy).relic:
                self._update_relic_status(*xy, status=True)
                for x, y in nearby_positions(*xy, Global.RELIC_REWARD_RANGE):
                    if not self.get_node(x, y).reward:
                        self._update_reward_status(x, y, status=None)
        all_relics_found = True
        all_rewards_found = True
        for node in self:
            if node.is_visible and not node.explored_for_relic:
                self._update_relic_status(*node.coordinates, status=False)
            if not node.explored_for_relic:
                all_relics_found = False
            if not node.explored_for_reward:
                all_rewards_found = False
        Global.ALL_RELICS_FOUND = all_relics_found
        Global.ALL_REWARDS_FOUND = all_rewards_found
        match = get_match_number(step)
        match_step = get_match_step(step)
        num_relics_th = 2 * min(match, Global.LAST_MATCH_WHEN_RELIC_CAN_APPEAR) + 1
        if not Global.ALL_RELICS_FOUND:
            if len(self._relic_nodes) >= num_relics_th:
                Global.ALL_RELICS_FOUND = True
                for node in self:
                    if not node.explored_for_relic:
                        self._update_relic_status(*node.coordinates, status=False)
        if not Global.ALL_REWARDS_FOUND:
            if (
                match_step > Global.LAST_MATCH_STEP_WHEN_RELIC_CAN_APPEAR
                or len(self._relic_nodes) >= num_relics_th
            ):
                self._update_reward_status_from_relics_distribution()
                self._update_reward_results(obs, team_id, team_reward)
                self._update_reward_status_from_reward_results()

    def _update_reward_status_from_reward_results(self):
        for result in Global.REWARD_RESULTS:
            unknown_nodes = set()
            known_reward = 0
            for n in result["nodes"]:
                if n.explored_for_reward and not n.reward:
                    continue
                if n.reward:
                    known_reward += 1
                    continue
                unknown_nodes.add(n)
            if not unknown_nodes:
                continue
            reward = result["reward"] - known_reward
            if reward == 0:
                for node in unknown_nodes:
                    self._update_reward_status(*node.coordinates, status=False)
            elif reward == len(unknown_nodes):
                for node in unknown_nodes:
                    self._update_reward_status(*node.coordinates, status=True)
            elif reward > len(unknown_nodes):
                pass

    def _update_reward_results(self, obs, team_id, team_reward):
        ship_nodes = set()
        for active, energy, position in zip(
            obs["units_mask"][team_id],
            obs["units"]["energy"][team_id],
            obs["units"]["position"][team_id],
        ):
            if active and energy >= 0:
                ship_nodes.add(self.get_node(*position))
        Global.REWARD_RESULTS.append({"nodes": ship_nodes, "reward": team_reward})

    def _update_reward_status_from_relics_distribution(self):
        relic_map = np.zeros((self.space_size, self.space_size), np.int32)
        for node in self:
            if node.relic or not node.explored_for_relic:
                relic_map[node.y][node.x] = 1
        reward_size = 2 * Global.RELIC_REWARD_RANGE + 1
        reward_map = convolve2d(
            relic_map,
            np.ones((reward_size, reward_size), dtype=np.int32),
            mode="same",
            boundary="fill",
            fillvalue=0,
        )
        for node in self:
            if reward_map[node.y][node.x] == 0:
                node.update_reward_status(False)

    def _update_relic_status(self, x, y, status):
        node = self.get_node(x, y)
        node.update_relic_status(status)
        opp_node = self.get_node(*get_opposite(x, y))
        opp_node.update_relic_status(status)
        if status:
            self._relic_nodes.add(node)
            self._relic_nodes.add(opp_node)

    def _update_reward_status(self, x, y, status):
        node = self.get_node(x, y)
        node.update_reward_status(status)
        opp_node = self.get_node(*get_opposite(x, y))
        opp_node.update_reward_status(status)
        if status:
            self._reward_nodes.add(node)
            self._reward_nodes.add(opp_node)

    def _update_map(self, obs):
        current_obstacles = np.zeros((self.space_size, self.space_size), dtype=bool)
        for node in self:
            if node.type == NodeType.asteroid:
                current_obstacles[node.y, node.x] = True
        self.obstacle_history.append(current_obstacles)
        sensor_mask = obs["sensor_mask"]
        obs_energy = obs["map_features"]["energy"]
        obs_tile_type = obs["map_features"]["tile_type"]
        for node in self:
            x, y = node.coordinates
            is_visible = bool(sensor_mask[x, y])
            node.is_visible = is_visible
            if is_visible:
                node.type = NodeType(int(obs_tile_type[x, y]))
                self.get_node(*get_opposite(x, y)).type = node.type
                node.energy = int(obs_energy[x, y])
                self.get_node(*get_opposite(x, y)).energy = node.energy
            elif node.energy is not None:
                node.energy = None
        self.movement_counter += 1
        if self.movement_counter >= self.best_period:
            self.update_obstacle_movement_prediction()
            self.movement_counter = 0

    def update_obstacle_movement_prediction(self):
        best_direction = (0, 0)
        best_period = 0
        best_match_score = -1
        for period in [10, 20, 40]:
            for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                match_score = self.calculate_match_score(direction, period)
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_direction = direction
                    best_period = period
        if best_match_score > 0.5:
            self.best_direction = best_direction
            self.best_period = best_period

    def calculate_match_score(self, direction, period):
        if len(self.obstacle_history) < period:
            return 0
        past_obstacles = self.obstacle_history[-period]
        moved_past_obstacles = self.move_obstacles_array(past_obstacles, direction)
        current_obstacles = self.obstacle_history[-1]
        matches = np.sum(moved_past_obstacles == current_obstacles)
        total_tiles = self.space_size * self.space_size
        match_score = matches / total_tiles
        return match_score

    def move_obstacles_array(self, obstacles, direction):
        dx, dy = direction
        moved_obstacles = np.copy(obstacles)
        moved_obstacles = np.roll(moved_obstacles, shift=(dy, dx), axis=(0, 1))
        return moved_obstacles

    def clear(self):
        for node in self:
            node.is_visible = False

    def move_obstacles(self, step):
        if self.best_period > 0:
            speed = 1.0 / self.best_period
            if (step - 1) * speed % 1 > step * speed % 1:
                self.move(*self.best_direction, inplace=True)

    def move(self, dx: int, dy: int, *, inplace=False) -> "Space":
        if not inplace:
            new_space = copy.deepcopy(self)
            for node in self:
                x, y = warp_point(node.x + dx, node.y + dy)
                new_space.get_node(x, y).type = node.type
            return new_space
        else:
            types = [n.type for n in self]
            for node, node_type in zip(self, types):
                x, y = warp_point(node.x + dx, node.y + dy)
                self.get_node(x, y).type = node_type
            return self

    def clear_exploration_info(self):
        Global.REWARD_RESULTS = []
        Global.ALL_RELICS_FOUND = False
        Global.ALL_REWARDS_FOUND = False
        for node in self:
            if not node.relic:
                self._update_relic_status(node.x, node.y, status=None)

class Ship:
    def __init__(self, unit_id: int):
        self.unit_id = unit_id
        self.energy = 0
        self.node: Node | None = None
        self.task: str | None = None
        self.target: Node | None = None
        self.action: ActionType | tuple | None = None

    def __repr__(self):
        return f"Ship({self.unit_id}, node={self.node.coordinates}, energy={self.energy})"

    @property
    def coordinates(self):
        return self.node.coordinates if self.node else None

    def clean(self):
        self.energy = 0
        self.node = None
        self.task = None
        self.target = None
        self.action = None

class Fleet:
    def __init__(self, team_id):
        self.team_id: int = team_id
        self.points: int = 0
        self.ships = [Ship(unit_id) for unit_id in range(Global.MAX_UNITS)]

    def __repr__(self):
        return f"Fleet({self.team_id})"

    def __iter__(self):
        for ship in self.ships:
            if ship.node is not None:
                yield ship

    def clear(self):
        self.points = 0
        for ship in self.ships:
            ship.clean()

    def update(self, obs, space: Space):
        self.points = int(obs["team_points"][self.team_id])
        for ship, active, position, energy in zip(
            self.ships,
            obs["units_mask"][self.team_id],
            obs["units"]["position"][self.team_id],
            obs["units"]["energy"][self.team_id],
        ):
            if active:
                ship.node = space.get_node(*position)
                ship.energy = int(energy)
                ship.action = None
            else:
                ship.clean()

class Agent:
    def __init__(self, player: str, env_cfg, training=True) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg
        self.training = training
        self.map_width = env_cfg["map_width"]
        self.map_height = env_cfg["map_height"]
        self.UNIT_MOVE_COST = env_cfg["unit_move_cost"]
        self.UNIT_SAP_COST = env_cfg["unit_sap_cost"]
        self.UNIT_SAP_RANGE = env_cfg["unit_sap_range"]
        self.UNIT_SENSOR_RANGE = env_cfg["unit_sensor_range"]

        self.space = Space(self.map_width)
        self.fleet = Fleet(self.team_id)
        self.opp_fleet = Fleet(self.opp_team_id)

        # --- DQN Parameters (Carefully Tuned) ---
        self.local_view_size = 11
        self.input_channels = 9
        self.action_size = 5
        self.hidden_size = 512
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995  # Adjust decay rate if needed
        self.learning_rate = 0.0001  # Consider increasing if training is too slow
        self.replay_buffer_capacity = 50000
        self.beta_frames = 200000
        self.frame_count = 1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.input_channels, self.action_size, self.local_view_size, self.hidden_size).to(self.device)
        self.target_net = DQN(self.input_channels, self.action_size, self.local_view_size, self.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.memory = ReplayBuffer(self.replay_buffer_capacity, beta_frames=self.beta_frames)
        self.epsilon = self.epsilon_start

        # Activate the learning rate scheduler
        self.scheduler = StepLR(self.optimizer, step_size=10000, gamma=0.1)

        if not training:
            self.load_model()
            self.epsilon = 0.0

        self.unit_explore_locations = dict()

    def _state_representation(self, unit_pos, unit_energy, step, obs):
        local_view_size = self.local_view_size
        pad_width = local_view_size // 2

        tile_type_channel = np.zeros((local_view_size, local_view_size))
        energy_channel = np.zeros((local_view_size, local_view_size))
        relic_channel = np.zeros((local_view_size, local_view_size))
        reward_channel = np.zeros((local_view_size, local_view_size))
        units_channel = np.zeros((local_view_size, local_view_size))
        unit_energy_channel = np.full((local_view_size, local_view_size), unit_energy / 100.0)
        step_channel = np.full((local_view_size, local_view_size), step / 505.0)
        dist_relic_channel = np.full((local_view_size, local_view_size), 1.0)
        dist_reward_channel = np.full((local_view_size, local_view_size), 1.0)

        x_center, y_center = unit_pos
        for x_local in range(-pad_width, pad_width + 1):
            for y_local in range(-pad_width, pad_width + 1):
                x_map = (x_center + x_local) % self.map_width
                y_map = (y_center + y_local) % self.map_height
                x_view = x_local + pad_width
                y_view = y_local + pad_width

                node = self.space.get_node(x_map, y_map)
                tile_type_channel[y_view, x_view] = node.type.value / 3.0
                if node.energy is not None:
                    energy_channel[y_view, x_view] = node.energy / 100.0
                if node.relic:
                    relic_channel[y_view, x_view] = 1.0
                if node.explored_for_reward:
                    reward_channel[y_view, x_view] = 1.0 if node.reward else 0.0

                for ship in self.fleet:
                    if ship.coordinates == (x_map, y_map):
                        units_channel[y_view, x_view] = ship.energy / 100.0
                for opp_ship in self.opp_fleet:
                    if opp_ship.coordinates == (x_map, y_map):
                        units_channel[y_view, x_view] = -opp_ship.energy / 100.0

                if not node.explored_for_relic:
                    dist_relic_channel[y_view, x_view] = min(dist_relic_channel[y_view, x_view],
                        manhattan_distance((x_center, y_center), (x_map, y_map)) / (self.map_width + self.map_height))
                if node.explored_for_reward and node.reward:
                    dist_reward_channel[y_view, x_view] = min(dist_reward_channel[y_view, x_view],
                        manhattan_distance((x_center, y_center), (x_map, y_map)) / (self.map_width + self.map_height))

        state = np.stack([
            tile_type_channel,
            energy_channel,
            relic_channel,
            reward_channel,
            units_channel,
            unit_energy_channel,
            step_channel,
            dist_relic_channel,
            dist_reward_channel,
        ], axis=0)

        return torch.FloatTensor(state).to(self.device)

    def _get_action_mask(self, unit_pos):
        mask = np.ones(self.action_size)
        x, y = unit_pos
        if x == 0:
            mask[ActionType.left] = 0
        if x == self.map_width - 1:
            mask[ActionType.right] = 0
        if y == 0:
            mask[ActionType.up] = 0
        if y == self.map_height - 1:
            mask[ActionType.down] = 0
        return mask

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        points = int(obs["team_points"][self.team_id])
        reward = max(0, points - self.fleet.points)
        self.space.update(step, obs, self.team_id, reward)
        self.fleet.update(obs, self.space)
        self.opp_fleet.update(obs, self.space)
        actions_array = []

        for unit_id in range(len(self.fleet.ships)):
            if self.fleet.ships[unit_id].node is None:
                actions_array.append((0, 0, 0))
                continue

            ship = self.fleet.ships[unit_id]
            unit_pos = obs["units"]["position"][self.team_id][unit_id]
            unit_energy = obs["units"]["energy"][self.team_id][unit_id]

            if self.try_sap_on_enemies(ship):
                actions_array.append(ship.action)
                continue

            
            state = self._state_representation(unit_pos, unit_energy, step, obs)
            action_mask = self._get_action_mask(unit_pos)
            if random.random() < self.epsilon and self.training:  # Explore
                valid_actions = np.where(action_mask == 1)[0]
                action_type = random.choice(valid_actions) if valid_actions.size > 0 else 0
            else:  # Exploit
                with torch.no_grad():
                    q_values = self.policy_net(state.unsqueeze(0))
                    q_values = q_values * torch.FloatTensor(action_mask).to(self.device)
                    action_type = q_values
                    action_type = q_values.argmax().item()

            actions_array.append((action_type, 0, 0))
        return np.array(actions_array)

    def try_sap_on_enemies(self, ship) -> bool:
        targets = []
        for opp_ship in self.opp_fleet:
            if opp_ship.node is None:
                continue
            dist = manhattan_distance(ship.coordinates, opp_ship.coordinates)
            if dist <= self.UNIT_SAP_RANGE:
                priority = opp_ship.energy - dist
                targets.append((priority, opp_ship))
        if targets:
            targets.sort(key=lambda x: x[0], reverse=True)
            best_target_ship = targets[0][1]
            dx = best_target_ship.coordinates[0] - ship.coordinates[0]
            dy = best_target_ship.coordinates[1] - ship.coordinates[1]
            ship.action = (ActionType.sap, dx, dy)
            return True
        return False

    def learn(self, step, last_obs, actions, obs, rewards, dones):
        if not self.training or len(self.memory) < self.batch_size:
            return

        batch, indices, weights = self.memory.sample(self.batch_size)
        self.frame_count += 1
        states, actions_tensor, rewards_tensor, next_states, dones_tensor = zip(*batch)
        states = torch.stack(states)
        actions_tensor = torch.LongTensor(actions_tensor).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards_tensor).to(self.device)
        next_states = torch.stack(next_states)
        dones_tensor = torch.FloatTensor(dones_tensor).to(self.device)
        weights = weights.view(-1, 1)

        current_q_values = self.policy_net(states).gather(1, actions_tensor.unsqueeze(1).clamp(0, self.action_size - 1))
        next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
        next_q_values = self.target_net(next_states).gather(1, next_actions).detach()
        target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values

        errors = (target_q_values - current_q_values).abs().cpu().detach().numpy()
        self.memory.update_priorities(indices, errors)

        loss = (weights * (current_q_values - target_q_values) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.scheduler.step()  # Dynamic learning rate update

        tau = 0.005
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        print(f"Loss: {loss.item():.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}, Epsilon: {self.epsilon:.4f}, Frame: {self.frame_count}")

    def save_model(self):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'frame_count': self.frame_count,
            'epsilon': self.epsilon,
        }, f'dqn_model_{self.player}.pth')

    def load_model(self):
        try:
            checkpoint = torch.load(f'dqn_model_{self.player}.pth')
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.frame_count = checkpoint.get('frame_count', 1)
            self.epsilon = checkpoint.get('epsilon', self.epsilon_start)
            self.target_net.eval()
        except FileNotFoundError:
            print(f"No trained model found for {self.player}, starting from scratch.")

class SimpleAgent:
    def __init__(self, player: str, env_cfg, training=False) -> None:
        self.player = player
        self.team_id = 0 if player == "player_0" else 1
        self.env_cfg = env_cfg

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = []
        for unit_id in range(self.env_cfg["max_units"]):
            if obs["units_mask"][self.team_id][unit_id]:
                valid_moves = []
                x, y = obs["units"]["position"][self.team_id][unit_id]
                if x > 0: valid_moves.append(4)  # left
                if x < self.env_cfg["map_width"] - 1: valid_moves.append(2)  # right
                if y > 0: valid_moves.append(1)  # up
                if y < self.env_cfg["map_height"] - 1: valid_moves.append(3)  # down
                valid_moves.append(0) # center
                action = random.choice(valid_moves) if valid_moves else 0
                actions.append((action, 0, 0))
            else:
                actions.append((0, 0, 0))
        return np.array(actions)
```

```python
from luxai_s3.wrappers import LuxAIS3GymEnv
import numpy as np
import torch
import random

def evaluate_agents(agent_1_cls, agent_2_cls, seed=42, training=True, games_to_play=3, validate_every=5):
    # Seeding for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env = LuxAIS3GymEnv(numpy_output=True)
    obs, info = env.reset(seed=seed)
    env_cfg = info["params"]
    if training:
        player_0 = agent_1_cls("player_0", env_cfg, training=training)
        player_1 = agent_2_cls("player_1", env_cfg, training=training)
    else:
        player_0 = agent_1_cls("player_0", env_cfg, training=training)
        player_1 = agent_2_cls("player_1", env_cfg, training=training)
        player_0.load_model()
        player_1.load_model()


    for i in range(games_to_play):
        obs, info = env.reset(seed=seed+i) # Add different seed
        game_done = False
        step = 0
        last_obs = None
        last_actions = None

        while not game_done:
            actions = {}
            if training:
                last_obs = {
                    "player_0": obs["player_0"].copy() if obs["player_0"] is not None else None,
                    "player_1": obs["player_1"].copy() if obs["player_1"] is not None else None,
                }

            for agent in [player_0, player_1]:
                if obs[agent.player] is not None:
                    actions[agent.player] = agent.act(step=step, obs=obs[agent.player])
                else:
                    actions[agent.player] = np.zeros((env_cfg["max_units"], 3), dtype=int)

            if training:
                last_actions = actions.copy()

            obs, rewards, terminated, truncated, info = env.step(actions)
            dones = {k: terminated[k] | truncated[k] for k in terminated}

            new_rewards = {
                "player_0": obs["player_0"]["team_points"][player_0.team_id] if obs["player_0"] is not None else 0,
                "player_1": obs["player_1"]["team_points"][player_1.team_id] if obs["player_1"] is not None else 0,
            }
            if last_obs is not None:
                rewards = {
                    "player_0": new_rewards["player_0"] - (last_obs["player_0"]["team_points"][player_0.team_id] if last_obs["player_0"] is not None else 0),
                    "player_1": new_rewards["player_1"] - (last_obs["player_1"]["team_points"][player_1.team_id] if last_obs["player_1"] is not None else 0),
                }
            else:
                rewards = new_rewards

            if training and last_obs is not None:
                for agent in [player_0, player_1]:
                    for unit_id in range(env_cfg["max_units"]):
                        if last_obs[agent.player] is not None and last_obs[agent.player]["units_mask"][agent.team_id][unit_id]:
                            current_state = agent._state_representation(
                                last_obs[agent.player]["units"]["position"][agent.team_id][unit_id],
                                last_obs[agent.player]["units"]["energy"][agent.team_id][unit_id],
                                step,
                                last_obs[agent.player]
                            )
                            # Handle cases where a unit might disappear
                            if obs[agent.player] is not None and obs[agent.player]["units_mask"][agent.team_id][unit_id]:
                                next_state = agent._state_representation(
                                    obs[agent.player]["units"]["position"][agent.team_id][unit_id],
                                    obs[agent.player]["units"]["energy"][agent.team_id][unit_id],
                                    step + 1,
                                    obs[agent.player]
                                )
                            else:
                                 next_state = torch.zeros_like(current_state) # If unit disappears, use zeroed state

                            agent.memory.push(
                                current_state,
                                last_actions[agent.player][unit_id][0],
                                rewards[agent.player],
                                next_state,
                                dones[agent.player]
                            )

                if last_obs["player_0"] is not None:
                    player_0.learn(step, last_obs["player_0"], actions["player_0"],
                                 obs["player_0"], rewards["player_0"], dones["player_0"])
                if last_obs["player_1"] is not None:
                    player_1.learn(step, last_obs["player_1"], actions["player_1"],
                                 obs["player_1"], rewards["player_1"], dones["player_1"])

            if dones["player_0"] or dones["player_1"]:
                game_done = True
                if training:
                    player_0.save_model()
                    player_1.save_model()

            step += 1
        if training and (i + 1) % validate_every == 0:
            # Validate against the SimpleAgent
            player_1_validation = SimpleAgent("player_1", env_cfg, training=False)
            validate(player_0, player_1_validation, env_cfg)


    env.close()
    if training:
        player_0.save_model()
        player_1.save_model()

def validate(agent_0, agent_1, env_cfg, num_validation_games=5):
    print("Validating...")
    env = LuxAIS3GymEnv(numpy_output=True)
    total_score_0 = 0
    total_score_1 = 0

    for _ in range(num_validation_games):
        seed = random.randint(0, 100000) # Use different seed.
        obs, _ = env.reset(seed=seed) # Add seed
        game_done = False
        step = 0

        while not game_done:
            actions = {}
            for agent in [agent_0, agent_1]:
                if obs[agent.player] is not None:
                    actions[agent.player] = agent.act(step=step, obs=obs[agent.player])
                else:
                    actions[agent.player] = np.zeros((env_cfg["max_units"], 3), dtype=int)

            obs, _, terminated, truncated, _ = env.step(actions)
            dones = {k: terminated[k] | truncated[k] for k in terminated}

            if dones["player_0"] or dones["player_1"]:
                game_done = True
                if obs["player_0"] is not None:
                    total_score_0 += obs["player_0"]["team_points"][agent_0.team_id]
                if obs["player_1"] is not None:
                    total_score_1 += obs["player_1"]["team_points"][agent_1.team_id]
            step += 1

    avg_score_0 = total_score_0 / num_validation_games
    avg_score_1 = total_score_1 / num_validation_games
    print(f"Validation - Avg Score Player 0: {avg_score_0}, Avg Score Player 1: {avg_score_1}")
    env.close()
```

train

```python
evaluate_agents(Agent, Agent, seed=42, training=True, games_to_play=1, validate_every=1)
```

#### in this case the more loss increase mens model learning  new thinks is interesting and i'm also add a sap action model use to enemy is really the interesting journey i hope i complete this project before end this comp.

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

evaluate_agents(Agent, SimpleAgent) # here we evaluate our dummy agent against itself, it will auto render in the notebook
```