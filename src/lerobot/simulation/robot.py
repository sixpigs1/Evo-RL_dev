"""LeRobot recording adapter for MuJoCo."""
import json

from lerobot.robots.robot import Robot

from .backend import MujocoSimulation, control_names
from .config import MujocoRobotConfig

ROBOT_TYPES = {
    "so101": "so101_follower",
    "bi_so101": "bi_so_follower",
    "piper": "piper_follower",
    "bi_piper": "bi_piper_follower",
}


class MujocoRobot(Robot):
    config_class = MujocoRobotConfig
    name = "mujoco"

    def __init__(self, config):
        self.config = config
        self.robot_type = ROBOT_TYPES[config.embodiment]
        self.id = config.id
        self.sim = None
        self._trace = []
        self.expert = None

    @property
    def action_features(self):
        return dict.fromkeys(control_names(self.config), float)

    @property
    def observation_features(self):
        return {**self.action_features,
                **{name: (cam.height, cam.width, 3) for name, cam in self.config.camera_views.items()}}

    @property
    def is_connected(self):
        return self.sim is not None

    @property
    def is_calibrated(self):
        return self.is_connected

    def connect(self, calibrate=True):
        if self.is_connected:
            raise RuntimeError("MuJoCo robot is already connected")
        self.sim = MujocoSimulation(self.config)
        self._start_trace()

    def calibrate(self):
        pass

    def configure(self):
        pass

    def reset(self, seed=None):
        self._simulation().reset(seed)
        self._start_trace()

    def _simulation(self):
        if self.sim is None:
            raise RuntimeError("MuJoCo robot is disconnected")
        return self.sim

    def get_observation(self):
        return self._simulation().observation()

    def _start_trace(self):
        if self.config.scripted_expert:
            from .expert import ScriptedExpert

            self.expert = ScriptedExpert(self.sim)
        self._trace = [self.sim.snapshot()] if self.config.state_dir else []

    def send_action(self, action):
        executed = self._simulation().step(action)
        if self.config.state_dir:
            self._trace.append({"action": executed, "state": self.sim.snapshot()})
        return executed

    def save_trace(self, dataset_id, episode_index):
        if not self.config.state_dir:
            return
        directory = self.config.state_dir / dataset_id.replace("/", "__")
        directory.mkdir(exist_ok=True, parents=True)
        path = directory / f"episode_{episode_index:06d}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._trace, default=str), encoding="utf-8")
        temporary.replace(path)

    def disconnect(self):
        self._simulation().close()
        self.sim = None
