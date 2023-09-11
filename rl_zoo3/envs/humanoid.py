import os
import warnings
from typing import Callable, Dict, Union

import gymnasium as gym

# from jax._src.cc  # ! see https://github.com/google/jax/discussions/13736#discussioncomment-5887985
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxsim
import jaxsim.typing as jtp
import numpy as np

# import tensorflow as tf
from gymnasium.spaces import Box
from jaxsim import high_level, logging
from jaxsim.high_level.model import Model
from jaxsim.physics.algos.soft_contacts import SoftContactsParams
from jaxsim.simulation.simulator import JaxSim, SimulatorData
from meshcat_viz.world import MeshcatWorld
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from torch import nn

from envs.training_callbacks import BatchNormExtractor, HParamCallback, linear_schedule

# import wandb

warnings.filterwarnings("ignore")
import pathlib

metadata = {"render_modes": ["human"]}

# Prevent TensorFlow using GPU, otherwise will collide with JAX GPU usage
# tf.config.experimental.set_visible_devices([], "GPU")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class Stickbot(gym.Env):
    """
    Description:
        A humanoid model made with primitive elements.

    Action Space:
        The action space is a `Box(-1, 1, (34,), float32)`. An action represents the torques applied at the hinge joints.

        | Num | Action                                                   | Min   | Max  | Name (as in URDF file)    | Joint  | Unit           |
        |-----|----------------------------------------------------------|-------|------|---------------------------|--------|----------------|
        | 0   | Torque applied between root_link and torso_1             | -50.0 | 50.0 | torso_roll                | hinge  | torque (N m)   |
        | 1   | Torque applied between root_link and r_hip_1             | -50.0 | 50.0 | r_hip_pitch               | hinge  | torque (N m)   |
        | 2   | Torque applied between root_link and l_hip_1             | -50.0 | 50.0 | l_hip_pitch               | hinge  | torque (N m)   |
        | 3   | Torque applied between l_hip_1 and l_hip_2               | -50.0 | 50.0 | l_hip_roll                | hinge  | torque (N m)   |
        | 4   | Torque applied between l_hip_3 and l_upper_leg           | -50.0 | 50.0 | l_hip_yaw                 | hinge  | torque (N m)   |
        | 5   | Torque applied between l_upper_leg and l_lower_leg       | -50.0 | 50.0 | l_knee                    | hinge  | torque (N m)   |
        | 6   | Torque applied between l_lower_leg and l_ankle_1         | -50.0 | 50.0 | l_ankle_pitch             | hinge  | torque (N m)   |
        | 7   | Torque applied between l_ankle_1 and l_ankle_2           | -50.0 | 50.0 | l_ankle_roll              | hinge  | torque (N m)   |
        | 8   | Torque applied between r_hip_1 and r_hip_2               | -50.0 | 50.0 | r_hip_roll                | hinge  | torque (N m)   |
        | 9   | Torque applied between r_hip_3 and r_upper_leg           | -50.0 | 50.0 | r_hip_yaw                 | hinge  | torque (N m)   |
        | 10  | Torque applied between r_upper_leg and r_lower_leg       | -50.0 | 50.0 | r_knee                    | hinge  | torque (N m)   |
        | 11  | Torque applied between r_lower_leg and r_ankle_1         | -50.0 | 50.0 | r_ankle_pitch             | hinge  | torque (N m)   |
        | 12  | Torque applied between r_ankle_1 and r_ankle_2           | -50.0 | 50.0 | r_ankle_roll              | hinge  | torque (N m)   |
        | 13  | Torque applied between torso_1 and torso_2               | -50.0 | 50.0 | torso_pitch               | hinge  | torque (N m)   |
        | 14  | Torque applied between torso_2 and chest                 | -50.0 | 50.0 | torso_yaw                 | hinge  | torque (N m)   |
        | 15  | Torque applied between chest and r_shoulder_1            | -50.0 | 50.0 | r_shoulder_pitch          | hinge  | torque (N m)   |
        | 16  | Torque applied between chest and l_shoulder_1            | -50.0 | 50.0 | l_shoulder_pitch          | hinge  | torque (N m)   |
        | 17  | Torque applied between neck_1 and neck_2                 | -50.0 | 50.0 | neck_pitch                | hinge  | torque (N m)   |
        | 18  | Torque applied between neck_2 and neck_3                 | -50.0 | 50.0 | neck_roll                 | hinge  | torque (N m)   |
        | 19  | Torque applied between neck_3 and head                   | -50.0 | 50.0 | neck_yaw                  | hinge  | torque (N m)   |
        | 20  | Torque applied between head and camera_tilt              | -50.0 | 50.0 | camera_tilt_joint         | hinge  | torque (N m)   |
        | 21  | Torque applied between head and lidar                    | -50.0 | 50.0 | lidar_joint               | hinge  | torque (N m)   |
        | 22  | Torque applied between l_shoulder_1 and l_shoulder_2     | -50.0 | 50.0 | l_shoulder_roll           | hinge  | torque (N m)   |
        | 23  | Torque applied between l_shoulder_3 and l_upper_arm      | -50.0 | 50.0 | l_shoulder_yaw            | hinge  | torque (N m)   |
        | 24  | Torque applied between l_upper_arm and l_elbow_1         | -50.0 | 50.0 | l_elbow                   | hinge  | torque (N m)   |
        | 25  | Torque applied between l_elbow_1 and l_forearm           | -50.0 | 50.0 | l_wrist_prosup            | hinge  | torque (N m)   |
        | 26  | Torque applied between l_forearm and l_wrist_1           | -50.0 | 50.0 | l_wrist_pitch             | hinge  | torque (N m)   |
        | 27  | Torque applied between l_wrist_1 and l_hand              | -50.0 | 50.0 | l_wrist_yaw               | hinge  | torque (N m)   |
        | 28  | Torque applied between r_shoulder_1 and r_shoulder_2     | -50.0 | 50.0 | r_shoulder_roll           | hinge  | torque (N m)   |
        | 29  | Torque applied between r_shoulder_3 and r_upper_arm      | -50.0 | 50.0 | r_shoulder_yaw            | hinge  | torque (N m)   |
        | 30  | Torque applied between r_upper_arm and r_elbow_1         | -50.0 | 50.0 | r_elbow                   | hinge  | torque (N m)   |
        | 31  | Torque applied between r_elbow_1 and r_forearm           | -50.0 | 50.0 | r_wrist_prosup            | hinge  | torque (N m)   |
        | 32  | Torque applied between r_forearm and r_wrist_1           | -50.0 | 50.0 | r_wrist_pitch             | hinge  | torque (N m)   |
        | 33  | Torque applied between r_wrist_1 and r_hand              | -50.0 | 50.0 | r_wrist_yaw               | hinge  | torque (N m)   |

    Observation Space:

        Observations consist of positional values of different body parts of the Humanoid,
        followed by the velocities of those individual parts (their derivatives) with all the
        positions ordered before all the velocities.

        However, by default, the observation is a `ndarray` with shape `(324,)` where the elements correspond to the following:

        | Num | Observation                                        | Min  | Max | Name (in corresponding XML file) | Joint | Unit             |
        | --- | -------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ---------------- |
        | 0   | x-coordinate of the base (world frame)             | -Inf | Inf | root                             | free  | position (m)     |
        | 1   | y-coordinate of the base (world frame)             | -Inf | Inf | root                             | free  | position (m)     |
        | 2   | z-coordinate of the base (world frame)             | -Inf | Inf | root                             | free  | position (m)     |
        | 3   | x-velocity of the base (world frame)               | -Inf | Inf | root                             | free  | velocity (m/s)   |
        | 4   | y-velocity of the base (world frame)               | -Inf | Inf | root                             | free  | velocity (m/s)   |
        | 5   | z-velocity of the base (world frame)               | -Inf | Inf | root                             | free  | velocity (m/s)   |
        | 6   | x-angular velocity of the base (world frame)       | -Inf | Inf | root                             | free  | velocity (rad/s) |
        | 7   | y-angular velocity of the base (world frame)       | -Inf | Inf | root                             | free  | velocity (rad/s) |
        | 8   | z-angular velocity of the base (world frame)       | -Inf | Inf | root                             | free  | velocity (rad/s) |
        | 9   | x-coordinate of the torso (centre)                 | -Inf | Inf | root                             | free  | position (m)     |
        | 10  | y-coordinate of the torso (centre)                 | -Inf | Inf | root                             | free  | position (m)     |
        | 11  | z-coordinate of the torso (centre)                 | -Inf | Inf | root                             | free  | position (m)     |
        | 12  | x-orientation of the torso (centre)                | -Inf | Inf | torso_1                          | free  | angle (rad)      |
        | 13  | y-orientation of the torso (centre)                | -Inf | Inf | torso_1                          | free  | angle (rad)      |
        | 14  | z-orientation of the torso (centre)                | -Inf | Inf | torso_1                          | free  | angle (rad)      |
        | 16  | Orientation of the Right hip pitch                 | -Inf | Inf | r_hip_pitch                      | hinge | angle (rad)      |
        | 17  | Orientation of the Torso roll                      | -Inf | Inf | torso_roll                       | hinge | angle (rad)      |
        | 18  | Orientation of the Left hip roll                   | -Inf | Inf | l_hip_roll                       | hinge | angle (rad)      |
        | 19  | Orientation of the Right hip roll                  | -Inf | Inf | r_hip_roll                       | hinge | angle (rad)      |
        | 20  | Orientation of the Torso pitch                     | -Inf | Inf | torso_pitch                      | hinge | angle (rad)      |
        | 21  | Orientation of the Left hip yaw                    | -Inf | Inf | l_hip_yaw                        | hinge | angle (rad)      |
        | 22  | Orientation of the Right hip yaw                   | -Inf | Inf | r_hip_yaw                        | hinge | angle (rad)      |
        | 23  | Orientation of the Torso yaw                       | -Inf | Inf | torso_yaw                        | hinge | angle (rad)      |
        | 24  | Orientation of the Left knee                       | -Inf | Inf | l_knee                           | hinge | angle (rad)      |
        | 25  | Orientation of the Right knee                      | -Inf | Inf | r_knee                           | hinge | angle (rad)      |
        | 26  | Orientation of the Left shoulder pitch             | -Inf | Inf | l_shoulder_pitch                 | hinge | angle (rad)      |
        | 27  | Orientation of the Neck pitch                      | -Inf | Inf | neck_pitch                       | hinge | angle (rad)      |
        | 28  | Orientation of the Right shoulder pitch            | -Inf | Inf | r_shoulder_pitch                 | hinge | angle (rad)      |
        | 29  | Orientation of the Left ankle pitch                | -Inf | Inf | l_ankle_pitch                    | hinge | angle (rad)      |
        | 30  | Orientation of the Right ankle pitch               | -Inf | Inf | r_ankle_pitch                    | hinge | angle (rad)      |
        | 31  | Orientation of the Left shoulder roll              | -Inf | Inf | l_shoulder_roll                  | hinge | angle (rad)      |
        | 32  | Orientation of the Neck roll                       | -Inf | Inf | neck_roll                        | hinge | angle (rad)      |
        | 33  | Orientation of the Right shoulder roll             | -Inf | Inf | r_shoulder_roll                  | hinge | angle (rad)      |
        | 34  | Orientation of the Camera tilt joint               | -Inf | Inf | camera_tilt_joint                | hinge | angle (rad)      |
        | 35  | Orientation of the Lidar joint                     | -Inf | Inf | lidar_joint                      | hinge | angle (rad)      |
        | 36  | Orientation of the Right elbow                     | -Inf | Inf | r_elbow                          | hinge | angle (rad)      |
        | 37  | Orientation of the Left wrist prosup               | -Inf | Inf | l_wrist_prosup                   | hinge | angle (rad)      |
        | 38  | Orientation of the Right wrist prosup              | -Inf | Inf | r_wrist_prosup                   | hinge | angle (rad)      |
        | 39  | Orientation of the Left wrist pitch                | -Inf | Inf | l_wrist_pitch                    | hinge | angle (rad)      |
        | 40  | Orientation of the Right wrist pitch               | -Inf | Inf | r_wrist_pitch                    | hinge | angle (rad)      |
        | 41  | Orientation of the Left wrist yaw                  | -Inf | Inf | l_wrist_yaw                      | hinge | angle (rad)      |
        | 42  | Orientation of the Right wrist yaw                 | -Inf | Inf | r_wrist_yaw                      | hinge | angle (rad)      |
        | 43  | Orientation of the Left ankle roll                 | -Inf | Inf | l_ankle_roll                     | hinge | angle (rad)      |
        | 44  | Orientation of the Right ankle roll                | -Inf | Inf | r_ankle_roll                     | hinge | angle (rad)      |
        | 45  | Orientation of the Left ankle yaw                  | -Inf | Inf | l_ankle_yaw                      | hinge | angle (rad)      |
        | 46  | Orientation of the Right ankle yaw                 | -Inf | Inf | r_ankle_yaw                      | hinge | angle (rad)      |
        | 47  | Orientation of the Left wrist roll                 | -Inf | Inf | l_wrist_roll                     | hinge | angle (rad)      |
        | 48  | Orientation of the Right wrist roll                | -Inf | Inf | r_wrist_roll                     | hinge | angle (rad)      |
        | 49  | Angular velocity of the Left hip pitch             | -Inf | Inf | l_hip_pitch                      | hinge | velocity (rad/s) |
        | 50  | Angular velocity of the Right hip pitch            | -Inf | Inf | r_hip_pitch                      | hinge | velocity (rad/s) |
        | 51  | Angular velocity of the Torso roll                 | -Inf | Inf | torso_roll                       | hinge | velocity (rad/s) |
        | 52  | Angular velocity of the Left hip roll              | -Inf | Inf | l_hip_roll                       | hinge | velocity (rad/s) |
        | 53  | Angular velocity of the Right hip roll             | -Inf | Inf | r_hip_roll                       | hinge | velocity (rad/s) |
        | 54  | Angular velocity of the Torso pitch                | -Inf | Inf | torso_pitch                      | hinge | velocity (rad/s) |
        | 55  | Angular velocity of the Left hip yaw               | -Inf | Inf | l_hip_yaw                        | hinge | velocity (rad/s) |
        | 56  | Angular velocity of the Right hip yaw              | -Inf | Inf | r_hip_yaw                        | hinge | velocity (rad/s) |
        | 57  | Angular velocity of the Torso yaw                  | -Inf | Inf | torso_yaw                        | hinge | velocity (rad/s) |
        | 58  | Angular velocity of the Left knee                  | -Inf | Inf | l_knee                           | hinge | velocity (rad/s) |
        | 59  | Angular velocity of the Right knee                 | -Inf | Inf | r_knee                           | hinge | velocity (rad/s) |
        | 60  | Angular velocity of the Left shoulder pitch        | -Inf | Inf | l_shoulder_pitch                 | hinge | velocity (rad/s) |
        | 61  | Angular velocity of the Neck pitch                 | -Inf | Inf | neck_pitch                       | hinge | velocity (rad/s) |
        | 62  | Angular velocity of the Right shoulder pitch       | -Inf | Inf | r_shoulder_pitch                 | hinge | velocity (rad/s) |
        | 63  | Angular velocity of the Left ankle pitch           | -Inf | Inf | l_ankle_pitch                    | hinge | velocity (rad/s) |
        | 64  | Angular velocity of the Right ankle pitch          | -Inf | Inf | r_ankle_pitch                    | hinge | velocity (rad/s) |
        | 65  | Angular velocity of the Left shoulder roll         | -Inf | Inf | l_shoulder_roll                  | hinge | velocity (rad/s) |
        | 66  | Angular velocity of the Neck roll                  | -Inf | Inf | neck_roll                        | hinge | velocity (rad/s) |
        | 67  | Angular velocity of the Right shoulder roll        | -Inf | Inf | r_shoulder_roll                  | hinge | velocity (rad/s) |
        | 68  | Angular velocity of the Camera tilt joint          | -Inf | Inf | camera_tilt_joint                | hinge | velocity (rad/s) |
        | 69  | Angular velocity of the Lidar joint                | -Inf | Inf | lidar_joint                      | hinge | velocity (rad/s) |
        | 70  | Angular velocity of the Right elbow                | -Inf | Inf | r_elbow                          | hinge | velocity (rad/s) |
        | 71  | Angular velocity of the Left wrist prosup          | -Inf | Inf | l_wrist_prosup                   | hinge | velocity (rad/s) |
        | 72  | Angular velocity of the Right wrist prosup         | -Inf | Inf | r_wrist_prosup                   | hinge | velocity (rad/s) |
        | 73  | Angular velocity of the Left wrist pitch           | -Inf | Inf | l_wrist_pitch                    | hinge | velocity (rad/s) |
        | 74  | Angular velocity of the Right wrist pitch          | -Inf | Inf | r_wrist_pitch                    | hinge | velocity (rad/s) |
        | 75  | Angular velocity of the Left wrist yaw             | -Inf | Inf | l_wrist_yaw                      | hinge | velocity (rad/s) |
        | 76  | Angular velocity of the Right wrist yaw            | -Inf | Inf | r_wrist_yaw                      | hinge | velocity (rad/s) |
        | 77  | Angular velocity of the Left ankle roll            | -Inf | Inf | l_ankle_roll                     | hinge | velocity (rad/s) |
        | 78  | Angular velocity of the Right ankle roll           | -Inf | Inf | r_ankle_roll                     | hinge | velocity (rad/s) |
        | 79  | Angular velocity of the Left ankle yaw             | -Inf | Inf | l_ankle_yaw                      | hinge | velocity (rad/s) |
        | 80  | Angular velocity of the Right ankle yaw            | -Inf | Inf | r_ankle_yaw                      | hinge | velocity (rad/s) |
        | 81  | Angular velocity of the Left wrist roll            | -Inf | Inf | l_wrist_roll                     | hinge | velocity (rad/s) |
        | 82  | Angular velocity of the Right wrist roll           | -Inf | Inf | r_wrist_roll                     | hinge | velocity (rad/s) |

        Additionally, after the position and velocity based values in the table, the observation array contains:
            - `gravity_projection`: Projection of gravitational forces on the robot's body and joints.
            - `actuator_forces`: Forces exerted by the actuators controlling the robot's joints.
            - `contact_points`: Number of contact points between the robot and the ground.

    Reward:
        The reward is composed by nine main components:
            - `forward_reward`: Encourages forward motion of the robot.
            - `balancing_reward`: Encourages maintaining balance of the robot.
            - `height_reward`: Encourages the robot to maintain a certain height from the ground.
            - `healthy_reward`: Encourages the robot to remain healthy, i.e., not falling or experiencing damage.
            - `l_foot_placement_reward`: Encourages the left foot of the robot to be correctly placed.
            - `r_foot_placement_reward`: Encourages the right foot of the robot to be correctly placed.
            - `contact_reward`: Encourages the robot to make contact with the ground.
            - `target_distance_reward`: Encourages the robot to approach the target position.
            - `control_penalty`: Penalizes large control signals, encouraging smoother actions.


    Starting State:
        The starting state is found by making the robot fall from a small heigth.

    Episode Termination:
        Episode length is greater than the TBD limit
        Solved Requirements
        [...]
    """

    # PPO_Position_2 has 2.0-sqrt2.0 /20.0
    def __init__(
        self,
        forward_reward_weight=2.0,
        healthy_reward=0.000_25,
        ctrl_cost_weight=0.000_05,
        render_mode="none",  # "human",
        terminate_when_unhealthy=True,
        healthy_z_range=[0.3, 4.0],
    ):
        super(Stickbot, self).__init__()
        self._step_size = 0.000_5
        steps_per_run = 1

        # Load model from urdf
        self.model_urdf_path = (
            pathlib.Path.home()
            / "element_rl-for-codesign"
            / "assets"
            / "model"
            / "Stickbot.urdf"
        )
        assert self.model_urdf_path.exists()

        # Create the JAXsim simulator
        self.simulator = JaxSim.build(
            step_size=self._step_size,
            steps_per_run=steps_per_run,
            velocity_representation=high_level.model.VelRepr.Body,
            integrator_type=high_level.model.IntegratorType.RungeKutta4,
            simulator_data=SimulatorData(
                contact_parameters=SoftContactsParams(K=5e6, D=3.5e4, mu=0.8),
            ),
        ).mutable(validate=False)

        # Insert model into the simulator
        model = self.simulator.insert_model_from_description(
            model_description=self.model_urdf_path
        ).mutable(validate=True)

        model.reduce(
            considered_joints=[
                "r_shoulder_pitch",
                "r_shoulder_roll",
                "r_shoulder_yaw",
                "r_elbow",
                "l_shoulder_pitch",
                "l_shoulder_roll",
                "l_shoulder_yaw",
                "l_elbow",
                "r_hip_pitch",
                "r_hip_roll",
                "r_hip_yaw",
                "r_knee",
                "r_ankle_pitch",
                "r_ankle_roll",
                "l_hip_pitch",
                "l_hip_roll",
                "l_hip_yaw",
                "l_knee",
                "l_ankle_pitch",
                "l_ankle_roll",
                "torso_roll",
                "torso_pitch",
                "torso_yaw",
            ]
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(83,), dtype=np.float64
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(23,), dtype=np.float64)

        self.forward_reward_weight = forward_reward_weight
        self.healthy_reward = healthy_reward
        self.ctrl_cost_weight = ctrl_cost_weight
        self.render_mode = render_mode
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._integration_time = 0.02
        self.start_position_found = False
        self.healthy_steps = 0
        self.joint_limits = [
            model.get_joint(j).joint_description.position_limit
            for j in model.joint_names()
        ]

        def env_step(
            sim: jaxsim.JaxSim, sim_data: SimulatorData, action: np.ndarray
        ) -> jaxsim.JaxSim:
            """"""

            with sim.editable(validate=True) as sim_rw:
                sim_rw.data = sim_data

            with sim.editable(validate=True) as sim_rw:
                # Update the model state after simulation
                model = sim_rw.get_model(model_name="stickBot").mutable(validate=True)

                # Apply forces to model
                model.zero_input()
                model.set_joint_generalized_force_targets(
                    forces=jnp.atleast_1d(action), joint_names=model.joint_names()
                )

                sim_rw = sim_rw.step_over_horizon(
                    horizon_steps=int(self._integration_time / self._step_size),
                    clear_inputs=False,
                )

                return sim_rw

        self.env_step = jax.jit(env_step)
        self.contact_points = jax.jit(
            lambda model: jnp.count_nonzero(model.in_contact())
        )
        self.feet_orientation = jax.jit(
            lambda model: (
                model.links()[11].orientation(dcm=True),
                model.links()[12].orientation(dcm=True),
            )
        )

    def step(self, action: np.ndarray) -> Union[np.array, np.array, bool, bool, Dict]:
        """
        Perform a single step in the environment by taking the given action.

        Args:
            action (np.ndarray): The action taken by the agent in terms of joint positions.

        Returns:
            observation (np.array): The current observation/state of the environment after the step.
            reward (float): The reward received after the step.
            terminated (bool): A flag indicating whether the episode is terminated after this step.
            done (bool): A flag indicating whether the episode is done after this step.
            info (Dict): Additional information about the step.
        """

        observation = self._get_observation()

        kp = 324.0
        kd = np.sqrt(kp) / 22.0

        model = self.simulator.get_model(model_name="stickBot").mutable(validate=True)

        action = kp * (action - model.joint_positions()) - kd * model.joint_velocities()

        self.simulator = self.env_step(
            sim=self.simulator,
            sim_data=self.simulator.data,
            action=jnp.array(action, dtype=float),
        )

        model = self.simulator.get_model(model_name="stickBot").mutable(validate=True)

        forward_reward = 2.5 * model.base_velocity()[0] * self.forward_reward_weight

        control_penalty = self.ctrl_cost_weight * np.square(action).sum()

        limits_penalty = -0.05 * any(
            map(
                lambda j, l: j < l[0] or j > l[1],
                model.joint_positions(),
                self.joint_limits,
            )
        )

        target_distance_reward = np.exp(
            -0.2 * np.linalg.norm(model.base_position() - self.target_position)
        )

        contact_reward = 0.05 if int(self.contact_points(model)) > 0 else -0.05

        gravity_projection = model.base_orientation(dcm=True).T @ (
            self.simulator.gravity() / np.linalg.norm(self.simulator.gravity())
        )

        delta_gravity_angle = (
            gravity_projection
            @ np.array([0.0, 0.0, -1.0])
            / (
                np.linalg.norm(gravity_projection)
                * np.linalg.norm(np.array([0.0, 0.0, -1.0]))
            )
        )

        balancing_reward = 0.1 * delta_gravity_angle

        height_reward = (
            np.exp(model.base_position()[2] - self.starting_base_position[2]) ** 2
        )

        if self.is_healthy:
            self.healthy_steps += 1
        else:
            self.healthy_steps = 0

        # healthy_reward = (
        #     self.healthy_reward * self.healthy_steps
        #     if self.is_healthy
        #     else -self.healthy_reward * self.healthy_steps
        # )

        l_foot_placement_reward = 0.1 * np.exp(
            -np.linalg.norm(self.feet_orientation(model)[0][2, 2] - 1.0)
        )
        r_foot_placement_reward = 0.1 * np.exp(
            -np.linalg.norm(self.feet_orientation(model)[1][2, 2] - 1.0)
        )

        reward = float(
            forward_reward
            + balancing_reward
            + height_reward
            # + healthy_reward
            + l_foot_placement_reward
            + r_foot_placement_reward
            + contact_reward
            + target_distance_reward
            - control_penalty
            + limits_penalty
        )

        terminated = self.terminated

        done = np.linalg.norm(model.base_position() - self.target_position) < 0.1

        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, done, info

    def reset(
        self,
        seed: int = None,
    ) -> np.array:
        """
        Resets the state of the object and initializes it for a new episode.

        Args:
            seed (int): The seed value for random number generation. Default is 0.

        Returns:
            observation (np.array): The initial observation after resetting the object.
            info (dict): Additional information about the reset process.

        """
        logging.warning("RESET CALLED!")
        seed = 0 if seed == None else seed
        super().reset(seed=seed)

        # key = jax.random.PRNGKey(seed)

        # Find starting base position if not already found
        if not self.start_position_found:
            import time

            now = time.time()
            self.starting_base_position = self._find_start_position()
            logging.warning(f"Initialization completed in {time.time() - now} seconds.")

            self.target_position = self.starting_base_position + jnp.array(
                [10.0, 0.0, 0.0]
            )

        model = self.simulator.get_model("stickBot").mutable(validate=True)

        # Reset the base position of the new model to match the saved starting position
        model.reset_base_position(
            position=self.starting_base_position + jnp.array([0.0, 0.0, 0.008])
        )

        # Reset base and joints
        model.reset_base_orientation(orientation=jnp.array([1.0, 0.0, 0.0, 0.0]))
        model.reset_base_velocity(base_velocity=jnp.zeros_like(model.base_velocity()))

        # model.reset_joint_positions(positions=self.joints_goal_position * 0.1)
        model.reset_joint_positions(positions=jnp.zeros_like(model.joint_positions()))

        model.reset_joint_velocities(velocities=jnp.zeros_like(model.joint_positions()))

        model.set_mutability(False)

        if self.render_mode == "human":
            self.render()

        observation = self._get_observation()

        info = {}
        return observation, info

    def close(self) -> None:
        """
        Clean up and close the environment.

        This method is responsible for cleaning up any resources used by the environment and closing any
        visualization or simulation tools.

        Returns:
            None
        """
        # self.world.close()
        pass

    def render(self) -> None:
        """
        Render the current state of the environment for visualization.

        This method displays the current state of the environment, including the robot's position, orientation,
        joint angles, and other relevant information, in a visualization tool. It uses Meshcat for rendering.

        Returns:
            None

        Note:
        - The rendering functionality is enabled if the `render_mode` is set to "human".
        """
        if not hasattr(self, "world"):
            self.world = MeshcatWorld()
            self.world.open()

        if "stickBot" not in self.world._meshcat_models.keys():
            _ = self.world.insert_model(
                model_description=self.model_urdf_path,
                is_urdf=True,
                model_name="stickBot",
            )
            import rod
            from rod.builder.primitives import SphereBuilder

            # Insert target point
            target_sdf = (
                pathlib.Path.home()
                / "element_rl-for-codesign"
                / "assets"
                / "model"
                / "Sphere.urdf"
            )

            _ = self.world.insert_model(
                model_description=target_sdf,
            )

            self.world.update_model(
                model_name="ball",
                base_position=self.target_position,
            )

        model = self.simulator.get_model(model_name="stickBot").mutable(
            mutable=False, validate=True
        )

        try:
            # Update the model
            self.world.update_model(
                model_name="stickBot",
                joint_names=model.joint_names(),
                joint_positions=model.joint_positions(),
                base_position=model.base_position(),
                base_quaternion=model.base_orientation(),
            )
        except:
            pass

    def _toggle_render(self):
        self.render_mode = "human" if self.render_mode == None else None

    def _get_observation(self) -> np.array:
        """
        Get the current observation/state of the environment.

        This method retrieves the current observation/state of the environment, which includes various state
        variables related to the robot's position, orientation, joint angles, velocities, gravity projection,
        actuator forces, contact points, and target distance.

        Returns:
            observation (np.array): The current observation/state of the environment as a NumPy array.
        """
        model = self.simulator.get_model(model_name="stickBot").mutable(validate=True)
        base_height = np.atleast_1d(model.base_position()[2])
        base_velocity = model.base_velocity()
        position = model.joint_positions()
        velocity = model.joint_velocities()
        gravity_projection = model.base_orientation(dcm=True).T @ (
            self.simulator.gravity() / jnp.linalg.norm(self.simulator.gravity())
        )

        actuator_forces = model.data.model_input.tau

        target_position = self.target_position

        # contact_forces = np.atleast_1d(self.contact_forces(model))

        return np.concatenate(
            (
                base_height,
                base_velocity,
                position,
                velocity,
                gravity_projection,
                actuator_forces,
                np.atleast_1d(self.contact_points(model)),
                target_position,
            )
        )

    def _find_start_position(self):
        """
        Find the initial position of the robot.

        Returns:
            start_position (jnp.array): The initial position of the robot as a JAX NumPy array.

        Note:
            - This method uses the iDynTree library to compute the initial position.
        """
        import idyntree.bindings as iDynTree

        dynComp = iDynTree.KinDynComputations()
        mdlLoader = iDynTree.ModelLoader()
        mdlLoader.loadModelFromFile(str(self.model_urdf_path))
        dynComp.loadRobotModel(mdlLoader.model())

        root_H_sole = (
            dynComp.getRelativeTransform("root_link", "l_sole").getPosition().toNumPy()
        )

        start_position = jnp.array([0.0, 0.0, -1.0 * root_H_sole[2]])
        self.start_position_found = True

        return start_position

    @property
    def terminated(self) -> bool:
        return not self.is_healthy if self._terminate_when_unhealthy else False

    @property
    def is_healthy(self) -> bool:
        min_z, max_z = self._healthy_z_range
        model = self.simulator.get_model("stickBot").mutable(validate=True)
        is_healthy = min_z < model.base_position()[2] < max_z
        return is_healthy
