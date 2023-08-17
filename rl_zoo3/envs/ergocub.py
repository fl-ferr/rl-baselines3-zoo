import logging
import os
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import gymnasium as gym
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxsim
import jaxsim.typing as jtp
import numpy as np
from gymnasium.spaces import Box
from jaxsim import high_level, logging
from jaxsim.high_level.model import IntegratorType, Model, VelRepr
from jaxsim.physics.algos.soft_contacts import SoftContactsParams
from jaxsim.simulation import simulator_callbacks
from jaxsim.simulation.simulator import JaxSim, SimulatorData, StepData
from meshcat_viz.world import MeshcatWorld

warnings.filterwarnings("ignore")

metadata = {"render_modes": [None, "human"]}

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["IGN_GAZEBO_RESOURCE_PATH"] = "/root/element_rl-for-codesign/assets/model/ErgoCub"


class ErgoCub(gym.Env):
    """
    Description:
        A custom JAXsim-based Gymnasium environment for the ErgoCub robot.

    Action Space:`
        The action space is a `Box(-1, 1, (23,), float32)`. An action represents the torques applied at the hinge joints.

        | Num | Action                                    | Min    | Max   | Joint  | Unit           |
        |-----|-------------------------------------------|--------|-------|--------|----------------|
        | 0   | Torque applied on Left hip pitch          | -100.0 | 100.0 | hinge  | torque (N m)   |
        | 1   | Torque applied on Right hip pitch         | -100.0 | 100.0 | hinge  | torque (N m)   |
        | 2   | Torque applied on Torso roll              | -100.0 | 100.0 | hinge  | torque (N m)   |
        | 3   | Torque applied on Left hip roll           | -100.0 | 100.0 | hinge  | torque (N m)   |
        | 4   | Torque applied on Right hip roll          | -100.0 | 100.0 | hinge  | torque (N m)   |
        | 5   | Torque applied on Torso pitch             | -100.0 | 100.0 | hinge  | torque (N m)   |
        | 6   | Torque applied on Left hip yaw            | -100.0 | 100.0 | hinge  | torque (N m)   |
        | 7   | Torque applied on Right hip yaw           | -100.0 | 100.0 | hinge  | torque (N m)   |
        | 8   | Torque applied on Torso yaw               | -100.0 | 100.0 | hinge  | torque (N m)   |
        | 9   | Torque applied on Left knee               | -100.0 | 100.0 | hinge  | torque (N m)   |
        | 10  | Torque applied on Right knee              | -100.0 | 100.0 | hinge  | torque (N m)   |
        | 11  | Torque applied on Left shoulder pitch     | -100.0 | 100.0 | hinge  | torque (N m)   |
        | 12  | Torque applied on Right shoulder pitch    | -100.0 | 100.0 | hinge  | torque (N m)   |
        | 13  | Torque applied on Left ankle pitch        | -100.0 | 100.0 | hinge  | torque (N m)   |
        | 14  | Torque applied on Right ankle pitch       | -100.0 | 100.0 | hinge  | torque (N m)   |
        | 15  | Torque applied on Left shoulder roll      | -100.0 | 100.0 | hinge  | torque (N m)   |
        | 16  | Torque applied on Right shoulder roll     | -100.0 | 100.0 | hinge  | torque (N m)   |
        | 17  | Torque applied on Right elbow             | -100.0 | 100.0 | hinge  | torque (N m)   |
        | 18  | Torque applied on Left ankle roll         | -100.0 | 100.0 | hinge  | torque (N m)   |
        | 19  | Torque applied on Right ankle roll        | -100.0 | 100.0 | hinge  | torque (N m)   |
        | 20  | Torque applied on Left ankle yaw          | -100.0 | 100.0 | hinge  | torque (N m)   |
        | 21  | Torque applied on Right ankle yaw         | -100.0 | 100.0 | hinge  | torque (N m)   |

    Observation Space:

        Observations consist of positional values of different body parts of the ErgoCub,
        followed by the velocities of those individual parts (their derivatives) with all the
        positions ordered before all the velocities.

        However, by default, the observation is a `ndarray` with shape `(80,)` where the elements correspond to the following:

        | Num | Observation                                    | Min  | Max | Joint | Unit             |
        | --- | ---------------------------------------------- | ---- | --- | ----- | ---------------- |
        | 0   | z-coordinate of the base (world frame)         | -Inf | Inf | free  | position (m)     |
        | 1   | x-velocity of the base (world frame)           | -Inf | Inf | free  | velocity (m/s)   |
        | 2   | y-velocity of the base (world frame)           | -Inf | Inf | free  | velocity (m/s)   |
        | 3   | z-velocity of the base (world frame)           | -Inf | Inf | free  | velocity (m/s)   |
        | 4   | x-angular velocity of the base (world frame)   | -Inf | Inf | free  | velocity (rad/s) |
        | 5   | y-angular velocity of the base (world frame)   | -Inf | Inf | free  | velocity (rad/s) |
        | 6   | z-angular velocity of the base (world frame)   | -Inf | Inf | free  | velocity (rad/s) |
        | 7   | Position of the left hip pitch                 | -Inf | Inf | free  | position (m)     |
        | 8   | Position of the Right hip pitch                | -Inf | Inf | free  | position (m)     |
        | 9   | Position of the Torso roll                     | -Inf | Inf | free  | position (m)     |
        | 10  | Position of the Left hip roll                  | -Inf | Inf | free  | angle (rad)      |
        | 11  | Position of the Right hip roll                 | -Inf | Inf | free  | angle (rad)      |
        | 12  | Position of the Torso pitch                    | -Inf | Inf | free  | angle (rad)      |
        | 13  | Position of the Left hip yaw                   | -Inf | Inf | hinge | angle (rad)      |
        | 14  | Position of the Right hip yaw                  | -Inf | Inf | hinge | angle (rad)      |
        | 15  | Position of the Torso yaw                      | -Inf | Inf | hinge | angle (rad)      |
        | 16  | Position of the Left knee                      | -Inf | Inf | hinge | angle (rad)      |
        | 17  | Position of the Right knee                     | -Inf | Inf | hinge | angle (rad)      |
        | 18  | Position of the Left shoulder pitch            | -Inf | Inf | hinge | angle (rad)      |
        | 19  | Position of the Right shoulder pitch           | -Inf | Inf | hinge | angle (rad)      |
        | 20  | Position of the Left ankle pitch               | -Inf | Inf | hinge | angle (rad)      |
        | 21  | Position of the Right ankle pitch              | -Inf | Inf | hinge | angle (rad)      |
        | 22  | Position of the Left shoulder roll             | -Inf | Inf | hinge | angle (rad)      |
        | 23  | Position of the Right shoulder roll            | -Inf | Inf | hinge | angle (rad)      |
        | 24  | Position of the Right elbow                    | -Inf | Inf | hinge | angle (rad)      |
        | 25  | Position of the Left ankle roll                | -Inf | Inf | hinge | angle (rad)      |
        | 26  | Position of the Right ankle roll               | -Inf | Inf | hinge | angle (rad)      |
        | 27  | Position of the Left ankle yaw                 | -Inf | Inf | hinge | angle (rad)      |
        | 28  | Position of the Right ankle yaw                | -Inf | Inf | hinge | angle (rad)      |
        | 29  | Velocity of the Left hip pitch                 | -Inf | Inf | hinge | angle (rad)      |
        | 30  | Velocity of the Right hip pitch                | -Inf | Inf | hinge | angle (rad)      |
        | 31  | Velocity of the Torso roll                     | -Inf | Inf | hinge | angle (rad)      |
        | 32  | Velocity of the Left hip roll                  | -Inf | Inf | hinge | angle (rad)      |
        | 33  | Velocity of the Right hip roll                 | -Inf | Inf | hinge | angle (rad)      |
        | 34  | Velocity of the Torso pitch                    | -Inf | Inf | hinge | velocity (rad/s) |
        | 35  | Velocity of the Left hip yaw                   | -Inf | Inf | hinge | velocity (rad/s) |
        | 36  | Velocity of the Right hip yaw                  | -Inf | Inf | hinge | velocity (rad/s) |
        | 37  | Velocity of the Torso yaw                      | -Inf | Inf | hinge | velocity (rad/s) |
        | 38  | Velocity of the Left knee                      | -Inf | Inf | hinge | velocity (rad/s) |
        | 39  | Velocity of the Right knee                     | -Inf | Inf | hinge | velocity (rad/s) |
        | 40  | Velocity of the Left shoulder pitch            | -Inf | Inf | hinge | velocity (rad/s) |
        | 41  | Velocity of the Right shoulder pitch           | -Inf | Inf | hinge | velocity (rad/s) |
        | 42  | Velocity of the Left ankle pitch               | -Inf | Inf | hinge | velocity (rad/s) |
        | 43  | Velocity of the Right ankle pitch              | -Inf | Inf | hinge | velocity (rad/s) |
        | 44  | Velocity of the Left shoulder roll             | -Inf | Inf | hinge | velocity (rad/s) |
        | 45  | Velocity of the Right shoulder roll            | -Inf | Inf | hinge | velocity (rad/s) |
        | 46  | Velocity of the Right elbow                    | -Inf | Inf | hinge | velocity (rad/s) |
        | 47  | Velocity of the Left ankle roll                | -Inf | Inf | hinge | velocity (rad/s) |
        | 48  | Velocity of the Right ankle roll               | -Inf | Inf | hinge | velocity (rad/s) |
        | 49  | Velocity of the Left ankle yaw                 | -Inf | Inf | hinge | velocity (rad/s) |
        | 50  | Velocity of the Right ankle yaw                | -Inf | Inf | hinge | velocity (rad/s) |

        Additionally, after the position and velocity based values in the table, the observation array contains:
            - `Gravity Projection: Projection of gravitational forces on the robot's body and joints.
            - `Actuator Forces: Forces exerted by the actuators controlling the robot's joints.
            - `Number of contact points`: Number of points in contact with the ground.
            - `Target Position`: The position of the target in the world frame.

        | Num | Observation                              | Min  | Max | Unit                 |
        | --- | ---------------------------------------- | ---- | --- | -------------------- |
        | 51  | x-component of gravity projection        | -Inf | Inf | acceleration (m/s^2) |
        | 52  | y-component of gravity projection        | -Inf | Inf | acceleration (m/s^2) |
        | 53  | z-component of gravity projection        | -Inf | Inf | acceleration (m/s^2) |
        | 54  | Torque applied on left hip pitch         | -Inf | Inf | torque (N m)         |
        | 55  | Torque applied on Right hip pitch        | -Inf | Inf | torque (N m)         |
        | 56  | Torque applied on Torso roll             | -Inf | Inf | torque (N m)         |
        | 57  | Torque applied on Left hip roll          | -Inf | Inf | torque (N m)         |
        | 58  | Torque applied on Right hip roll         | -Inf | Inf | torque (N m)         |
        | 59  | Torque applied on Torso pitch            | -Inf | Inf | torque (N m)         |
        | 60  | Torque applied on Left hip yaw           | -Inf | Inf | torque (N m)         |
        | 61  | Torque applied on Right hip yaw          | -Inf | Inf | torque (N m)         |
        | 62  | Torque applied on Torso yaw              | -Inf | Inf | torque (N m)         |
        | 63  | Torque applied on Left knee              | -Inf | Inf | torque (N m)         |
        | 64  | Torque applied on Right knee             | -Inf | Inf | torque (N m)         |
        | 65  | Torque applied on Left shoulder pitch    | -Inf | Inf | torque (N m)         |
        | 66  | Torque applied on Right shoulder pitch   | -Inf | Inf | torque (N m)         |
        | 67  | Torque applied on Left ankle pitch       | -Inf | Inf | torque (N m)         |
        | 68  | Torque applied on Right ankle pitch      | -Inf | Inf | torque (N m)         |
        | 69  | Torque applied on Left shoulder roll     | -Inf | Inf | torque (N m)         |
        | 70  | Torque applied on Right shoulder roll    | -Inf | Inf | torque (N m)         |
        | 71  | Torque applied on Right elbow            | -Inf | Inf | torque (N m)         |
        | 72  | Torque applied on Left ankle roll        | -Inf | Inf | torque (N m)         |
        | 73  | Torque applied on Right ankle roll       | -Inf | Inf | torque (N m)         |
        | 74  | Torque applied on Left ankle yaw         | -Inf | Inf | torque (N m)         |
        | 75  | Torque applied on Right ankle yaw        | -Inf | Inf | torque (N m)         |
        | 76  | Number of contact points                 | -Inf | Inf | [-]                  |
        | 77  | x-coordinate of the target               | -Inf | Inf | distance (m)         |
        | 78  | y-coordinate of the target               | -Inf | Inf | distance (m)         |
        | 79  | z-coordinate of the target               | -Inf | Inf | distance (m)         |

        The reward is composed by the following main components:
            - Forward velocity of the robot
            - Control penalty
            - Limits penalty
            - Target distance reward
            - Contact reward
            - Gravity projection reward

    Starting State:
        The starting state is found by freezing the joints and making the robot fall from a small heigth, this allows to consider a stable initial
            position that is then used to initialize a model with active joints.

    Episode Termination:
        - Episode length is greater than the limit
        - Solved Requirements are met
        - NaN araise in the observation
    """

    def __init__(
        self,
        forward_reward_weight=2.0,
        healthy_reward=3.0,
        ctrl_cost_weight=0.1,
        contact_cost_weight=5e-7,
        _contact_cost_range: Tuple[float, float] = (-np.inf, 10.0),
        render_mode=None,  # "human",
        terminate_when_unhealthy=True,
        healthy_z_range: Union[float, float] = [0.4, 4.0],
    ):
        super(ErgoCub, self).__init__()
        self._step_size = 0.000_5
        steps_per_run = 1

        # Load model from urdf
        self.model_urdf_path = (
            Path.home()
            / "element_rl-for-codesign"
            / "assets"
            / "model"
            / "ErgoCub"
            / "ergoCub"
            / "robots"
            / "ergoCubGazeboV1_minContacts"
            / "model.urdf"
        )

        assert self.model_urdf_path.exists()

        # Create the JAXsim simulator
        self.simulator = JaxSim.build(
            step_size=self._step_size,
            steps_per_run=steps_per_run,
            velocity_representation=VelRepr.Body,
            integrator_type=IntegratorType.RungeKutta4,
            simulator_data=SimulatorData(
                contact_parameters=SoftContactsParams(K=5e6, D=3.5e4, mu=0.8),
            ),
        ).mutable(validate=False)

        # Insert model into the simulator
        model = self.simulator.insert_model_from_description(model_description=self.model_urdf_path).mutable(validate=True)

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
                "torso_yaw",
            ]
        )
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(82,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(22,), dtype=np.float64)

        self.forward_reward_weight = forward_reward_weight
        self.healthy_reward = healthy_reward
        self.ctrl_cost_weight = ctrl_cost_weight
        self.render_mode = render_mode
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._integration_time = 0.01
        self.start_position_found = False
        self.healthy_steps = 0
        self.joint_limits = [model.get_joint(j).joint_description.position_limit for j in model.joint_names()]
        self._contact_cost_range = _contact_cost_range
        self.contact_cost_weight = contact_cost_weight
        self.step_data = []

        def env_step(sim: jaxsim.JaxSim, sim_data: SimulatorData, action: np.ndarray) -> Union[jaxsim.JaxSim, StepData]:
            """"""

            @jdc.pytree_dataclass
            class PostStepLogger(simulator_callbacks.PostStepCallback):
                def post_step(self, sim: JaxSim, step_data: Dict[str, StepData]) -> Tuple[JaxSim, jtp.PyTree]:
                    return sim, step_data

            with sim.editable(validate=True) as sim_rw:
                sim_rw.data = sim_data

                # Update the model state after simulation
                model = sim_rw.get_model(model_name="ergoCub").mutable(validate=True)

                # Apply forces to model
                model.zero_input()
                model.set_joint_generalized_force_targets(forces=jnp.atleast_1d(action), joint_names=model.joint_names())

                cb_logger = PostStepLogger()

                sim_rw, (cb_logger, step_data) = sim_rw.step_over_horizon(
                    horizon_steps=int(self._integration_time / self._step_size),
                    callback_handler=cb_logger,
                    clear_inputs=False,
                )

                return sim_rw, step_data

        self.env_step = jax.jit(env_step)
        self.contact_points = jax.jit(lambda model: jnp.count_nonzero(model.in_contact()))
        self.contact_force = lambda foot, step_data, model: step_data["ergoCub"].aux["t0"]["contact_forces_links"][
            0, model.link_names().index(f"{foot}_ankle_2"), :3
        ]

        # self.feet_orientation = jax.jit(
        #     lambda model: (
        #         model.links()[11].orientation(dcm=True),
        #         model.links()[12].orientation(dcm=True),
        #     )
        # )

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
        kp = 324.0
        kd = np.sqrt(kp) / 22.0

        model = self.simulator.get_model(model_name="ergoCub").mutable(validate=True)

        action = (
            kp * (action - model.joint_positions()) - kd * model.joint_velocities()
            if not self._has_NaNs
            else np.zeros_like(model.joint_names())
        )

        action = np.clip(action, -100.0, 100.0)

        self.simulator, step_data = self.env_step(
            sim=self.simulator,
            sim_data=self.simulator.data,
            action=jnp.array(action, dtype=float),
        )

        reward = self._get_reward(action=action, step_data=step_data)

        self.step_data = step_data

        observation = self._get_observation()

        self.observation = observation if not self._has_NaNs else self.observation

        terminated = self.terminated

        done = False

        info = (
            {
                "reward_components": reward_component_value
                for reward_component, reward_component_value in zip(
                    [
                        "forward_reward",
                        # "control_penalty",
                        # "contact_cost",
                        "healthy_reward",
                        "balancing_reward",
                        "regularizer_reward",
                    ],
                    [
                        self.forward_reward,
                        # self.control_penalty,
                        # self.contact_cost,
                        self.healthy_reward,
                        self.balancing_reward,
                        self.regularizer_reward,
                    ],
                )
            }
            if not self._has_NaNs
            else {"has_NaNs": True}
        )

        if self.render_mode == "human":
            self.render()

        return self.observation, reward, terminated, done, info

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
        seed = 0 if seed == None else seed
        super().reset(seed=seed)

        key = jax.random.PRNGKey(seed)

        # Find starting base position if not already found
        if not self.start_position_found:
            self.starting_base_position = self._find_start_position()
            self.x = 0.0

        model = self.simulator.get_model("ergoCub").mutable(validate=True)

        # Reset the base position of the new model to match the saved starting position
        model.reset_base_position(
            position=self.starting_base_position + jnp.array([0.0, 0.0, 0.007]) * jax.random.uniform(key, (3,))
        )

        # Reset base and joints
        model.reset_base_orientation(orientation=jnp.array([1.0, 0.0, 0.0, 0.0]))
        model.reset_base_velocity(base_velocity=jax.random.uniform(key, (6,)) * 0.05)

        # model.reset_joint_positions(positions=self.joints_goal_position * 0.1)
        model.reset_joint_positions(positions=jax.random.uniform(key, (model.dofs(),)) * 0.005)

        model.reset_joint_velocities(velocities=jax.random.uniform(key, (model.dofs(),)) * 0.005)

        self.target_configuration = model.joint_positions()

        model.set_mutability(False)

        if self.render_mode == "human":
            self.render()

        observation = self._get_observation(reset=True)

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

        if "ergoCub" not in self.world._meshcat_models.keys():
            _ = self.world.insert_model(
                model_description=self.model_urdf_path,
                is_urdf=True,
                model_name="ergoCub",
            )

        model = self.simulator.get_model(model_name="ergoCub").mutable(mutable=False, validate=True)

        try:
            # Update the model
            self.world.update_model(
                model_name="ergoCub",
                joint_names=model.joint_names(),
                joint_positions=model.joint_positions(),
                base_position=model.base_position(),
                base_quaternion=model.base_orientation(),
            )
        except:
            pass

    def _get_reward(self, action: np.array, step_data: StepData) -> float:
        """
        Calculates the reward for the current state of the environment.

        Returns:
            reward (float): The reward for the current state of the environment.

        Note:
            The reward is calculated based on the current state of the environment.
            The reward is calculated as a weighted sum of the following components:
                - Forward velocity of the robot
                - Control penalty
                - Limits penalty
                - Target distance reward
                - Contact reward
                - Gravity projection reward
        """

        model = self.simulator.get_model(model_name="ergoCub").mutable(validate=True)

        forward_reward = np.clip(
            (model.base_position()[0] - self.x) / self._integration_time * self.forward_reward_weight, 0, np.inf
        )
        control_penalty = self.ctrl_cost_weight * np.square(action / 50.0).sum()

        # limits_penalty = -0.05 * any(
        #     map(
        #         lambda j, l: j < l[0] or j > l[1],
        #         model.joint_positions(),
        #         self.joint_limits,
        #     )
        # )

        # target_distance_reward = 0.5 * np.exp(-0.2 * np.linalg.norm(model.base_position() - self.target_position))

        # contact_reward = 0.5 if int(self.contact_points(model)) > 0 else -0.5
        min_cost, max_cost = self._contact_cost_range
        contact_cost = self.contact_cost_weight * np.sum(
            np.square(np.array([self.contact_force("l", step_data, model), self.contact_force("r", step_data, model)]))
        )
        contact_cost = np.clip(contact_cost, min_cost, max_cost)

        gravity_projection = model.base_orientation(dcm=True).T @ (
            self.simulator.gravity() / np.linalg.norm(self.simulator.gravity())
        )

        delta_gravity_angle = (
            gravity_projection
            @ np.array([0.0, 0.0, -1.0])
            / (np.linalg.norm(gravity_projection) * np.linalg.norm(np.array([0.0, 0.0, -1.0])))
        )

        balancing_reward = 1.0 * delta_gravity_angle  # ! -> 1.0

        # height_reward = np.exp(model.base_position()[2] - self.starting_base_position[2]) * 2

        # if self.is_healthy:
        #     self.healthy_steps += 1
        # else:
        #     self.healthy_steps = 0

        # healthy_reward = (
        #     self.healthy_reward * self.healthy_steps if self.is_healthy else -self.healthy_reward / 10 * self.healthy_steps
        # )

        healthy_reward = self.healthy_reward * self.is_healthy

        reg = 0.1
        regularizer_reward = (
            0.000_1 * np.square(model.joint_velocities() - reg * (self.target_configuration - model.joint_positions())).sum()
        )

        # l_foot_placement_reward = 0.1 * np.exp(-np.linalg.norm(self.feet_orientation(model)[0][2, 2] - 1.0))
        # r_foot_placement_reward = 0.1 * np.exp(-np.linalg.norm(self.feet_orientation(model)[1][2, 2] - 1.0))

        # base_orientation_reward = 2.0 * np.exp(
        #     -step_data["ergoCub"].tf_model_state.base_quaternion[-1] - np.array([1, 0, 0, 0])
        # )

        # force_delta = lambda foot: np.linalg.norm(
        #     step_data["ergoCub"].aux["t0"]["contact_forces_links"][-1, model.link_names().index(f"{foot}_ankle_2"), :3]
        #     - step_data["ergoCub"].aux["t0"]["contact_forces_links"][0, model.link_names().index(f"{foot}_ankle_2"), :3]
        # )

        # l_foot_force_reward = 0.2 if force_delta("l") < 5.0 else 0.0

        # r_foot_force_reward = 0.2 if force_delta("r") < 5.0 else 0.0

        print(f"forward_reward: {forward_reward}")
        print(f"balancing_reward: {balancing_reward}")
        print(f"healthy_reward: {healthy_reward}")
        print(f"regularizer_reward: {-regularizer_reward}")
        # print(f"contact_reward: {-contact_cost}")
        # print(f"control_penalty: {-control_penalty}")

        self.forward_reward = forward_reward
        # self.control_penalty = control_penalty
        # self.contact_cost = contact_cost
        self.healthy_reward = healthy_reward
        self.balancing_reward = balancing_reward
        self.regularizer_reward = regularizer_reward

        return (
            float(
                forward_reward
                + balancing_reward
                # + height_reward
                + healthy_reward
                - regularizer_reward
                # + l_foot_force_reward
                # + r_foot_force_reward
                # + base_orientation_reward
                # + l_foot_placement_reward
                # + r_foot_placement_reward
                # - contact_cost
                # + target_distance_reward
                # - control_penalty
                # + limits_penalty
            )
            if not self._has_NaNs
            else -10.0
        )

    def _get_observation(self, reset: bool = False) -> np.array:
        """
        Get the current observation/state of the environment.

        This method retrieves the current observation/state of the environment, which includes various state
        variables related to the robot's position, orientation, joint angles, velocities, gravity projection,
        actuator forces, contact points, and target distance.

        Returns:
            observation (np.array): The current observation/state of the environment as a NumPy array.
        """
        model = self.simulator.get_model(model_name="ergoCub").mutable(validate=True)
        base_height = np.atleast_1d(model.base_position()[2])
        base_orientation = np.array(model.base_orientation())
        w_x_velocity = (model.base_position()[0, None] - self.x) / self._integration_time
        base_yz_velocity = model.base_velocity()[1:]
        position = model.joint_positions()
        velocity = model.joint_velocities()
        gravity_projection = model.base_orientation(dcm=True).T @ (
            self.simulator.gravity() / jnp.linalg.norm(self.simulator.gravity())
        )

        actuator_forces = model.data.model_input.tau

        self.x = model.base_position()[0]

        contact_forces = np.concatenate(
            [self.contact_force(foot, self.step_data, model) if not reset else [0, 0, 0] for foot in ("l", "r")]
        )

        return np.concatenate(
            (
                base_height,
                w_x_velocity,
                base_yz_velocity,
                position,
                velocity,
                gravity_projection,
                actuator_forces,
                contact_forces,
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

        root_H_sole = dynComp.getRelativeTransform("root_link", "l_sole").getPosition().toNumPy()

        start_position = jnp.array([0.0, 0.0, -1.0 * root_H_sole[2]])
        self.start_position_found = True

        return start_position

    def _toggle_render(self):
        self.render_mode = "human" if self.render_mode == None else None

    @property
    def terminated(self) -> bool:
        return not self.is_healthy if self._terminate_when_unhealthy else False

    @property
    def is_healthy(self) -> bool:
        min_z, max_z = self._healthy_z_range
        model = self.simulator.get_model("ergoCub").mutable(validate=True)
        is_healthy = min_z < model.base_position()[2] < max_z
        return is_healthy

    @property
    def _has_NaNs(self) -> bool:
        found = np.isnan(self._get_observation(True)).any()
        if found:
            logging.warning("NaNs found in observation!")
        return found


if __name__ == "__main__":
    env = ErgoCub()
    env.reset()
    env.step(np.zeros(22))
    env.step(np.zeros(22))

from rod import Sdf
from rod.urdf.exporter import UrdfExporter