import logging
import os
import warnings
from pathlib import Path
from typing import Dict, Tuple, Union
import idyntree.bindings as iDynTree
import copy
import gymnasium as gym
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxsim
import jaxsim.typing as jtp
from jaxsim.utils import Mutability
import numpy as np
from gymnasium.spaces import Box
from jaxsim import logging, sixd
from jaxsim.high_level.model import Model, VelRepr
from jaxsim.physics.algos.soft_contacts import SoftContactsParams
from jaxsim.simulation import simulator_callbacks
from jaxsim.simulation.simulator import JaxSim, SimulatorData, StepData
from jaxsim.simulation.ode_integration import IntegratorType
from meshcat_viz.world import MeshcatWorld
from resolve_robotics_uri_py import resolve_robotics_uri

warnings.filterwarnings("ignore")

metadata = {"render_modes": [None, "human"]}

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["IGN_GAZEBO_RESOURCE_PATH"] = "/conda/share/"


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

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        forward_reward_weight=2.0,
        healthy_reward_weight=2.0,
        ctrl_cost_weight=0.01,
        prior_reward_weight=5.0,
        render_mode=None,
        terminate_when_unhealthy=True,
        healthy_z_range: Union[float, float] = [0.4, 4.0],
    ):
        super().__init__()
        self._step_size = 0.000_25
        steps_per_run = 1

        # Load model from urdf
        self.model_urdf_path = resolve_robotics_uri(
            "package://ergoCub/robots/ergoCubGazeboV1_minContacts/model.urdf"
        )
        assert self.model_urdf_path.exists()

        # jax.profiler.device_memory_profile()

        # Create the JAXsim simulator
        self.simulator = JaxSim.build(
            step_size=self._step_size,
            steps_per_run=steps_per_run,
            velocity_representation=VelRepr.Body,
            integrator_type=IntegratorType.EulerSemiImplicit,
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
                "torso_yaw",
                "torso_pitch",
            ]
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(53,), dtype=np.float32
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(23,), dtype=np.float64)

        self.forward_reward_weight = forward_reward_weight
        self.healthy_reward_weight = healthy_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.prior_reward_weight = prior_reward_weight
        self.render_mode = render_mode
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._integration_time = 0.01666  # 60Hz
        self.start_position_found = False
        self.step_data = None
        self.world = None
        self.prior = None
        # self.kp = 324.0
        # self.kd = np.sqrt(self.kp) / 22.0
        self.s_min, self.s_max = model.joint_limits()
        logging.info(f"Joint limits: {self.s_min} {self.s_max}")

        def env_step(
            sim: jaxsim.JaxSim, sim_data: SimulatorData, action: np.ndarray
        ) -> Union[jaxsim.JaxSim, StepData]:
            """"""

            @jdc.pytree_dataclass
            class PostStepLogger(simulator_callbacks.PostStepCallback):
                """
                A post-step callback that logs the data after each step.
                """

                def post_step(
                    self, sim: JaxSim, step_data: Dict[str, StepData]
                ) -> Tuple[JaxSim, jtp.PyTree]:
                    return sim, step_data

            with sim.mutable_context(mutability=Mutability.MUTABLE_NO_VALIDATION):
                sim.data = sim_data

            # Update the model state after simulation
            model = sim.get_model(model_name="ergoCub").mutable(validate=True)

            # Apply forces to model
            model.zero_input()
            model.set_joint_generalized_force_targets(
                forces=jnp.atleast_1d(action), joint_names=model.joint_names()
            )

            cb_logger = PostStepLogger()

            sim, (cb_logger, (_, step_data)) = sim.step_over_horizon(
                horizon_steps=int(self._integration_time / self._step_size),
                callback_handler=cb_logger,
                clear_inputs=False,
            )
            return sim, step_data

        self.env_step = jax.jit(env_step)
        self.contact_force = lambda foot, step_data, model: step_data["ergoCub"].aux[
            "t0"
        ]["contact_forces_links"][0, model.link_names().index(f"{foot}_ankle_2"), :3]
        # self._pd_controller = (
        #     lambda action, model: self.kp * (action - model.joint_positions())
        #     - self.kd * model.joint_velocities()
        # )
        self.CoM = jax.jit(lambda model: model.com_position())

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

        model = self.simulator.get_model(model_name="ergoCub").mutable(validate=True)

        # Rescale actions for [-1, 1] to joints limits
        # action = self.s_min + (action + 1) * 0.5 * (self.s_max - self.s_min)

        # action = (
        #     np.clip(
        #         (self._pd_controller(action, model)),
        #         -70.0,
        #         70.0,
        #     )
        #     if not self._has_NaNs
        #     else np.zeros_like(model.joint_names())
        # )
        action = action * 80.0

        self.simulator, step_data = self.env_step(
            sim=self.simulator,
            sim_data=self.simulator.data,
            action=jnp.array(action, dtype=float),
        )

        reward, reward_info = self._get_reward(action=action, step_data=step_data)

        observation = self._get_observation()

        if not self._has_NaNs:
            # self.step_data = step_data #! Slow

            self.observation = observation

            info = {}
        else:
            info = {"has_NaNs": True}

        terminated = self.terminated
        self.global_step += 1

        done = False

        info = info | reward_info

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
        seed = 0 if seed is None else seed
        super().reset(seed=seed)

        key = jax.random.PRNGKey(seed)

        model = self.simulator.get_model("ergoCub").mutable(validate=True)

        # Manually found starting position
        self.static_deflection = jnp.array([0.0, 0.0, 0.75356])
        self.static_deflection = self._find_start_position()

        # Reset the base position of the new model to match the saved starting position
        model.reset_base_position(position=self.static_deflection)
        model.reset_base_orientation(orientation=jnp.array([1.0, 0.0, 0.0, 0.0]))
        model.reset_base_velocity(base_velocity=jax.random.uniform(key, (6,)) * 0.000_5)

        model.reset_joint_positions(
            positions=jax.random.uniform(key, (model.dofs(),)) * 0.000_5
        )
        model.reset_joint_velocities(
            velocities=jax.random.uniform(key, (model.dofs(),)) * 0.000_5
        )

        self.global_step = 0

        # self.x = self.CoM(model)[0]
        # self.x = model.base_position()[0]

        self.target_configuration = model.joint_positions()

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
        if self.world is None:
            self.world = MeshcatWorld()
            self.world.open()

        if "ergoCub" not in self.world._meshcat_models.keys():
            _ = self.world.insert_model(
                model_description=self.model_urdf_path,
                is_urdf=True,
                model_name="ergoCub",
            )

        model = self.simulator.get_model(model_name="ergoCub").mutable(
            mutable=False, validate=True
        )

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

        control_penalty = self.ctrl_cost_weight * np.square(action / 80.0).sum()

        # balancing_reward = self._get_balancing_reward(model)

        # healthy_reward = self.healthy_reward_weight if self.is_healthy else 0.0

        prior_position, prior_orientation, prior_joints = self._get_prior_reward(model)

        prior_reward = self.prior_reward_weight * (
            prior_position + prior_orientation + prior_joints
        )

        info = {
            "reward_components": {
                # "healthy_reward": healthy_reward,
                "prior_position": prior_position,
                "prior_orientation": prior_orientation,
                "prior_joints": prior_joints,
                "prior_reward": prior_reward,
                "control_penalty": -control_penalty,
            }
        }

        reward = float(prior_reward - control_penalty) if not self._has_NaNs else -10.0

        return reward, info

    def _get_balancing_reward(self, model: Model) -> float:
        gravity_projection = model.base_orientation(dcm=True).T @ (
            self.simulator.gravity() / np.linalg.norm(self.simulator.gravity())
        )

        return 1.0 * np.linalg.norm(gravity_projection - np.array([0.0, 0.0, -1.0]))

    def _get_prior_reward(self, model: Model) -> float:
        self.prior = self._get_prior() if self.prior is None else self.prior

        prior_quaternion = sixd.so3.SO3.from_quaternion_xyzw(
            self.prior["base_orientation"][np.array([1, 2, 3, 0])]
        ).as_matrix()

        joints_reward = 1.0 * np.exp(
            -np.linalg.norm(model.joint_positions() - self.prior["joint_positions"])
        )

        orientation_reward = 1.0 * np.exp(
            -np.linalg.norm(
                np.eye(3) - model.base_orientation(dcm=True).T @ prior_quaternion
            )
        )
        position_reward = 0.5 * np.exp(
            -np.linalg.norm(model.base_position() - self.prior["base_position"])
        )
        # + np.linalg.norm(model.get_link("l_anke_2").position() - self.prior["l_ankle_2"])
        # + np.linalg.norm(model.get_link("r_anke_2").position() - self.prior["r_ankle_2"])

        return (position_reward, orientation_reward, joints_reward)

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
        # base_height = np.atleast_1d(model.base_position()[2])
        base_position = np.array(model.base_position())
        # com_position = np.array(self.CoM(model))
        base_orientation = np.array(model.base_orientation())
        # w_x_velocity = (model.base_position()[0, None] - self.x) / self._integration_time
        # base_yz_velocity = model.base_velocity()[1:]
        position = model.joint_positions()
        # velocity = model.joint_velocities()
        # gravity_projection = model.base_orientation(dcm=True).T @ (
        #     self.simulator.gravity() / jnp.linalg.norm(self.simulator.gravity())
        # )

        actuator_forces = model.data.model_input.tau

        # self.x = model.base_position()[0]

        # contact_forces = np.concatenate(
        #     [self.contact_force(foot, self.step_data, model) if not reset else [0, 0, 0] for foot in ("l", "r")]
        # )

        return np.concatenate(
            (
                base_position,
                # com_position,
                base_orientation,
                # w_x_velocity,
                position,
                # velocity,
                # gravity_projection,
                actuator_forces,
                # contact_forces,
            )
        ).astype(np.float32)

    def _find_start_position(self):
        """
        Find the initial position of the robot.

        Returns:
            start_position (jnp.array): The initial position of the robot as a JAX NumPy array.

        Note:
            - This method uses the iDynTree library to compute the initial position.
        """

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

    def toggle_render(self):
        self.render_mode = "human" if self.render_mode == None else None
        return self.render_mode

    @property
    def terminated(self) -> bool:
        return not self.is_healthy if self._terminate_when_unhealthy else False

    @property
    def is_healthy(self) -> bool:
        min_z, max_z = self._healthy_z_range
        model = self.simulator.get_model("ergoCub").mutable(validate=True)
        is_healthy = min_z < self.CoM(model)[2] < max_z
        return is_healthy

    @property
    def _has_NaNs(self) -> bool:
        found = np.isnan(self._get_observation(True)).any()
        if found:
            logging.warning("NaNs found in observation!")
        return found

    def _get_prior(self):
        # Load motions from .npy file
        amp_motions = np.load(
            Path(__file__).parent / "ergocub_motions.npy", allow_pickle=True
        )

        model = self.simulator.get_model("ergoCub").mutable(validate=True)

        # Filter joint list according to model joints, maintaning the order of model joints
        idxs = [
            amp_motions.item()["joints_list"].index(joint_name)
            for joint_name in model.joint_names()
        ]

        if self.global_step > 32:
            self.global_step = 0
            self.reset()
            print("Resetting")

        base_pos, base_quat, joint_pos, l_sole, r_sole = (
            np.array(amp_motions.item()["root_position"][self.global_step]),
            np.array(amp_motions.item()["root_quaternion"][self.global_step]),
            np.array(amp_motions.item()["joint_positions"])[self.global_step, idxs],
            np.array(amp_motions.item()["l_sole"])[self.global_step],
            np.array(amp_motions.item()["r_sole"])[self.global_step],
        )

        return {
            "base_position": base_pos,
            "base_orientation": base_quat,
            "joint_positions": joint_pos,
            "l_sole": l_sole,
            "r_sole": r_sole,
        }

    def apply_motions(self):
        # Load motions from .npy file
        amp_motions = np.load(
            Path(__file__).parent / "icub_backflip.npy", allow_pickle=True
        )
        model = self.simulator.get_model("ergoCub").mutable(validate=True)

        # Filter joint list according to model joints, maintaning the order of model joints
        idxs = [
            amp_motions.item()["joints_list"].index(joint_name)
            for joint_name in model.joint_names()
        ]
        self.render()

        import time

        for base_pos, base_quat, joint_pos in zip(
            np.array(amp_motions.item()["root_position"]),
            np.array(amp_motions.item()["root_quaternion"]),
            np.array(amp_motions.item()["joint_positions"])[:, idxs],
        ):
            with self.simulator.editable() as sim:
                model = sim.get_model(model_name="ergoCub").mutable(validate=True)
                model.reset_base_orientation(orientation=base_quat)
                model.reset_base_position(position=base_pos)
                model.reset_joint_positions(positions=joint_pos)
            self.simulator = sim
            self.render()
            time.sleep(0.01)
