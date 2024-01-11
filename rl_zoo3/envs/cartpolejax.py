import copy
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, Tuple, Union

import gymnasium as gym
import idyntree.bindings as iDynTree
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxsim
import jaxsim.typing as jtp
import numpy as np
from gymnasium.spaces import Box
from jaxsim import logging
from jaxsim.high_level.model import Model, VelRepr
from jaxsim.physics.algos.soft_contacts import SoftContactsParams
from jaxsim.simulation import simulator_callbacks
from jaxsim.simulation.ode_integration import IntegratorType
from jaxsim.simulation.simulator import JaxSim, SimulatorData, StepData
from jaxsim.utils import Mutability
from meshcat_viz.world import MeshcatWorld
from resolve_robotics_uri_py import resolve_robotics_uri

warnings.filterwarnings("ignore")

metadata = {"render_modes": [None, "human"]}

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["IGN_GAZEBO_RESOURCE_PATH"] = "/conda/share/"


class JaxPole(gym.Env):
    def __init__(
        self,
        forward_reward_weight=2.0,
        healthy_reward_weight=2.0,
        ctrl_cost_weight=0.1,
        prior_reward_weight=5.0,
        render_mode=None,
        terminate_when_unhealthy=True,
        healthy_z_range: Union[float, float] = [0.4, 4.0],
    ):
        super().__init__()
        self._step_size = 0.000_5
        steps_per_run = 1

        # Load model from urdf
        self.model_urdf_path = Path(
            "/root/rl-baselines3-zoo/rl_zoo3/envs/cartpole.urdf"
        )

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

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)

        self.ctrl_cost_weight = ctrl_cost_weight
        self.render_mode = render_mode
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._integration_time = 0.0333 / 2  # 30Hz
        self.step_data = None
        self.world = None
        self.kp = 2
        self.kd = np.sqrt(self.kp) * 1
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
            model = sim.get_model(model_name="cartpole").mutable(validate=True)

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
        self._pd_controller = (
            lambda action, model: self.kp * (action - model.joint_positions())
            - self.kd * model.joint_velocities()
        )

    def step(self, action: np.ndarray) -> Union[np.array, np.array, bool, bool, Dict]:
        model = self.simulator.get_model(model_name="cartpole").mutable(validate=True)

        # action = self.s_min + (action + 1) * 0.5 * (self.s_max - self.s_min)
        # print(f"Action unscaled: {action}")

        action = action * 10.0
        # print(f"Action: {action}")

        # action = (
        #     np.clip(
        #         (self._pd_controller(action, model)),
        #         -50.0,
        #         50.0,
        #     )
        #     if not self._has_NaNs
        #     else np.zeros_like(model.joint_names())
        # )

        self.simulator, step_data = self.env_step(
            sim=self.simulator,
            sim_data=self.simulator.data,
            action=jnp.array(action, dtype=float),
        )

        reward = self._get_reward(action=action, step_data=step_data)

        observation = self._get_observation()

        if not self._has_NaNs:
            self.step_data = step_data

            self.observation = observation

            info = {}
        else:
            info = {"has_NaNs": True}

        terminated = self.terminated

        done = False

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

        model = self.simulator.get_model("cartpole").mutable(validate=True)

        # Reset the base position of the new model to match the saved starting position
        model.reset_joint_positions(
            positions=jax.random.uniform(key, (model.dofs(),)) * 1.0
        )
        model.reset_joint_velocities(
            velocities=jax.random.uniform(key, (model.dofs(),)) * 0.5
        )

        if self.render_mode == "human":
            self.render()

        observation = self._get_observation(reset=True)

        info = {}

        return observation, info

    def close(self) -> None:
        pass

    def render(self) -> None:
        if self.world is None:
            self.world = MeshcatWorld()
            self.world.open()

        if "cartpole" not in self.world._meshcat_models.keys():
            _ = self.world.insert_model(
                model_description=self.model_urdf_path,
                is_urdf=True,
                model_name="cartpole",
            )

        model = self.simulator.get_model(model_name="cartpole").mutable(
            mutable=False, validate=True
        )

        try:
            # Update the model
            self.world.update_model(
                model_name="cartpole",
                joint_names=model.joint_names(),
                joint_positions=model.joint_positions(),
                base_position=model.base_position(),
                base_quaternion=model.base_orientation(),
            )
        except:
            pass

    def _get_reward(self, action: np.array, step_data: StepData) -> float:
        theta_threshold_radians = 5 * 2 * np.pi / 360
        x_threshold = 2.4

        model = self.simulator.get_model("cartpole").mutable(validate=True)

        balanced = model.joint_positions()[0] < x_threshold and (
            model.joint_positions()[1] < theta_threshold_radians
            and model.joint_positions()[1] > -theta_threshold_radians
            and model.joint_positions()[0] > -x_threshold
        )

        ctrl_cost = self.ctrl_cost_weight * np.square(action / 10).sum()

        return 2.0 - ctrl_cost if balanced else 0.0

    def _get_observation(self, reset: bool = False) -> np.array:
        model = self.simulator.get_model(model_name="cartpole").mutable(validate=True)
        position = model.joint_positions()
        velocity = model.joint_velocities()

        return np.concatenate(
            (
                position,
                velocity,
            )
        ).astype(np.float32)

    @property
    def terminated(self) -> bool:
        return not self.is_healthy if self._terminate_when_unhealthy else False

    @property
    def is_healthy(self) -> bool:
        theta_threshold_radians = 80 * 2 * np.pi / 360
        x_threshold = 2.4
        model = self.simulator.get_model("cartpole").mutable(validate=True)
        is_healthy = model.joint_positions()[0] < x_threshold and (
            model.joint_positions()[1] < theta_threshold_radians
            and model.joint_positions()[1] > -theta_threshold_radians
            and model.joint_positions()[0] > -x_threshold
        )
        return is_healthy

    @property
    def _has_NaNs(self) -> bool:
        found = np.isnan(self._get_observation(True)).any()
        if found:
            logging.warning("NaNs found in observation!")
        return found

    def toggle_render(self):
        self.render_mode = "human" if self.render_mode is None else None
        return self.render_mode
