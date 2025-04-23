import importlib
import os
from collections import deque
from pathlib import Path

import torch
from dataclasses import dataclass
from typing import Dict, Any
import time
import copy
from benchmarl.environments import MAgentTask
from benchmarl.experiment.experiment import ExperimentConfig
from benchmarl.algorithms.common import AlgorithmConfig, Algorithm
from benchmarl.models.common import ModelConfig
from benchmarl.utils import seed_everything
from benchmarl.algorithms import VdnConfig
from benchmarl.models import (
    SequenceModelConfig,
    GnnConfig,
    MlpConfig,
    CnnConfig,
    SelectiveGnnConfig,
    GumbelSelectiveGnnConfig,
    GnnTwoLayersConfig,
)
from benchmarl.experiment.logger import Logger
from torchrl.envs.utils import set_exploration_type, ExplorationType
from torchrl.record.loggers import generate_exp_name
from benchmarl.utils import seed_everything


_has_hydra = importlib.util.find_spec("hydra") is not None
if _has_hydra:
    from hydra.core.hydra_config import HydraConfig


@dataclass
class MultiAlgorithmEvaluationConfig(ExperimentConfig):
    """
    Inherits from the standard BenchMARL ExperimentConfig, so we can
    read e.g. train_device, buffer_device, evaluation_episodes, etc.
    """
    pass


class MultiAlgorithmEvaluation:
    """
    Each group => its own Algorithm object (with possibly unique architecture).
    We'll do a single environment rollout for multi-agent evaluation.

    Pseudocode flow:
      1) Setup environment from self.task
      2) For each group => create a "GroupStub" that fakes experiment references
         => group_algorithm_config.get_algorithm(experiment=GroupStub)
      3) Merge sub-policies => run rollout
      4) load_state_dict => partial load for each group
    """

    def __init__(
        self,
        task: MAgentTask,  # or any multi-agent Task
        group_algorithm_configs: Dict[str, AlgorithmConfig],
        group_model_configs: Dict[str, ModelConfig],
        seed: int,
        config: MultiAlgorithmEvaluationConfig,
        name: str
    ):
        self.task = task
        self.group_algorithm_configs = group_algorithm_configs
        self.group_model_configs = group_model_configs
        self.seed = seed
        self.config = config
        self.env = None
        self.group_map = {}
        self.max_steps = 0
        self.name = name

        # We'll store each group's algorithm, losses, replay buffers, etc.
        self.group_algorithms: Dict[str, Algorithm] = {}
        self.losses: Dict[str, torch.nn.Module] = {}
        self.replay_buffers: Dict[str, Any] = {}

        # This might be your final joint policy
        self.joint_policy = None

        # Some ephemeral states
        self.total_time = 0.0
        self.total_frames = 0
        self.n_iters_performed = 0
        self.mean_return = 0

        # Perform the setup
        self._setup()

    def _setup(self):
        seed_everything(self.seed)

        # 1) build environment
        self.env = self.task.get_env_fun(
            num_envs=self.config.evaluation_episodes,
            continuous_actions=False,  # or True, or handle each group separately
            seed=self.seed,
            device=self.config.sampling_device
        )()
        self.group_map = self.task.group_map(self.env)
        self.max_steps = self.task.max_steps(self.env)

        # 2) For each group => build a "group experiment stub", then create an Algorithm
        for group in self.group_map:
            stub = _GroupExperimentStub(
                eval_obj=self,
                group=group,
                model_config=self.group_model_configs[group],
                algorithm_config=self.group_algorithm_configs[group],
            )
            algo = self.group_algorithm_configs[group].get_algorithm(stub)
            self.group_algorithms[group] = algo

            loss_module, target_updater = algo.get_loss_and_updater(group)
            self.losses[group] = loss_module

            replay_buffer = algo.get_replay_buffer(group)
            self.replay_buffers[group] = replay_buffer

        self._build_joint_policy()
        self._setup_name()
        self._setup_logger()

    def _setup_name(self):
        self.algorithm_name = "eval"
        self.model_name = self.name
        self.environment_name = self.task.env_name().lower()
        self.task_name = self.task.name.lower()
        self._checkpointed_files = deque([])

        if self.config.save_folder is not None:
            # If the user specified a folder for the experiment we use that
            save_folder = Path(self.config.save_folder)
        else:
            # Otherwise, if the user is restoring from a folder, we will save in the folder they are restoring from
            if self.config.restore_file is not None:
                save_folder = Path(
                    self.config.restore_file
                ).parent.parent.parent.resolve()
            # Otherwise, the user is not restoring and did not specify a save_folder so we save in the hydra directory
            # of the experiment or in the directory where the experiment was run (if hydra is not used)
            else:
                if _has_hydra and HydraConfig.initialized():
                    save_folder = Path(HydraConfig.get().runtime.output_dir)
                else:
                    save_folder = Path(os.getcwd())

        if self.config.restore_file is None:
            self.name = generate_exp_name(
                f"{self.algorithm_name}_{self.model_name}", ""
            )
            self.folder_name = save_folder / self.name

        else:
            # If restoring, we use the name of the previous experiment
            self.name = Path(self.config.restore_file).parent.parent.resolve().name
            self.folder_name = save_folder / self.name

        if (
            len(self.config.loggers)
            or self.config.checkpoint_interval > 0
            or self.config.create_json
        ):
            self.folder_name.mkdir(parents=False, exist_ok=True)

    def _setup_logger(self):
        self.logger = Logger(
            project_name=self.config.project_name,
            experiment_name=self.name,
            folder_name=str(self.folder_name),
            experiment_config=self.config,
            algorithm_name=self.algorithm_name,
            model_name=self.model_name,
            environment_name=self.environment_name,
            task_name=self.task_name,
            group_map=self.group_map,
            seed=self.seed,
        )



    def _build_joint_policy(self):
        """
        Gathers sub-policies from each group, merges them into a single
        TensorDictSequential. This is the policy used for rollouts/evaluation.
        """
        from tensordict.nn import TensorDictSequential
        subpolicies = []
        for group, algo in self.group_algorithms.items():
            subpol = algo.get_policy_for_collection()  # e.g. with exploration off
            subpolicies.append(subpol)
        self.joint_policy = TensorDictSequential(*subpolicies)

    @torch.no_grad()
    def _evaluation_loop(self):
        evaluation_start = time.time()
        with set_exploration_type(
                ExplorationType.DETERMINISTIC
                if self.config.evaluation_deterministic_actions
                else ExplorationType.RANDOM
        ):
            if self.task.has_render(self.env) and self.config.render:
                video_frames = []

                def callback(env, td):
                    video_frames.append(
                        self.task.__class__.render_callback(self, env, td)
                    )

            else:
                video_frames = None
                callback = None

            seed_everything(self.seed)
            if self.env.batch_size == ():
                rollouts = []
                for eval_episode in range(self.config.evaluation_episodes):
                    td = self.env.rollout(
                        max_steps=self.max_steps,
                        policy=self.joint_policy,
                        callback=callback if eval_episode == 0 else None,
                        auto_cast_to_device=True,
                        break_when_any_done=True,
                    )
                    rollouts.append(td)
            else:
                td = self.env.rollout(
                    max_steps=self.max_steps,
                    policy=self.joint_policy,
                    callback=callback,
                    auto_cast_to_device=True,
                    break_when_any_done=False,
                )
                rollouts = list(td.unbind(0))

        evaluation_time = time.time() - evaluation_start
        self.logger.log(
            {"timers/evaluation_time": evaluation_time}, step=self.n_iters_performed
        )
        self.logger.log_evaluation(
            rollouts,
            video_frames=video_frames,
            step=self.n_iters_performed,
            total_frames=self.total_frames,
        )


    def load_state_dict(self, merged_dict: dict):
        """
        Merge partial sub-dicts for each group: e.g. "loss_predator" => self.losses["predator"].load_state_dict(...)
        or if your algorithms store them differently, adapt accordingly.
        """
        for group in self.group_map:
            # e.g. "loss_predator"
            lk = f"loss_{group}"
            if lk in merged_dict:
                self.losses[group].load_state_dict(merged_dict[lk])

            bk = f"buffer_{group}"
            if bk in merged_dict and merged_dict[bk] is not None:
                self.replay_buffers[group].load_state_dict(merged_dict[bk])

    def close(self):
        if self.env is not None:
            self.env.close()


class _GroupExperimentStub:
    """
    This is the "fake experiment" that your base Algorithm constructor references:

        def __init__(self, experiment):
            self.experiment = experiment
            # self.model_config = experiment.model_config
            # etc.

    We'll override those references for each group, so the group can have its own model_config,
    etc.
    """

    def __init__(
            self,
            eval_obj: MultiAlgorithmEvaluation,
            group: str,
            model_config: ModelConfig,
            algorithm_config: AlgorithmConfig
    ):
        self._eval_obj = eval_obj
        self._group = group
        self.model_config = model_config
        self.algorithm_config = algorithm_config

        self.config = eval_obj.config  # an ExperimentConfig
        self.device = self.config.train_device
        self.buffer_device = self.config.buffer_device
        self.group_map = {group: eval_obj.group_map[group]}
        self.observation_spec = eval_obj.task.observation_spec(eval_obj.env)
        self.action_spec = eval_obj.task.action_spec(eval_obj.env)
        self.state_spec = eval_obj.task.state_spec(eval_obj.env)
        self.action_mask_spec = eval_obj.task.action_mask_spec(eval_obj.env)
        self.on_policy = algorithm_config.on_policy()
        self.has_independent_critic = algorithm_config.has_independent_critic()
        self.has_centralized_critic = algorithm_config.has_centralized_critic()
        self.has_critic = algorithm_config.has_critic()
        self.critic_model_config = copy.deepcopy(self.model_config)
        self.critic_model_config.is_critic = True
        self.has_rnn = self.model_config.is_rnn or (
                self.critic_model_config.is_rnn and self.has_critic
        )
