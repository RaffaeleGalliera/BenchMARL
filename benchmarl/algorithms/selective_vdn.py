from dataclasses import dataclass
from typing import Type

from torchrl.objectives import LossModule, QMixerLoss, ValueEstimators

from benchmarl.algorithms.vdn import Vdn, VdnConfig
from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
import warnings
import torch
from typing import Optional, Sequence, Union, Tuple, Dict
from tensordict import TensorDictBase
from torchrl.modules import SafeModule
from torchrl.data import Composite, TensorSpec, Unbounded, Categorical
from torchrl.envs.utils import exploration_type, ExplorationType
from tensordict.nn import (
    dispatch,
    TensorDictModule,
    TensorDictModuleBase,
    TensorDictSequential,
)
from tensordict.utils import expand_as_right, NestedKey


def _process_action_space_spec(action_space: Optional[str], spec: Optional[TensorSpec]):
    """Helper function, same as done in QValueModule, to unify action_space vs. spec usage."""
    if spec is not None:
        # If spec is provided, deduce action_space from spec, or confirm they're consistent
        # For simplicity, we just return (action_space, spec).
        # In QValueModule, there's more logic. Adjust as needed.
        return action_space, spec
    else:
        # If no spec, we rely on action_space.
        return action_space, None


class TwoHeadQValueModule(TensorDictModuleBase):
    """
    A specialized Q-Value module that handles two discrete Q-value heads:
      1) A "group action" Q-value
      2) An "environment action" Q-value

    It applies a similar logic as QValueModule to each set of Q-values:
      - Optional masking
      - Argmax or one-hot action format (depending on action_space)
      - Gathers the chosen action-value
    Finally, it sums group-chosen-action-value + env-chosen-action-value into a
    final "chosen_action_value".

    By default, the output keys are:
      [ group_action, group_action_value, group_chosen_action_value,
        action, action_value, env_chosen_action_value,
        chosen_action_value ]

    Args:
        group_action_space (str): The group action space type,
            one of "one_hot", "mult_one_hot", "binary", "categorical".
        group_action_value_key (NestedKey): Key for the group action Q-values.
            Defaults to ("agent", "group_action_value").
        group_action_mask_key (NestedKey, optional): Key for a boolean mask for
            the group action Q-values. If provided, unmasked entries remain
            unchanged, masked entries are set to -inf for argmax.
        env_action_space (str): The env action space type,
            one of "one_hot", "mult_one_hot", "binary", "categorical".
        env_action_value_key (NestedKey): Key for the environment action Q-values.
            Defaults to ("agent", "action_value").
        env_action_mask_key (NestedKey, optional): Key for a boolean mask for
            the env action Q-values.
        out_keys (list of NestedKey, optional): The 7 output keys
            representing
             1) group_action,
             2) group_action_value,
             3) group_chosen_action_value,
             4) action,
             5) action_value,
             6) env_chosen_action_value,
             7) chosen_action_value
            in that order by default.
        group_var_nums (int, optional): if `group_action_space="mult_one_hot"`,
            cardinalities for each sub-action dimension.
        env_var_nums (int, optional): if `env_action_space="mult_one_hot"`,
            cardinalities for each sub-action dimension.
        spec (TensorSpec, optional): optional spec for bounding or masking the final output.
            Usually not strictly needed, but you can define it if you want SafeModule checks.
        safe (bool): if True, output will be checked and projected onto `spec`.
                     Default = False.

    Usage:
    - Expects that the tensordict has
        (group_action_value_key) -> shape [*batch, group_dim]
        (env_action_value_key)   -> shape [*batch, action_dim]
      Possibly with boolean masks if needed.

    - On forward:
        1) Picks group_action with argmax or one-hot,
           gathers group_chosen_action_value
        2) Picks env_action with argmax or one-hot,
           gathers env_chosen_action_value
        3) Sums them => chosen_action_value
        4) Writes them to the out_keys in the tensordict
        5) Returns the updated tensordict

    """

    def __init__(
        self,
        model_module: TensorDictModule,
        group_action_space: str = "categorical",
        group_action_value_key: NestedKey = ("agent", "group_action_value"),
        group_action_mask_key: Optional[NestedKey] = None,
        env_action_space: str = "categorical",
        env_action_value_key: NestedKey = ("agent", "action_value"),
        env_action_mask_key: Optional[NestedKey] = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
        group_var_nums: Optional[int] = None,
        env_var_nums: Optional[int] = None,
        spec: Optional[TensorSpec] = None,
        safe: bool = False,
    ):
        # Process the action_space vs. spec logic
        # (like QValueModule does, but we do it for each head).
        group_action_space, group_spec = _process_action_space_spec(
            group_action_space, None
        )
        env_action_space, env_spec = _process_action_space_spec(
            env_action_space, None
        )

        self.group_action_space = group_action_space
        self.env_action_space = env_action_space
        self.group_var_nums = group_var_nums
        self.env_var_nums = env_var_nums

        # For each head, define the function mappings
        self.action_func_mapping = {
            "one_hot": self._one_hot,
            "mult_one_hot": self._mult_one_hot,
            "binary": self._binary,
            "categorical": self._categorical,
        }
        self.action_value_func_mapping = {
            "categorical": self._categorical_action_value,
        }

        # Validate group_action_space
        if group_action_space not in self.action_func_mapping:
            raise ValueError(
                f"group_action_space must be one of {list(self.action_func_mapping.keys())}, "
                f"got {group_action_space}"
            )
        # Validate env_action_space
        if env_action_space not in self.action_func_mapping:
            raise ValueError(
                f"env_action_space must be one of {list(self.action_func_mapping.keys())}, "
                f"got {env_action_space}"
            )

        self.group_action_value_key = group_action_value_key
        self.group_action_mask_key = group_action_mask_key
        self.env_action_value_key = env_action_value_key
        self.env_action_mask_key = env_action_mask_key

        # Build in_keys
        in_keys = [group_action_value_key, env_action_value_key]
        if group_action_mask_key is not None:
            in_keys.append(group_action_mask_key)
        if env_action_mask_key is not None:
            in_keys.append(env_action_mask_key)
        self.in_keys = in_keys
        # For output, default 7 keys:
        #   1) group_action
        #   2) group_action_value
        #   3) group_chosen_action_value
        #   4) action
        #   5) action_value
        #   6) env_chosen_action_value
        #   7) chosen_action_value
        if out_keys is None:
            out_keys = [
                ("agent", "group_action"),
                ("agent", "group_action_value"),
                ("agent", "group_chosen_action_value"),
                ("agent", "action"),
                ("agent", "action_value"),
                ("agent", "env_chosen_action_value"),
                ("agent", "chosen_action_value"),
            ]
        if len(out_keys) != 7:
            raise ValueError(
                "TwoHeadQValueModule expects exactly 7 out_keys in [group_action, group_action_value, "
                "group_chosen_action_value, action, action_value, env_chosen_action_value, chosen_action_value]. "
                f"Got: {out_keys}"
            )

        self.out_keys = out_keys
        super().__init__()

        # If you'd like to define a Composite spec for your out_keys, do so here
        # For example:
        action_key_g = out_keys[0]
        action_key_e = out_keys[3]
        final_key = out_keys[-1]

        if not isinstance(spec, Composite):
            # We'll create a minimal composite spec for demonstration:
            spec = Composite(
                {
                    action_key_g: Unbounded(dtype=torch.long),
                    action_key_e: Unbounded(dtype=torch.long),
                    final_key: Unbounded(),
                }
            )
        self.register_spec(safe=safe, spec=spec)
        self.model_module = model_module


    # same approach as in QValueModule
    register_spec = SafeModule.register_spec

    # =============== Utilities (like QValueModule) ===============

    @staticmethod
    def _one_hot(value: torch.Tensor) -> torch.Tensor:
        # Argmax, then one-hot
        out = (value == value.max(dim=-1, keepdim=True)[0]).to(torch.long)
        return out

    @staticmethod
    def _categorical(value: torch.Tensor) -> torch.Tensor:
        # Argmax (long)
        return torch.argmax(value, dim=-1).to(torch.long)

    def _mult_one_hot(self, value: torch.Tensor, var_nums: int) -> torch.Tensor:
        if var_nums is None:
            raise ValueError(
                "var_nums must be provided for multi one-hot action spaces."
            )
        values = value.split(var_nums, dim=-1)
        ohs = []
        for v in values:
            ohs.append(self._one_hot(v))
        return torch.cat(ohs, -1)

    @staticmethod
    def _binary(value: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Binary action space not yet implemented.")

    @staticmethod
    def _default_action_value(values: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # For one_hot or multi_one_hot
        return (action * values).sum(-1, keepdim=True)

    @staticmethod
    def _categorical_action_value(values: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # For categorical: gather
        return values.gather(-1, action.unsqueeze(-1))

    # =============== Forward Helper ===============
    def _select_action_and_value(
        self,
        values: torch.Tensor,
        mask: Optional[torch.Tensor],
        action_space: str,
        var_nums: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given Q-values 'values' (shape [*batch, n_actions]) and an optional bool mask,
        (1) applies the mask (set to -inf),
        (2) picks the action (argmax or one-hot),
        (3) gathers the chosen action value,
        (4) returns (action, chosen_action_value).
        """
        if mask is not None:
            # mask == True => allowed. mask == False => disallowed
            # We'll set disallowed to -inf
            values = torch.where(mask, values, torch.finfo(values.dtype).min)

        # pick an action
        if action_space not in self.action_func_mapping:
            raise ValueError(f"Unsupported action_space: {action_space}")

        if action_space == "mult_one_hot":
            # for multi-one-hot, we pass var_nums
            action = self._mult_one_hot(values, var_nums=var_nums)
            action_value_func = self._default_action_value
        else:
            # one_hot, categorical, or binary
            action = self.action_func_mapping[action_space](values)
            action_value_func = self.action_value_func_mapping.get(
                action_space, self._default_action_value
            )

        chosen_value = action_value_func(values, action)
        return action, chosen_value

    # =============== Forward ===============
    @dispatch(auto_batch_size=False)
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Group Q-values. Note that these are group action values!
        group_q = tensordict.get(self.group_action_value_key)
        # Optionally get group mask
        group_mask = (tensordict.get(self.group_action_mask_key, None)
                      if self.group_action_mask_key else None)

        # 1) Pick group action + group chosen Q
        group_action, group_chosen_q = self._select_action_and_value(
            group_q, group_mask, self.group_action_space, var_nums=self.group_var_nums
        )

        # Set the chosen group action in tensordict and recompute the Values
        tensordict.set(self.out_keys[0], group_action)
        tensordict = self.model_module(tensordict)

        # Env Q-values
        env_q = tensordict.get(self.env_action_value_key)
        # Optionally get env mask
        env_mask = (tensordict.get(self.env_action_mask_key, None)
                    if self.env_action_mask_key else None)

        # 2) Pick env action + env chosen Q
        env_action, env_chosen_q = self._select_action_and_value(
            env_q, env_mask, self.env_action_space, var_nums=self.env_var_nums
        )

        # 3) Sum them => final chosen_action_value
        # TODO: check if we need this sum or just get the env_chosen_q
        chosen_q = group_chosen_q + env_chosen_q

        # 4) Write them into tensordict
        # out_keys = [ grp_act, grp_q, grp_chosen_q, act, action_q, env_chosen_q, chosen_q ]
        tensordict.set(self.out_keys[0], group_action)
        tensordict.set(self.out_keys[1], group_q)
        tensordict.set(self.out_keys[2], group_chosen_q)
        tensordict.set(self.out_keys[3], env_action)
        tensordict.set(self.out_keys[4], env_q)
        tensordict.set(self.out_keys[5], env_chosen_q)
        tensordict.set(self.out_keys[6], chosen_q)

        return tensordict


class DualEGreedyModule(TensorDictModuleBase):
    """
    Dual epsilon-greedy module that updates two discrete action keys (one for group and one for environment)
    with random samples drawn from their respective TensorSpecs with probability eps.

    This implementation closely mirrors the original EGreedyModule implementation.

    Args:
        group_action_key (NestedKey): the key where the group action is stored in the tensordict.
        env_action_key (NestedKey): the key where the environment action is stored.
        group_action_spec (TensorSpec): the spec used to sample group actions.
        env_action_spec (TensorSpec): the spec used to sample environment actions.
        eps_init (float, optional): initial epsilon value (default 1.0).
        eps_end (float, optional): final epsilon value (default 0.1).
        annealing_num_steps (int, optional): number of steps over which to anneal epsilon (default 1000).
        group_action_mask_key (NestedKey, optional): key for an optional boolean mask on group actions.
        env_action_mask_key (NestedKey, optional): key for an optional boolean mask on environment actions.
    """

    def __init__(
            self,
            model_module: TensorDictModule,
            group_action_key: NestedKey,
            env_action_key: NestedKey,
            group_action_spec: TensorSpec,
            env_action_spec: TensorSpec,
            eps_init: float = 1.0,
            eps_end: float = 0.1,
            annealing_num_steps: int = 1000,
            group_action_mask_key: Optional[NestedKey] = None,
            env_action_mask_key: Optional[NestedKey] = None,
    ):
        if not isinstance(eps_init, float):
            warnings.warn("eps_init should be a float.")
        if eps_end > eps_init:
            raise RuntimeError("eps should decrease over time or be constant")

        self.group_action_key = group_action_key
        self.env_action_key = env_action_key
        self.group_action_spec = group_action_spec
        self.env_action_spec = env_action_spec
        self.group_action_mask_key = group_action_mask_key
        self.env_action_mask_key = env_action_mask_key
        in_keys = [self.env_action_key, self.group_action_key]
        if self.env_action_mask_key is not None:
            in_keys.append(self.env_action_mask_key)
        if self.group_action_mask_key is not None:
            in_keys.append(self.group_action_mask_key)
        self.in_keys = in_keys
        self.out_keys = [self.env_action_key, self.group_action_key]

        super().__init__()

        self.register_buffer("eps_init", torch.as_tensor([eps_init]))
        self.register_buffer("eps_end", torch.as_tensor([eps_end]))
        self.register_buffer("eps", torch.as_tensor([eps_init], dtype=torch.float32))
        self.annealing_num_steps = annealing_num_steps
        self.model_module = model_module

    def step(self, frames: int = 1) -> None:
        """Update epsilon by decaying it for a given number of frames."""
        for _ in range(frames):
            self.eps.data[0] = max(
                self.eps_end.item(),
                (self.eps - (self.eps_init - self.eps_end) / self.annealing_num_steps).item(),
            )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if exploration_type() == ExplorationType.RANDOM or exploration_type() is None:
            if isinstance(self.group_action_key, tuple) and len(self.group_action_key) > 1:
                group_action_tensordict = tensordict.get(self.group_action_key[:-1])
                group_action_key = self.group_action_key[-1]
            else:
                group_action_tensordict = tensordict
                group_action_key = self.group_action_key

            if isinstance(self.env_action_key, tuple) and len(self.env_action_key) > 1:
                env_action_tensordict = tensordict.get(self.env_action_key[:-1])
                env_action_key = self.env_action_key[-1]
            else:
                env_action_tensordict = tensordict
                env_action_key = self.env_action_key

            # --- Process group actions ---
            group_out = group_action_tensordict.get(group_action_key).clone()
            eps_val = self.eps.item()
            # Create a condition tensor for group actions
            cond_group = torch.rand(group_action_tensordict.shape, device=group_out.device) < eps_val
            # Clone the spec so that any mask updates don't affect the original spec.
            cond_group = expand_as_right(cond_group, group_out)
            group_spec = self.group_action_spec.clone()
            if self.group_action_mask_key is not None:
                group_mask = tensordict.get(self.group_action_mask_key, None)
                if group_mask is None:
                    raise KeyError(f"Group action mask key {self.group_action_mask_key} not found in tensordict.")
                group_spec.update_mask(group_mask)
            # Expand the spec if necessary.
            if group_spec.shape != group_out.shape:
                group_spec = group_spec.expand(group_out.shape)
            random_group = group_spec.rand().to(group_out.device)
            new_group = torch.where(cond_group, random_group, group_out)

            # Apply the new group actions to the tensordict and rerun the model
            # TODO double check if we want to rerun also the entire TwoHeadQValueModule after changing the group actions
            tensordict.set(self.group_action_key, new_group)
            tensordict = self.model_module(tensordict)

            # --- Process environment actions ---
            env_out = env_action_tensordict.get(env_action_key).clone()
            cond_env = torch.rand(env_action_tensordict.shape, device=env_out.device) < eps_val
            env_spec = self.env_action_spec.clone()
            if self.env_action_mask_key is not None:
                env_mask = tensordict.get(self.env_action_mask_key, None)
                if env_mask is None:
                    raise KeyError(f"Env action mask key {self.env_action_mask_key} not found in tensordict.")
                env_spec.update_mask(env_mask)
            if env_spec.shape != env_out.shape:
                env_spec = env_spec.expand(env_out.shape)
            random_env = env_spec.rand().to(env_out.device)
            new_env = torch.where(cond_env, random_env, env_out)
            env_action_tensordict.set(env_action_key, new_env)
        return tensordict

class SelectiveVdn(Vdn):
    """
    A VDN variant that handles two discrete actions per agent:
      (agent, "group_action") and (agent, "action").
    It uses a "SelectiveGNN" or similar model to produce (group_action_value, action_value).
    Then a "TwoHeadQValueModule" to pick or reconstruct the final chosen actions.
    Finally, we sum them or store the final local Q in (chosen_action_value) for the VDN mixer.

    The big difference from classical Vdn is we override:
      - process_batch to ensure (group_action) is in replay.
      - _get_policy_for_loss to build the new model pipeline.
      - _get_policy_for_collection to incorporate DualEGreedy on both group_action and action.

    Otherwise, the standard QMixerLoss from TorchRL can be used as-is, as we just store
    local_value = (group, "chosen_action_value").
    """

    def __init__(self, delay_value: bool, loss_function: str, **kwargs):
        super().__init__(delay_value=delay_value, loss_function=loss_function, **kwargs)
        self.model_module = None
        self.num_groups = 4 # TODO: set this from config file
        self.group_action_spec = Composite(
            {
                "group_action": Categorical(n=self.num_groups),
            }
        )
        return


    def _check_specs(self):
        pass

    ########################
    # Overriden from Vdn
    ########################

    def _get_policy_for_loss(
        self, group: str, model_config: AlgorithmConfig, continuous: bool
    ) -> TensorDictModule:
        """
        Build the train-time policy that:
          1) Runs the model (SelectiveGNN) to produce (group_action_value, action_value)
          2) Runs a TwoHeadQValueModule that picks out or reconstructs (group_action, action)
             from the replay, and sums them into (chosen_action_value).
        This is used by QMixerLoss.
        """
        n_agents = len(self.group_map[group])
        logits_shape = [*self.action_spec[group, "action"].shape, self.action_spec[group, "action"].space.n, ]
        # We'll define an input_spec for the model
        actor_input_spec = Composite(
            {group: self.observation_spec[group].clone().to(self.device)}
        )
        # We'll define an output_spec that expects the model to fill:
        #   (group_action_value) -> [n_agents, num_groups]
        #   (action_value)       -> [n_agents, num_env_actions]
        actor_output_spec = Composite(
            {
                group: Composite(
                    {
                        "action_value": Unbounded(shape=logits_shape),
                    },
                    shape=(n_agents,),
                )
            }
        )

        # 1) The model (SelectiveGNN, or any you have) that outputs group_action_value + action_value
        model_module = model_config.get_model(
            input_spec=actor_input_spec,
            output_spec=actor_output_spec,
            agent_group=group,
            input_has_agent_dim=True,
            n_agents=n_agents,
            centralised=False,
            share_params=self.experiment_config.share_policy_params,
            device=self.device,
            action_spec=self.action_spec,
        )

        self.model_module = model_module

        if self.action_mask_spec is not None:
            action_mask_key = (group, "action_mask")
        else:
            action_mask_key = None

        # 2) Then a TwoHeadQValueModule that merges them into final (chosen_action_value)
        # TODO: add the action_mask_key
        two_head_q = TwoHeadQValueModule(
            model_module=model_module,
            group_action_value_key=(group, "group_action_value"),
            env_action_value_key=(group, "action_value"),
            env_action_mask_key=action_mask_key,
            out_keys=[
                (group, "group_action"),
                (group, "group_action_value"),
                (group, "group_chosen_action_value"),
                (group, "action"),
                (group, "action_value"),
                (group, "env_chosen_action_value"),
                (group, "chosen_action_value"),  # local Q
            ],
        )

        # Put them together in a sequential policy
        policy_for_loss = TensorDictSequential(model_module, two_head_q)
        return policy_for_loss

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        """
        Build the data-collection policy.
        We wrap `policy_for_loss` in a `DualEGreedyModule` so that both
        (group_action) and (action) are randomly perturbed with probability eps.
        """
        if self.action_mask_spec is not None:
            action_mask_key = (group, "action_mask")
        else:
            action_mask_key = None

        # We'll apply a custom EGreedy on both (agent, "group_action") and (agent, "action").
        #TODO Add the action_mask_key

        dual_egreedy = DualEGreedyModule(
            model_module=self.model_module,
            group_action_key=(group, "group_action"),
            env_action_key=(group, "action"),
            env_action_mask_key=action_mask_key,
            group_action_spec=self.group_action_spec["group_action"],
            env_action_spec=self.action_spec[group, "action"],
            eps_init=self.experiment_config.exploration_eps_init,
            eps_end=self.experiment_config.exploration_eps_end,
            annealing_num_steps=self.experiment_config.get_exploration_anneal_frames(
                self.on_policy
            ),
        )

        return TensorDictSequential(*policy_for_loss, dual_egreedy)

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        """
        Ensures we have (group_action) in the replay and calls parent to set "reward", "done", etc.
        """
        batch = super().process_batch(group, batch)

        if (group, "group_action") not in batch.keys(True, True):
            raise RuntimeError(f"Missing {group}, group_action in replay batch!")
        return batch

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        """
        Standard QMixerLoss, but we read local_value=(group, 'chosen_action_value')
        so that it sums over agents for the global Q.
        """
        if continuous:
            raise NotImplementedError("SelectiveVDN is not for continuous actions.")
        loss_module = QMixerLoss(
            policy_for_loss,
            self.get_mixer(group),
            delay_value=self.delay_value,
            loss_function=self.loss_function,
            action_space=self.action_spec[group, "action"],
        )
        loss_module.set_keys(
            reward="reward",
            action=(group, "action"),
            done="done",
            terminated="terminated",
            action_value=(group, "action_value"),
            local_value=(group, "chosen_action_value"),  # sum of groupQ + envQ
            global_value="chosen_action_value",
            priority="td_error",
        )
        loss_module.make_value_estimator(
            ValueEstimators.TD0, gamma=self.experiment_config.gamma
        )
        return loss_module, True


@dataclass
class SelectiveVdnConfig(VdnConfig):
    """
    Extends VdnConfig with additional references for group size, etc.
    """
    num_groups = 4

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return SelectiveVdn  # Or return the actual class object.
