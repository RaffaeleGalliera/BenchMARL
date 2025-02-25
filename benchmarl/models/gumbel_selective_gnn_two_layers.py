from dataclasses import MISSING, dataclass
import torch
import torch.nn.functional as F
from torch import nn

from torch_geometric.nn import Sequential

from tensordict import TensorDictBase
from tensordict.utils import _unravel_key_to_tuple
from .gumbel_selective_gnn import GumbelSelectiveGnn, GumbelSelectiveGnnConfig, _batch_from_dense_to_ptg_with_actions


class GumbelSelectiveGnnTwoLayers(GumbelSelectiveGnn):
    def __init__(
            self,
            _get_pos_from_features: bool = False,  ### NEW
            **kwargs
    ):
        super().__init__(**kwargs)
        self._get_pos_from_features = _get_pos_from_features
        gnn_class = kwargs.get("gnn_class", MISSING)
        if gnn_class is MISSING:
            raise ValueError("gnn_class must be provided")
        gnn_kwargs = kwargs.get("gnn_kwargs", {})
        gnn_kwargs["in_channels"] = self.output_features
        input_args = 'x, edge_index, edge_attr' if self.topology == 'from_pos' else 'x, edge_index'
        self.models = nn.ModuleList(
            [
                Sequential(
                    input_args,
                    [
                        (self.gnns[0], f'{input_args} -> x'),
                        nn.ReLU(inplace=True),
                        (gnn_class(**gnn_kwargs), f'{input_args} -> x'),
                    ]
                ).to(self.device)
                for _ in range(self.n_agents if not self.share_params else 1)
            ]
        )

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Collect the standard in_key features (excluding position_key and velocity_key):
        input_list = [
            tensordict.get(in_key)
            for in_key in self.in_keys
            if _unravel_key_to_tuple(in_key)[-1]
               not in (self.position_key, self.velocity_key)
        ]

        # Default pos, vel to None
        pos = None
        vel = None

        ########################################################################
        # 1) If we are NOT extracting position from the last channels,
        #    proceed to retrieve position/velocity from the tensordict.
        ########################################################################
        if not self._get_pos_from_features:
            # Retrieve pos from tensordict if position_key is set
            if self.position_key is not None:
                if self._full_position_key is None:
                    self._full_position_key = self._get_key_terminating_with(
                        list(tensordict.keys(True, True)), self.position_key
                    )
                    pos_ = tensordict.get(self._full_position_key)
                    if pos_.shape[-1] != self.pos_features - 1:
                        raise ValueError(
                            f"Position key in tensordict is {pos_.shape[-1]}-dim, "
                            f"while model was configured with pos_features={self.pos_features - 1}"
                        )
                else:
                    pos_ = tensordict.get(self._full_position_key)

                if not self.exclude_pos_from_node_features:
                    input_list.append(pos_)
                pos = pos_

            # Retrieve vel from tensordict if velocity_key is set
            if self.velocity_key is not None:
                if self._full_velocity_key is None:
                    self._full_velocity_key = self._get_key_terminating_with(
                        list(tensordict.keys(True, True)), self.velocity_key
                    )
                    vel_ = tensordict.get(self._full_velocity_key)
                    if vel_.shape[-1] != self.vel_features:
                        raise ValueError(
                            f"Velocity key in tensordict is {vel_.shape[-1]}-dim, "
                            f"while model was configured with vel_features={self.vel_features}"
                        )
                else:
                    vel_ = tensordict.get(self._full_velocity_key)

                input_list.append(vel_)
                vel = vel_

        # Concatenate input features
        input = torch.cat(input_list, dim=-1)
        batch_size = input.shape[:-2]

        # Flatten input for action head
        x = input.view(-1, input.shape[-1]).to(self.device)

        # Compute actions
        if self.group_action_replay_buffer_key in tensordict.keys(True, True):
            actions = tensordict.get(self.group_action_key)
            actions = actions.view(-1)
        else:
            action_logits = self.action_head(x)
            if self.is_critic:
                actions = action_logits.argmax(dim=-1)
            else:
                action_probs = F.gumbel_softmax(action_logits, tau=1, hard=True)
                actions = action_probs.argmax(dim=-1)  # Shape: [batch_size * n_agents]
                buffer_actions = actions.view(*batch_size, self.n_agents)
                tensordict.set(self.group_action_key, buffer_actions)

        # If we are *extracting pos from features*:
        if self._get_pos_from_features:
            original_features = tensordict.get(self.group_key, "observation")['observation']
            # e.g., we want the last 2 dims to be x,y position
            if original_features.shape[-1] < 2:
                raise ValueError(
                    "Original Features does not have enough feature dims to slice out (x,y) position."
                )
            pos = original_features[..., 0, 0, -2:]

        # Build the graph with both radius-based and action-based edges
        graph = _batch_from_dense_to_ptg_with_actions(
            x=input.to(self.device),
            edge_index=self.edge_index.to(self.device) if self.edge_index is not None else None,
            pos=pos.to(self.device) if pos is not None else None,
            vel=vel.to(self.device) if vel is not None else None,
            actions=actions,
            num_actions=self.num_actions,
            self_loops=self.self_loops,
            edge_radius=self.edge_radius,
            add_group_actions_to_node_features=self.add_group_actions_to_node_features
        )

        # Proceed with GNN forward pass
        forward_gnn_params = {
            "x": graph.x,
            "edge_index": graph.edge_index,
        }
        if (
            (self.position_key is not None or self.velocity_key is not None)
            and self.gnn_supports_edge_attrs
            and graph.edge_attr is not None
        ):
            forward_gnn_params.update({"edge_attr": graph.edge_attr})

        if not self.share_params:
            if not self.centralised:
                res = torch.stack(
                    [
                        gnn(**forward_gnn_params).view(
                            *batch_size,
                            self.n_agents,
                            self.output_features,
                        )[..., i, :]
                        for i, gnn in enumerate(self.models)
                    ],
                    dim=-2,
                )
            else:
                res = torch.stack(
                    [
                        gnn(**forward_gnn_params)
                        .view(
                            *batch_size,
                            self.n_agents,
                            self.output_features,
                        )
                        .mean(dim=-2)  # Mean pooling
                        for i, gnn in enumerate(self.models)
                    ],
                    dim=-2,
                )
        else:
            res = self.models[0](**forward_gnn_params).view(
                *batch_size, self.n_agents, self.output_features
            )
            if self.centralised:
                res = res.mean(dim=-2)  # Mean pooling

        tensordict.set(self.out_key, res)
        return tensordict


@dataclass
class GumbelSelectiveGnnTwoLayersConfig(GumbelSelectiveGnnConfig):
    _get_pos_from_features: bool = False

    @staticmethod
    def associated_class():
        return GumbelSelectiveGnnTwoLayers
