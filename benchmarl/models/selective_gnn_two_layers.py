from dataclasses import MISSING, dataclass
from typing import Optional, Type

import torch
import torch.nn.functional as F
import torch_geometric
from torch import nn, Tensor
from torch_geometric.nn import Sequential

from . import ModelConfig
from .selective_gnn import SelectiveGnn, _batch_from_dense_to_ptg_with_actions
from tensordict import TensorDictBase
from tensordict.utils import _unravel_key_to_tuple


class SelectiveGnnTwoLayers(SelectiveGnn):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
        input_features = [
            tensordict.get(in_key)
            for in_key in self.in_keys
            if _unravel_key_to_tuple(in_key)[-1]
               not in (self.position_key, self.velocity_key)
        ]

        # Retrieve position
        if self.position_key is not None:
            if self._full_position_key is None:
                self._full_position_key = self._get_key_terminating_with(
                    list(tensordict.keys(True, True)), self.position_key
                )
            pos = tensordict.get(self._full_position_key)
            if not self.exclude_pos_from_node_features:
                input_features.append(pos)
        else:
            pos = None

        # Retrieve velocity
        if self.velocity_key is not None:
            if self._full_velocity_key is None:
                self._full_velocity_key = self._get_key_terminating_with(
                    list(tensordict.keys(True, True)), self.velocity_key
                )
            vel = tensordict.get(self._full_velocity_key)
            input_features.append(vel)
        else:
            vel = None

        # Concatenate input features
        input = torch.cat(input_features, dim=-1)
        batch_size = input.shape[:-2]

        # Flatten input for action head
        x = input.view(-1, input.shape[-1]).to(self.device)

        # Compute actions
        action_logits = self.group_action_head(x)
        action_probs = F.gumbel_softmax(action_logits, tau=1, hard=True)
        actions = action_probs.argmax(dim=-1)  # Shape: [batch_size * n_agents]

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
            add_group_actions_to_node_features=self.add_group_actions_to_node_features,
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
class SelectiveGnnTwoLayersConfig(ModelConfig):
    """Dataclass config for a SelectiveGNN."""

    topology: str = MISSING
    self_loops: bool = MISSING

    gnn_class: Type[torch_geometric.nn.MessagePassing] = MISSING
    gnn_kwargs: Optional[dict] = None

    position_key: Optional[str] = None
    pos_features: Optional[int] = 0
    velocity_key: Optional[str] = None
    vel_features: Optional[int] = 0
    exclude_pos_from_node_features: Optional[bool] = None
    edge_radius: Optional[float] = None

    num_actions: int = 4  # Number of discrete actions for each node
    add_group_actions_to_node_features: bool = False # Add group actions to node features

    @staticmethod
    def associated_class():
        return SelectiveGnnTwoLayers