from dataclasses import MISSING, dataclass
from typing import Optional, Type

import torch
import torch.nn.functional as F
import torch_geometric
from torch import nn, Tensor

from . import ModelConfig
from .gnn import Gnn, _batch_from_dense_to_ptg, _RelVel
from tensordict import TensorDictBase
from tensordict.utils import _unravel_key_to_tuple

def _batch_from_dense_to_ptg_with_actions(
    x: Tensor,
    edge_index: Optional[Tensor],
    self_loops: bool,
    pos: Optional[Tensor] = None,
    vel: Optional[Tensor] = None,
    actions: Optional[Tensor] = None,
    num_actions: int = 4,
    edge_radius: Optional[float] = None,
) -> torch_geometric.data.Batch:
    batch_size, n_agents, input_dim = x.shape
    total_nodes = batch_size * n_agents

    # Flatten node features and optional inputs
    x = x.view(total_nodes, input_dim)
    if pos is not None:
        pos = pos.view(total_nodes, -1)
    if vel is not None:
        vel = vel.view(total_nodes, -1)

    # Batch indexing for nodes
    batch_indices = torch.arange(batch_size, device=x.device).repeat_interleave(n_agents)

    # Create initial graphs
    graphs = torch_geometric.data.Batch()
    graphs.ptr = torch.arange(0, (batch_size + 1) * n_agents, n_agents, device=x.device)
    graphs.batch = batch_indices
    graphs.x = x
    graphs.pos = pos
    graphs.vel = vel

    # Handle edge construction
    if edge_index is not None:
        # Replicate edge indices for each graph in the batch
        edge_index_offset = (
            torch.arange(batch_size, device=x.device).repeat_interleave(edge_index.shape[1]) * n_agents
        )
        edge_index = edge_index.repeat(1, batch_size) + edge_index_offset
        graphs.edge_index = edge_index
    else:
        if pos is None:
            raise RuntimeError("from_pos topology needs positions as input")
        graphs.edge_index = torch_geometric.nn.pool.radius_graph(
            pos, batch=graphs.batch, r=edge_radius, loop=self_loops
        )

    # Add action-based edges
    if actions is not None:
        actions = actions.view(-1)  # Flatten actions
        group = batch_indices * num_actions + actions  # Global group ID

        # Get unique groups and inverse indices
        unique_groups, inverse_indices = group.unique(return_inverse=True)

        # Sort nodes by group
        sorted_indices = torch.argsort(inverse_indices)
        sorted_inverse_indices = inverse_indices[sorted_indices]

        # Compute group sizes and offsets
        group_sizes = torch.bincount(sorted_inverse_indices)
        group_offsets = torch.cumsum(torch.cat([torch.tensor([0], device=x.device), group_sizes[:-1]]), dim=0)

        # Precompute total number of edges
        if self_loops:
            num_edges_per_group = group_sizes * group_sizes
        else:
            num_edges_per_group = group_sizes * (group_sizes - 1)
        total_num_edges = num_edges_per_group.sum().item()

        # Preallocate edge index tensors
        idx_i = torch.empty(total_num_edges, dtype=torch.long, device=x.device)
        idx_j = torch.empty(total_num_edges, dtype=torch.long, device=x.device)

        ptr = 0
        for i in range(len(group_sizes)):
            start = group_offsets[i].item()
            end = start + group_sizes[i].item()
            nodes = sorted_indices[start:end]
            n = nodes.size(0)
            if n > 1 or (n == 1 and self_loops):
                idx_i_g = nodes.repeat_interleave(n)
                idx_j_g = nodes.repeat(n)
                if not self_loops:
                    mask = idx_i_g != idx_j_g
                    idx_i_g = idx_i_g[mask]
                    idx_j_g = idx_j_g[mask]
                num_edges = idx_i_g.size(0)
                idx_i[ptr:ptr + num_edges] = idx_i_g
                idx_j[ptr:ptr + num_edges] = idx_j_g
                ptr += num_edges

        # Truncate idx_i and idx_j in case some groups have n <=1
        idx_i = idx_i[:ptr]
        idx_j = idx_j[:ptr]

        if idx_i.numel() > 0:
            edge_index_action = torch.stack([idx_i, idx_j], dim=0)
            graphs.edge_index = torch.cat([graphs.edge_index, edge_index_action], dim=1)

    # Apply graph transforms for edge attributes
    if pos is not None:
        graphs.edge_attr = None  # Reset edge_attr before recomputing
        graphs = torch_geometric.transforms.Cartesian(norm=False)(graphs)
        graphs = torch_geometric.transforms.Distance(norm=False)(graphs)
    if vel is not None:
        graphs = _RelVel()(graphs)

    return graphs


class SelectiveGnn(Gnn):
    def __init__(self, num_actions: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.num_actions = num_actions
        # Action head (for groups)
        self.action_head = nn.Linear(self.input_features, self.num_actions).to(self.device)

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
        action_logits = self.action_head(x)
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
                        for i, gnn in enumerate(self.gnns)
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
                        for i, gnn in enumerate(self.gnns)
                    ],
                    dim=-2,
                )
        else:
            res = self.gnns[0](**forward_gnn_params).view(
                *batch_size, self.n_agents, self.output_features
            )
            if self.centralised:
                res = res.mean(dim=-2)  # Mean pooling

        tensordict.set(self.out_key, res)
        return tensordict


@dataclass
class SelectiveGnnConfig(ModelConfig):
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

    @staticmethod
    def associated_class():
        return SelectiveGnn