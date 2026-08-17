"""Skip-connection module integrating GCN Attention (Sec. III-C, Eq. 6-8).

Projects the fused Global+Local feature map into a graph with a learnable
number of nodes N, runs graph convolution to capture long-range node
relations, then projects back into the spatial domain and adds it as the
skip-connection detail.
"""
import torch
import torch.nn as nn


class GCNAttention(nn.Module):
    def __init__(self, channels, num_nodes=32):
        """Builds the pixel->graph projection, graph convolution parameters,
        and the graph->pixel projection.

        Hyperparameters:
            channels (int): feature channel width of the skip connection this
                module is attached to (varies per decoder stage: 16, 16, 8,
                8, 4 in the final lightweight config).
            num_nodes (int, default=32): number of graph nodes N that the
                H*W pixels get soft-assigned into. This is the single biggest
                lever on this module's parameter count and its ability to
                capture long-range relations -- fewer nodes = fewer
                parameters but coarser graph reasoning. Reduced to N=8 in the
                final lightweight configuration.
        """
        super().__init__()
        self.num_nodes = num_nodes
        # X -> G : learnable projection from pixels to graph nodes
        self.to_graph = nn.Conv2d(channels, num_nodes, kernel_size=1)
        # graph convolution parameters (Eq. 7): adjacency Ag and feature Wg
        self.adjacency = nn.Parameter(torch.eye(num_nodes) + 0.01 * torch.randn(num_nodes, num_nodes))
        self.node_transform = nn.Linear(channels, channels)
        # G -> X : remap graph back to spatial domain (Eq. 8)
        self.from_graph = nn.Conv2d(num_nodes, channels, kernel_size=1)

    def forward(self, x):
        """Projects features to a graph, runs graph convolution, projects
        back to spatial form, and adds the result as a residual.

        Parameters:
            x (torch.Tensor): skip-connection feature map from the encoder,
                shape (B, channels, H, W).

        Returns:
            torch.Tensor of shape (B, channels, H, W): the input features
            plus GCN-refined detail (residual connection), used as the
            decoder's skip input.
        """
        b, c, h, w = x.shape
        assign = self.to_graph(x).flatten(2)              # (B, N, HW)
        assign = torch.softmax(assign, dim=-1)             # soft assignment of pixels to nodes
        feat_flat = x.flatten(2).transpose(1, 2)           # (B, HW, C)
        graph_nodes = torch.bmm(assign, feat_flat)         # (B, N, C)  -- X projected to graph space

        # graph convolution: G = (I - Ag) @ G @ Wg  (residual form of Eq. 7)
        adj = self.adjacency.unsqueeze(0).expand(b, -1, -1)
        graph_out = torch.bmm(adj, graph_nodes)
        graph_out = self.node_transform(graph_out)
        graph_out = torch.relu(graph_out)

        # remap graph back to coordinate space (Eq. 8) via transposed soft assignment
        spatial = torch.bmm(assign.transpose(1, 2), graph_out)  # (B, HW, C)
        spatial = spatial.transpose(1, 2).view(b, c, h, w)
        return x + spatial  # residual: detail features added at the skip connection

"""
             Encoder Feature
             (C × H × W)
                   │
                   ↓
          ┌─────────────────┐
          │  Pixel → Graph  │
          │    Projection   │
          └─────────────────┘
                   │
                   ↓
            N graph nodes
             (N = 32)
                   │
                   ↓
          ┌─────────────────┐
          │ Graph Convolution│
          │                 │
          │ Learn relations │
          │ between nodes   │
          └─────────────────┘
                   │
                   ↓
          Updated graph nodes
                   │
                   ↓
          ┌─────────────────┐
          │ Graph → Spatial │
          │   Projection    │
          └─────────────────┘
                   │
                   ↓
           Spatial features
                   │
                   ↓
        Original + GCN features
                   │
                   ↓
             Skip Connection
                   │
                   ↓
                Decoder
"""
