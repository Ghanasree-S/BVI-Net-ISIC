"""Skip-connection module integrating GCN Attention (Sec. III-C, Eq. 6-8).

Projects the fused Global+Local feature map into a graph with a learnable number
of nodes N, runs graph convolution to capture long-range node relations, then
projects back into the spatial domain and adds it as the skip-connection detail.
"""
import torch
import torch.nn as nn


class GCNAttention(nn.Module):
    def __init__(self, channels, num_nodes=32):
        super().__init__()
        self.num_nodes = num_nodes
        # X -> G : learnable projection from pixels to graph nodes
        self.to_graph = nn.Conv2d(channels, num_nodes, kernel_size=1)
        # graph convolution parameters (Eq. 7): adjacency Ag and feature Wg
        self.adjacency = nn.Parameter(torch.eye(num_nodes) + 0.01 * torch.randn(num_nodes, num_nodes))
        self.node_transform = nn.Linear(channels, channels)
        # G -> X : remap graph back to spatial domain (Eq. 8)
        self.from_graph = nn.Conv2d(num_nodes, channels, kernel_size=1)

    def forward(self, x):  # x: (B, C, H, W)
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
