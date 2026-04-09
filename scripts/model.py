"""Multi-architecture DG-GNN for respiratory boundary detection.

Supports: gat, gat_v2, gcn, sage, gin, pna, transformer, mlp_only, edge_conv
CRITICAL: add_self_loops=False on all attention-based convs (gfx1010 workaround).
"""

from __future__ import annotations

import inspect
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    BatchNorm,
    EdgeConv,
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    SAGEConv,
    TransformerConv,
    global_mean_pool,
)

try:
    from torch_geometric.nn import PNAConv

    _HAS_PNA = True
except ImportError:
    _HAS_PNA = False

SUPPORTED_ARCHS = frozenset(
    {"gat", "gat_v2", "gcn", "sage", "gin", "pna", "transformer", "mlp_only", "edge_conv"}
)

# Architectures whose conv layers accept edge_dim natively
_NATIVE_EDGE_ARCHS = frozenset({"gat", "gat_v2", "transformer", "pna"})

# Multi-head attention archs (output = hidden * heads when concat=True)
_MULTIHEAD_ARCHS = frozenset({"gat", "gat_v2", "transformer"})


class DGGNN(nn.Module):
    """Configurable Graph Neural Network on DegreeGraph structural graph.

    Args:
        config: dict with keys:
            arch: str — one of gat/gat_v2/gcn/sage/gin/pna/transformer/mlp_only/edge_conv
            in_dim: int — number of node features (default 6)
            edge_dim: int — number of edge features, 0 for none (default 2)
            hidden_dim: int — hidden layer size (default 64)
            n_heads: int — attention heads for GAT/GATv2/Transformer (default 8)
            n_layers: int — number of conv layers (default 3)
            dropout: float — dropout rate (default 0.1)
            residual: bool — skip connections (default False)
            batch_norm: bool — batch normalization (default False)
            concat_heads: bool — concat vs mean for multi-head (default True)
            aggr: str — aggregation for SAGE/GIN/EdgeConv (default "mean")
            boundary_head_layers: int — MLP depth for boundary head (default 1)
            use_rate_head: bool — include rate regression (default True)
            use_type_head: bool — include breath type classification (default False)
            deg: torch.Tensor — degree histogram for PNA (required if arch="pna")
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self._parse_config(config)
        self._resolve_pna_fallback(config)
        self._build_edge_projection()
        self._build_conv_layers()
        self._build_output_heads()

    # ------------------------------------------------------------------ #
    # Config                                                               #
    # ------------------------------------------------------------------ #

    def _parse_config(self, config: dict) -> None:
        self.arch: str = config.get("arch", "gat")
        if self.arch not in SUPPORTED_ARCHS:
            raise ValueError(
                f"Unknown arch '{self.arch}'. Supported: {sorted(SUPPORTED_ARCHS)}"
            )
        self.in_dim: int = config.get("in_dim", 6)
        self.edge_dim: int = config.get("edge_dim", 2)
        self.hidden_dim: int = config.get("hidden_dim", 64)
        self.n_heads: int = config.get("n_heads", 8)
        self.n_layers: int = max(config.get("n_layers", 3), 1)
        self.dropout: float = config.get("dropout", 0.1)
        self.use_residual: bool = config.get("residual", False)
        self.use_batch_norm: bool = config.get("batch_norm", False)
        self.concat_heads: bool = config.get("concat_heads", True)
        self.aggr: str = config.get("aggr", "mean")
        self.boundary_head_layers: int = max(config.get("boundary_head_layers", 1), 1)
        self.use_rate_head: bool = config.get("use_rate_head", True)
        self.use_type_head: bool = config.get("use_type_head", False)
        self._raw_config: dict = config

    def _resolve_pna_fallback(self, config: dict) -> None:
        """Fall back to SAGEConv if PNA prerequisites are missing."""
        if self.arch != "pna":
            return
        if not _HAS_PNA:
            warnings.warn("PNAConv unavailable; falling back to SAGEConv.", stacklevel=3)
            self.arch = "sage"
            return
        if "deg" not in config:
            warnings.warn(
                "PNA requires 'deg' in config; falling back to SAGEConv.", stacklevel=3
            )
            self.arch = "sage"

    # ------------------------------------------------------------------ #
    # Edge-feature projection (archs without native edge_dim)              #
    # ------------------------------------------------------------------ #

    def _build_edge_projection(self) -> None:
        self.edge_proj: nn.Linear | None = None
        self._effective_in_dim: int = self.in_dim

        needs_proj = (
            self.edge_dim > 0
            and self.arch not in _NATIVE_EDGE_ARCHS
            and self.arch != "mlp_only"
        )
        if needs_proj:
            self.edge_proj = nn.Linear(self.edge_dim, self.edge_dim)
            self._effective_in_dim = self.in_dim + self.edge_dim

    def _project_edge_to_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor:
        """Scatter-mean projected edge features to both endpoints, concat with *x*."""
        if self.edge_proj is None:
            return x
        if edge_attr is None:
            edge_attr = x.new_zeros(edge_index.size(1), self.edge_dim)

        e = self.edge_proj(edge_attr)  # [E, proj_dim]
        src, dst = edge_index
        n, d = x.size(0), e.size(1)

        agg = x.new_zeros(n, d)
        count = x.new_zeros(n, 1)
        ones = count.new_ones(src.size(0), 1)

        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(e), e)
        agg.scatter_add_(0, src.unsqueeze(1).expand_as(e), e)
        count.scatter_add_(0, dst.unsqueeze(1), ones)
        count.scatter_add_(0, src.unsqueeze(1), ones)

        agg = agg / count.clamp(min=1)
        return torch.cat([x, agg], dim=-1)

    # ------------------------------------------------------------------ #
    # Conv-layer construction                                              #
    # ------------------------------------------------------------------ #

    def _intermediate_dim(self) -> int:
        """Output dim of non-last conv layers (multi-head concat expands dim)."""
        if self.arch in _MULTIHEAD_ARCHS and self.concat_heads:
            return self.hidden_dim * self.n_heads
        return self.hidden_dim

    def _layer_out_dim(self, is_last: bool) -> int:
        return self.hidden_dim if is_last else self._intermediate_dim()

    def _edge_dim_or_none(self) -> int | None:
        return self.edge_dim if self.edge_dim > 0 else None

    def _make_attention_conv(
        self, cls: type, in_ch: int, out_ch: int, is_last: bool
    ) -> nn.Module:
        """Build GAT / GATv2 / Transformer conv with add_self_loops=False."""
        heads = 1 if is_last else self.n_heads
        concat = False if is_last else self.concat_heads

        kw: dict = dict(
            in_channels=in_ch,
            out_channels=out_ch,
            heads=heads,
            concat=concat,
            edge_dim=self._edge_dim_or_none(),
        )

        # gfx1010 workaround: disable self-loops where the conv supports it
        sig = inspect.signature(cls.__init__)
        if "add_self_loops" in sig.parameters:
            kw["add_self_loops"] = False
        if "dropout" in sig.parameters:
            kw["dropout"] = self.dropout

        return cls(**kw)

    def _make_conv(self, in_ch: int, out_ch: int, is_last: bool) -> nn.Module:
        """Create one conv / linear layer for the configured architecture."""
        arch = self.arch

        if arch == "gat":
            return self._make_attention_conv(GATConv, in_ch, out_ch, is_last)

        if arch == "gat_v2":
            return self._make_attention_conv(GATv2Conv, in_ch, out_ch, is_last)

        if arch == "transformer":
            return self._make_attention_conv(TransformerConv, in_ch, out_ch, is_last)

        if arch == "pna":
            return PNAConv(
                in_ch,
                out_ch,
                aggregators=["mean", "min", "max", "std"],
                scalers=["identity", "amplification", "attenuation"],
                deg=self._raw_config["deg"],
                edge_dim=self._edge_dim_or_none(),
            )

        if arch == "gcn":
            return GCNConv(in_ch, out_ch)

        if arch == "sage":
            return SAGEConv(in_ch, out_ch, aggr=self.aggr)

        if arch == "gin":
            return GINConv(
                nn.Sequential(nn.Linear(in_ch, out_ch), nn.ELU(), nn.Linear(out_ch, out_ch))
            )

        if arch == "edge_conv":
            return EdgeConv(
                nn.Sequential(
                    nn.Linear(2 * in_ch, out_ch), nn.ELU(), nn.Linear(out_ch, out_ch)
                ),
                aggr=self.aggr,
            )

        raise ValueError(f"No conv builder for arch '{arch}'")  # pragma: no cover

    def _build_conv_layers(self) -> None:
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if self.arch == "mlp_only":
            dims = [self.in_dim] + [self.hidden_dim] * self.n_layers
            for i in range(self.n_layers):
                self.convs.append(nn.Linear(dims[i], dims[i + 1]))
                self.norms.append(
                    nn.BatchNorm1d(dims[i + 1]) if self.use_batch_norm else nn.Identity()
                )
            return

        for i in range(self.n_layers):
            is_last = i == self.n_layers - 1
            in_ch = self._effective_in_dim if i == 0 else self._intermediate_dim()
            self.convs.append(self._make_conv(in_ch, self.hidden_dim, is_last))
            self.norms.append(
                BatchNorm(self._layer_out_dim(is_last))
                if self.use_batch_norm
                else nn.Identity()
            )

    # ------------------------------------------------------------------ #
    # Output heads                                                         #
    # ------------------------------------------------------------------ #

    def _build_output_heads(self) -> None:
        h = self.hidden_dim

        if self.boundary_head_layers <= 1:
            self.boundary_head: nn.Module = nn.Linear(h, 1)
        else:
            mid = max(h // 2, 1)
            self.boundary_head = nn.Sequential(
                nn.Linear(h, mid),
                nn.ELU(),
                nn.Dropout(self.dropout),
                nn.Linear(mid, 1),
            )

        self.rate_head: nn.Module | None = nn.Linear(h, 1) if self.use_rate_head else None
        self.type_head: nn.Module | None = nn.Linear(h, 3) if self.use_type_head else None

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run forward pass through conv layers and output heads.

        Args:
            x: [N, in_dim] node features.
            edge_index: [2, E] COO edge indices.
            edge_attr: [E, edge_dim] edge features (optional).
            batch: [N] graph membership for batched graphs (optional).

        Returns:
            dict with keys:
                boundary_logits: [N] per-node boundary scores.
                rate_pred: [B] per-graph rate (present when use_rate_head=True).
                type_logits: [N, 3] per-node type (present when use_type_head=True).
        """
        # Fold edge features into node features for non-native archs
        if self.arch != "mlp_only":
            x = self._project_edge_to_nodes(x, edge_index, edge_attr)

        # Prepare edge_attr for native-edge convs (zero-fill if missing)
        ea: torch.Tensor | None = None
        if self.arch in _NATIVE_EDGE_ARCHS and self.edge_dim > 0:
            ea = edge_attr if edge_attr is not None else x.new_zeros(edge_index.size(1), self.edge_dim)

        # Conv / MLP blocks
        for conv, norm in zip(self.convs, self.norms):
            x_in = x
            if self.arch == "mlp_only":
                x = conv(x)
            elif self.arch in _NATIVE_EDGE_ARCHS:
                x = conv(x, edge_index, ea)
            else:
                x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual and x_in.shape == x.shape:
                x = x + x_in

        # --- Output heads ---
        out: dict[str, torch.Tensor] = {}
        out["boundary_logits"] = self.boundary_head(x).squeeze(-1)

        if self.rate_head is not None:
            pooled = global_mean_pool(x, batch) if batch is not None else x.mean(dim=0, keepdim=True)
            out["rate_pred"] = self.rate_head(pooled).squeeze(-1)

        if self.type_head is not None:
            out["type_logits"] = self.type_head(x)

        return out


# ---------------------------------------------------------------------- #
# Factory and utilities                                                    #
# ---------------------------------------------------------------------- #


def build_model(config: dict) -> DGGNN:
    """Factory function to build a model from config dict."""
    return DGGNN(config)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
