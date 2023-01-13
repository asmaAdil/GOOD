from typing import Any, Dict, List, Optional, Tuple, Union

import dgl
import torch
from dgl.nn.pytorch.conv import EdgeWeightNorm, GraphConv


def _get_elu_family(activation: str) -> torch.nn.Module:
    if activation == "ELU":
        return torch.nn.ELU()
    if activation == "ReLU":
        return torch.nn.ReLU()
    if activation == "LeakyReLU":
        return torch.nn.LeakyReLU()
    raise NotImplementedError(f"Undefined {activation} activation.")


def get_activation(activation: str) -> torch.nn.Module:
    """
    Get activation functions based on a string description
    :param activation: str
    :return: torch.nn.Module
    """
    if "elu" in activation.lower():
        return _get_elu_family(activation=activation)
    if activation == "Sigmoid":
        return torch.nn.Sigmoid()
    if activation.startswith("Softmax"):
        return torch.nn.Softmax(dim=int(activation.split("_")[-1]))
    if activation.startswith("LogSoftmax"):
        return torch.nn.LogSoftmax(dim=int(activation.split("_")[-1]))
    if activation == "Tanh":
        return torch.nn.Tanh()
    raise NotImplementedError(f"Undefined {activation} activation.")


class ContextualBlock(torch.nn.Module):
    """
    Graph Neural Network Block per Context
    The given Block consists of a series of:
            GraphEmbeds + ResidualEmbeds [Optional] ->
            GCN ->
            Activation [Optional] ->
            Dropout [Optional] ->
            BatchNorm1D [Optional] ->
            Embedding Vectors
    """

    def __init__(
        self,
        in_feats: Dict[str, int],
        parameters_per_layer: List[Dict[str, Dict[str, Any]]],
        edge_weight: bool = False,
        use_residuals: bool = False,
        stack_aggregation: bool = False,
    ) -> None:
        super().__init__()

        self.edge_weight: bool = edge_weight
        self.use_residuals: bool = use_residuals

        self.edge_norm = EdgeWeightNorm(norm="both")

        residual_in_feats: Dict[str, int] = {}
        residual_out_feats: Dict[str, int] = {}
        number_of_layers = len(parameters_per_layer)

        contexts = set()
        self.block_layers = torch.nn.ModuleDict()
        for layer, parameters in enumerate(parameters_per_layer):
            for context, params in parameters.items():
                if layer > 0:
                    in_feats = {context: parameters_per_layer[layer - 1][context]["out_feats"]}

                # Residual
                if self.use_residuals:
                    if layer == 0:
                        residual_in_feats[context] = in_feats[context]
                    if layer == number_of_layers - 1:
                        residual_out_feats[context] = params["out_feats"]
                        if stack_aggregation:
                            residual_out_feats[context] *= len(parameters)

                # GNN
                contexts.add(context.split(sep="_", maxsplit=1)[0])
                self._update_gnn_layers(
                    in_feats=in_feats[context],
                    params=params,
                    context=context,
                    layer=layer,
                )

                # Sequential
                self._update_sequential_layers(
                    params=params,
                    context=context,
                    layer=layer,
                )

        self.residuals = torch.nn.ModuleDict()
        if self.use_residuals:
            assert sorted(residual_in_feats) == sorted(residual_out_feats)
            self._update_residual_layers(
                residual_in_feats=residual_in_feats,
                residual_out_feats=residual_out_feats,
            )

        self.contexts = list(contexts)

    def _update_sequential_layers(self, params: Dict[str, Any], context: str, layer: int) -> None:
        sequential_layers: List[torch.nn.Module] = []
        if isinstance(params["dropout"], float):
            sequential_layers.append(torch.nn.Dropout(p=params["dropout"]))
        if params["batch_norm"]:
            sequential_layers.append(torch.nn.BatchNorm1d(num_features=params["out_feats"]))
        self.block_layers[f"{context}_{layer}_sequential"] = torch.nn.Sequential(*sequential_layers)

    def _update_gnn_layers(self, in_feats: int, params: Dict[str, Any], context: str, layer: int) -> None:
        if params["graph_type"] == "GraphConv":
            self.block_layers[f"{context}_{layer}_gnn"] = GraphConv(
                in_feats=in_feats,
                out_feats=params["out_feats"],
                activation=get_activation(params["activation"]) if params["activation"] is not None else None,
                norm="none" if self.edge_weight else "both",
                weight=True,
                bias=True,
                allow_zero_in_degree=False,
            )
        else:
            raise NotImplementedError(f"{params['graph_type']} has not been implemented.")

    def _update_residual_layers(self, residual_in_feats: Dict[str, int], residual_out_feats: Dict[str, int]) -> None:
        for res_context, in_feats in residual_in_feats.items():
            if in_feats != residual_out_feats[res_context]:
                self.residuals[res_context] = torch.nn.Linear(
                    in_features=in_feats,
                    out_features=residual_out_feats[res_context],
                    bias=False,
                )

    def sub_forward(
        self,
        embeds: Dict[str, torch.Tensor],
        block: dgl.DGLHeteroGraph,
        context: str,
        layer: int,
        block_layer: str,
    ) -> Dict[str, torch.Tensor]:
        block_context = f"{context}_{block_layer}" if context != "style" else context
        edge_weight: Optional[torch.Tensor] = None
        if self.edge_weight:
            edge_weight = self.edge_norm(
                graph=block[block_context],
                edge_weight=block[block_context].edata["feat"].type_as(embeds[context]),
            )

        embeds[context] = self.block_layers[f"{context}_{layer}_gnn"](
            feat=embeds[context],
            graph=block[block_context],
            edge_weight=edge_weight,
        )

        embeds[context] = self.block_layers[f"{context}_{layer}_sequential"](embeds[context])
        return embeds

    def residual_forward(
        self,
        embeds: torch.Tensor,
        context: str,
    ) -> torch.Tensor:
        if context in self.residuals:
            residual: torch.Tensor = self.residuals[context](embeds)
            return residual
        return embeds.clone()

    def forward(
        self,
        feats: Dict[str, torch.Tensor],
        blocks: List[dgl.DGLHeteroGraph],
        block_layer: str,
        sub_layer: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        residuals: Dict[str, torch.Tensor] = {}
        output_nodes = blocks[-1].dstnodes()

        if sub_layer is not None:
            # Eval Mode
            for context in self.contexts:
                if sub_layer == 0 and self.use_residuals:
                    residuals[context] = self.residual_forward(
                        embeds=feats[context][output_nodes],
                        context=context,
                    )
                feats = self.sub_forward(
                    embeds=feats,
                    block=blocks[0],
                    context=context,
                    layer=sub_layer,
                    block_layer=block_layer,
                )
            return feats, residuals

        # Train Mode
        for layer, block in enumerate(blocks):
            for context in self.contexts:
                if layer == 0 and self.use_residuals:
                    residuals[context] = self.residual_forward(
                        embeds=feats[context][output_nodes],
                        context=context,
                    )
                feats = self.sub_forward(
                    embeds=feats,
                    block=block,
                    context=context,
                    layer=layer,
                    block_layer=block_layer,
                )
        return feats, residuals


class Aggregator(torch.nn.Module):
    def __init__(self, aggregation_type: str) -> None:
        super().__init__()
        self.aggregation_type = aggregation_type

    @staticmethod
    def _sum(embeds: Dict[str, torch.Tensor]) -> torch.Tensor:
        aggregated_embed = torch.empty(0)
        for enum, embed in enumerate(embeds.values()):
            if enum == 0:
                aggregated_embed = torch.zeros_like(embed)
            aggregated_embed += embed
        return aggregated_embed

    @staticmethod
    def _stack(embeds: Dict[str, torch.Tensor]) -> torch.Tensor:
        aggregated_embed = torch.hstack(list(embeds.values()))
        return aggregated_embed

    @staticmethod
    def compute_degrees(
        block: dgl.DGLHeteroGraph,
        etypes: List[str],
        weighted: bool = False,
    ) -> Dict[str, torch.Tensor]:
        return {
            etype: dgl.ops.copy_e_sum(
                g=block[etype],
                x=block[etype].edata["feat"].float()
                if weighted
                else torch.ones_like(block[etype].edata["feat"].float()),
            )
            for etype in etypes
        }

    @staticmethod
    def _degree_scaled(embeds: Dict[str, torch.Tensor], degrees: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            etype: degree.unsqueeze(1) * embeds[etype.split(sep="_", maxsplit=1)[0]]
            for etype, degree in degrees.items()
        }

    def _degree_sum(self, embeds: Dict[str, torch.Tensor], degrees: Dict[str, torch.Tensor]) -> torch.Tensor:
        scaled_embeds = self._degree_scaled(embeds=embeds, degrees=degrees)
        return self._sum(scaled_embeds)

    def _degree_stack(self, embeds: Dict[str, torch.Tensor], degrees: Dict[str, torch.Tensor]) -> torch.Tensor:
        scaled_embeds = self._degree_scaled(embeds=embeds, degrees=degrees)
        return self._stack(scaled_embeds)

    def forward(
        self,
        embeds: Dict[str, torch.Tensor],
        coefficients: Union[Dict[str, torch.Tensor], torch.nn.ModuleDict],
        degrees: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if isinstance(coefficients, torch.nn.ModuleDict):
            scaled_embeds = {etype: coef(embeds[etype]) for etype, coef in coefficients.items()}
        else:
            scaled_embeds = {etype: embeds[etype] * coef for etype, coef in coefficients.items()}
        if self.aggregation_type == "sum":
            return self._sum(embeds=scaled_embeds)
        if self.aggregation_type == "mean":
            return self._sum(embeds=scaled_embeds) / len(coefficients)
        if self.aggregation_type == "stack":
            return self._stack(embeds=scaled_embeds)
        assert degrees is not None
        if self.aggregation_type.endswith("degree-sum"):
            return self._degree_sum(embeds=scaled_embeds, degrees=degrees)
        if self.aggregation_type.endswith("degree-stack"):
            return self._degree_stack(embeds=scaled_embeds, degrees=degrees)
        raise NotImplementedError(f"There is no implementation for Aggregation Type: {self.aggregation_type}")


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        out_channels: int,
        internal_activation_function: Optional[str] = None,
        final_activation_function: Optional[str] = None,
        batch_norm: bool = False,
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()

        num_hidden_layers = len(hidden_channels)

        if num_hidden_layers == 0:
            layers: List[torch.nn.Module] = [
                torch.nn.Linear(
                    in_features=in_channels,
                    out_features=out_channels,
                    bias=True,
                ),
            ]
        else:
            layers = [
                torch.nn.Linear(
                    in_features=in_channels,
                    out_features=hidden_channels[0],
                    bias=True,
                )
            ]
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(num_features=hidden_channels[0]))
            if internal_activation_function is not None:
                layers.append(get_activation(internal_activation_function))
            if dropout is not None:
                layers.append(torch.nn.Dropout(p=dropout))
            for enum, hidden_features in enumerate(hidden_channels[1:]):
                layers.append(
                    torch.nn.Linear(
                        in_features=hidden_channels[enum],
                        out_features=hidden_features,
                        bias=True,
                    )
                )
                if batch_norm:
                    layers.append(torch.nn.BatchNorm1d(num_features=hidden_features))
                if internal_activation_function is not None:
                    layers.append(get_activation(internal_activation_function))
                if dropout is not None:
                    layers.append(torch.nn.Dropout(p=dropout))
            layers.append(
                torch.nn.Linear(
                    in_features=hidden_channels[-1],
                    out_features=out_channels,
                    bias=True,
                )
            )
        if final_activation_function is not None:
            layers.append(get_activation(final_activation_function))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.layers(x)
        return output


class LearnableCoefficients(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.coefficient = torch.nn.Parameter(torch.ones(1, requires_grad=True, dtype=torch.float))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return feats * self.coefficient
