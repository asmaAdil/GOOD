from typing import Any, Dict, List, Optional, Tuple, Type, Union

import dgl
import numpy as np
import torch.optim
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import (
    EPOCH_OUTPUT,
    EVAL_DATALOADERS,
    STEP_OUTPUT,
    TRAIN_DATALOADERS,
)
from torchmetrics import AUROC, F1, Accuracy, AveragePrecision, MeanSquaredLogError

from conf_manager import ConfManager
from contextual_modules import (
    MLP,
    Aggregator,
    ContextualBlock,
)
from metrics import HitsRateMetric


class ContextualModel(LightningModule):  # pylint: disable=too-many-ancestors # LightningModule itself has 10 ancestor
    def __init__(
        self,
        conf: Type[ConfManager],
        train_coefficients: Optional[Dict[str, float]] = None,
        eval_coefficients: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.conf = conf

        # Total number of graph-layers and blocks
        self.total_number_of_layers = sum(self.conf.get("number_of_layers"))
        self.number_of_block_layers = len(self.conf.get("number_of_layers"))

        # Variables used for Dynamic Programming
        self.block_layers_to_range: Dict[int, Tuple[int, int]] = {}
        self.dataloader_idx_to_block_and_index_layer: Dict[int, Tuple[int, int]] = {}

        # Model
        self.block_model = torch.nn.ModuleList()
        for block_layer, layers_per_block in enumerate(self.conf.get("number_of_layers")):
            if block_layer == 0:
                in_feats = {
                    etype: dims["in_feats"]
                    for etype, dims in self.conf.get("parameters_per_block_layer")["block_1_layer_1"].items()
                }
            else:
                in_feats = {
                    etype: dims["out_feats"]
                    for etype, dims in self.conf.get("parameters_per_block_layer")[
                        f"block_{block_layer}_layer_{self.conf.get('number_of_layers')[block_layer - 1]}"
                    ].items()
                }
                if (
                    "stack" in self.conf.get("aggregation_type")
                    and self.conf.get("aggregate_contexts")[f"block_{block_layer}"]
                ):
                    assert len(set(list(in_feats.values()))) == 1, "Number of out-feats per context are not equal."
                    in_feats = {etype: dims * len(in_feats) for etype, dims in in_feats.items()}
            self.block_model.append(
                ContextualBlock(
                    in_feats=in_feats,
                    parameters_per_layer=[
                        self.conf.get("parameters_per_block_layer")[f"block_{block_layer + 1}_layer_{layer + 1}"]
                        for layer in range(layers_per_block)
                    ],
                    edge_weight=self.conf.get("edge_weight"),
                    use_residuals=self.conf.get("use_residuals")[f"block_{block_layer + 1}"],
                    stack_aggregation="stack" in self.conf.get("aggregation_type")
                    and self.conf.get("aggregate_contexts")[f"block_{block_layer + 1}"]
                    and block_layer + 1 <= self.number_of_block_layers,
                )
            )

        self.aggregator = Aggregator(aggregation_type=self.conf.get("aggregation_type"))

        in_channels_contexts = {
            etype: dims["out_feats"]
            for etype, dims in self.conf.get("parameters_per_block_layer")[
                f"block_{len(self.conf.get('number_of_layers'))}_layer_{self.conf.get('number_of_layers')[-1]}"
            ].items()
        }
        if "stack" in self.conf.get("aggregation_type"):
            in_channels = sum(in_channels_contexts.values())
        else:
            assert len(set(list(in_channels_contexts.values()))) == 1, "Number of out-feats per context are not equal."
            in_channels = list(in_channels_contexts.values())[0]

        self.link_predictor = torch.nn.ModuleDict()
        for etype, params in self.conf.get("parameters_link_predictor").items():
            self.link_predictor[etype] = MLP(
                in_channels=in_channels,
                hidden_channels=params["hidden_channels"],
                out_channels=params["out_channels"],
                internal_activation_function=params["internal_activation_function"],
                final_activation_function=params["final_activation_function"],
                dropout=params["dropout"],
                batch_norm=params["batch_norm"],
            )

        self.coefficient_reconstruction: Optional[torch.nn.Module] = MLP(
            in_channels=in_channels,
            hidden_channels=self.conf.get("parameters_coefficient_reconstruction")["hidden_channels"],
            out_channels=len(in_channels_contexts),
            internal_activation_function=self.conf.get("parameters_coefficient_reconstruction")[
                "internal_activation_function"
            ],
            final_activation_function=self.conf.get("parameters_coefficient_reconstruction")[
                "final_activation_function"
            ],
            dropout=self.conf.get("parameters_coefficient_reconstruction")["dropout"],
            batch_norm=self.conf.get("parameters_coefficient_reconstruction")["batch_norm"],
        )

        self.sigmoid = torch.nn.Sigmoid()

        # Link Criterion & Coefficient Disentanglement Criterion (Objectives)
        self.link_criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss()
        self.coefficient_reconstruction_criterion: Optional[torch.nn.Module] = MeanSquaredLogError()

        # Metrics
        self.accuracy_metric = Accuracy()
        self.auc_metric = AUROC()
        self.hits_rate_metric = torch.nn.ModuleDict({str(k): HitsRateMetric(k=k) for k in self.conf.get("hits_k")})
        self.f1_metric = F1()
        self.ap_metric = AveragePrecision()

        # Initialize coefficients
        if isinstance(train_coefficients, dict) and isinstance(eval_coefficients, dict):
            assert sorted(train_coefficients) == sorted(eval_coefficients), "Contexts don't have the same order."
        if isinstance(self.conf, dict):
            self.conf.update(
                {
                    "train_coefficients": train_coefficients,
                    "eval_coefficients": eval_coefficients,
                }
            )
        else:
            self.conf.update_by_dict(
                {
                    "train_coefficients": train_coefficients,
                    "eval_coefficients": eval_coefficients,
                }
            )
        self.coefficients: Dict[str, torch.Tensor] = {}

        # Saving hyper-parameters (conf)
        if isinstance(self.conf, dict):
            self.save_hyperparameters()
        else:
            self.save_hyperparameters(self.conf.get_all_params(), ignore=["conf"])
        self.node_feat = torch.nn.Embedding(self.conf.get("num_nodes"), self.conf.get("num_feat_dim"))

    def _get_start_end_block_layers(self, layer: int) -> Tuple[int, int]:
        if layer in self.block_layers_to_range:
            return self.block_layers_to_range[layer]
        start = 0 if layer == 0 else sum(self.conf.get("number_of_layers")[:layer])
        end = start + self.conf.get("number_of_layers")[layer]
        self.block_layers_to_range[layer] = (start, end)
        return self.block_layers_to_range[layer]

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.conf.get("learning_rate"),
            weight_decay=self.conf.get("weight_decay"),
        )

    def _get_degree_etypes(self, phase: str, offset: int = 0) -> List[str]:
        degree_etypes: List[str] = []
        for etype in self.conf.get(f"{phase}_etypes_degrees"):
            if etype != "style":
                split_etype = etype.split("_")
                degree_etypes.append("_".join(split_etype[:-1] + [str(int(split_etype[-1]) - offset)]))
            else:
                degree_etypes.append(etype)
        return degree_etypes

    @staticmethod
    def _add_residuals(embeds: Dict[str, torch.Tensor], residuals: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for context, feats in residuals.items():
            embeds[context] += feats
        return embeds

    def forward(  # type: ignore # pylint: disable=arguments-differ # LightningModule expects *args, **kwargs
        self,
        input_nodes: torch.Tensor,
        blocks: List[dgl.DGLHeteroGraph],
        coefficients: Union[Dict[str, torch.Tensor], torch.nn.ModuleDict],
        edges: Dict[str, torch.Tensor],
        phase: str,
        layer_offset: int = 0,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        aggregated_embedding: Optional[torch.Tensor] = None

        # Graph Encoder
        embeds = {
            etype[1].split(sep="_", maxsplit=1)[0]: self.node_feat(input_nodes) for etype in blocks[0].canonical_etypes
        }
        residuals: Dict[str, torch.Tensor] = {}
        for layer, block_layers in enumerate(self.block_model):
            embeds = self._add_residuals(
                embeds=embeds,
                residuals=residuals,
            )

            start, end = self._get_start_end_block_layers(layer=layer)
            embeds, residuals = block_layers(
                feats=embeds,
                blocks=blocks[start:end],
                block_layer=f"season_{layer + 1 + layer_offset}",
            )

            # Aggregation
            if layer + 1 == self.number_of_block_layers or self.conf.get("aggregate_contexts")[f"block_{layer + 1}"]:
                degrees: Optional[Dict[str, torch.Tensor]] = None
                if "degree" in self.conf.get("aggregation_type"):
                    degrees = self.aggregator.compute_degrees(
                        block=blocks[end - 1],
                        etypes=self._get_degree_etypes(
                            phase=phase,
                            offset=self.number_of_block_layers - layer - 1,
                        ),
                        weighted="weighted" in self.conf.get("aggregation_type"),
                    )
                aggregated_embedding = self.aggregator(
                    embeds=embeds,
                    coefficients=coefficients,
                    degrees=degrees,
                )
                embeds = {etype: aggregated_embedding for etype in self.conf.get("etypes_input")}

        # Link-Predictions
        assert isinstance(aggregated_embedding, torch.Tensor)
        links = {
            etype: link_predictor(x=aggregated_embedding[edges[etype][0]] * aggregated_embedding[edges[etype][1]])
            for etype, link_predictor in self.link_predictor.items()
        }

        # Coefficient Disentanglement
        assert isinstance(self.coefficient_reconstruction, torch.nn.Module)
        reconstructed_coefficients = self.coefficient_reconstruction(x=aggregated_embedding)

        return links, reconstructed_coefficients

    def on_eval_shared_start(self, phase: str) -> None:
        if isinstance(self.conf.get("train_coefficients"), dict) and phase == "train":
            self.coefficients = {
                etype: torch.tensor(coef, requires_grad=False, device=self.device)
                for etype, coef in self.conf.get("train_coefficients").items()
            }
        elif isinstance(self.conf.get("eval_coefficients"), dict) and phase != "train":
            self.coefficients = {
                etype: torch.tensor(coef, requires_grad=False, device=self.device)
                for etype, coef in self.conf.get("eval_coefficients").items()
            }
        else:
            coef = 1.0 / len(self.conf.get("etypes_input"))
            self.coefficients = {
                etype: torch.tensor(coef, requires_grad=False, device=self.device)
                for etype in self.conf.get("etypes_input")
            }

        for etype in self.conf.get("etypes_input"):
            self.log(
                name=f"coefficients_{etype}",
                value={phase: self.coefficients[etype]},
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                logger=True,
                enable_graph=False,
                add_dataloader_idx=False,
            )

    def on_train_epoch_start(self) -> None:
        if self.conf.get("model_type") == "fixed":
            self.on_eval_shared_start(phase="train")
        elif self.conf.get("model_type") == "random":
            self.coefficients = {}
            for etype, coef in zip(
                self.conf.get("etypes_input"),
                np.random.dirichlet(np.random.random(len(self.conf.get("etypes_input")))),
            ):
                self.coefficients[etype] = torch.tensor(coef, requires_grad=False, device=self.device)
                self.log(
                    name=f"coefficients_{etype}",
                    value={"train": self.coefficients[etype]},
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    enable_graph=False,
                    add_dataloader_idx=False,
                )
        else:
            raise ValueError(f"Model-Type is not correct: {self.conf.get('model_type')}.")
        print(f"Training coefficients: {self.coefficients} at epoch: {self.current_epoch}")

    def on_validation_epoch_start(self) -> None:
        self.on_eval_shared_start(phase="valid")
        print(f"Validation coefficients: {self.coefficients} at epoch: {self.current_epoch}")

    def on_test_epoch_start(self) -> None:
        self.on_eval_shared_start(phase="test")
        print(f"Test coefficients: {self.coefficients} at epoch: {self.current_epoch}")

    def _get_edges(
        self, pos_graph: dgl.DGLHeteroGraph, neg_graph: dgl.DGLHeteroGraph, phase: str
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        target_edges: Dict[str, torch.Tensor] = {}
        edges: Dict[str, torch.Tensor] = {}
        for etype in self.conf.get(f"{phase}_etypes_target"):
            pos_src, pos_dst = pos_graph.edges(form="uv", etype=etype)
            neg_src, neg_dst = neg_graph.edges(form="uv", etype=etype)

            context = etype.split(sep="_", maxsplit=1)[0]
            edges[context] = torch.vstack(
                [
                    torch.hstack([pos_src, neg_src]),
                    torch.hstack([pos_dst, neg_dst]),
                ]
            )
            target_edges[context] = torch.hstack([torch.ones_like(pos_src), torch.zeros_like(neg_src)])
        return edges, target_edges

    @torch.no_grad()
    def _link_metric(
        self,
        links: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        phase: str,
        metric: torch.nn.Module,
        metric_name: str,
        device: Union[str, torch.device] = "cpu",
    ) -> Tuple[torch.Tensor, int]:
        total_score = torch.zeros(1, device=device)
        batch_size = 0
        for etype, preds in links.items():
            if preds.shape[0] == 0:
                continue
            score = metric(preds=preds.squeeze(), target=targets[etype])
            link_batch_size = preds.shape[0]
            batch_size += link_batch_size
            self.log(
                name=f"{metric_name}-{etype}",
                value={phase: score.cpu()},
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                logger=True,
                enable_graph=False,
                batch_size=link_batch_size,
                add_dataloader_idx=False,
            )
            total_score += score * link_batch_size
        total_score /= batch_size
        self.log(
            name=f"total_{metric_name}",
            value={phase: total_score.cpu()},
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            enable_graph=False,
            batch_size=batch_size,
            add_dataloader_idx=False,
        )
        if phase == "valid":
            self.log(
                name=f"total_{metric_name}_valid",
                value=total_score.cpu(),
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                logger=True,
                enable_graph=False,
                batch_size=batch_size,
                add_dataloader_idx=False,
            )
        return total_score.cpu(), batch_size

    @torch.no_grad()
    def _estimate_link_metrics(
        self,
        predict_links: Dict[str, torch.Tensor],
        target_links: Dict[str, torch.Tensor],
        phase: str,
    ) -> Dict[str, Dict[str, Union[int, torch.Tensor]]]:
        links = {etype: self.sigmoid(links).cpu() for etype, links in predict_links.items()}
        targets = {etype: target.cpu() for etype, target in target_links.items()}

        total_acc, batch_size_acc = self._link_metric(
            links=links,
            targets=targets,
            phase=phase,
            metric=self.accuracy_metric.cpu(),
            metric_name="accuracy",
            device=torch.device("cpu"),
        )
        total_auc, batch_size_auc = self._link_metric(
            links=links,
            targets=targets,
            phase=phase,
            metric=self.auc_metric.cpu(),
            metric_name="auc",
            device=torch.device("cpu"),
        )
        total_f1, batch_size_f1 = self._link_metric(
            links=links,
            targets=targets,
            phase=phase,
            metric=self.f1_metric.cpu(),
            metric_name="f1",
            device=torch.device("cpu"),
        )
        total_ap, batch_size_ap = self._link_metric(
            links=links,
            targets=targets,
            phase=phase,
            metric=self.ap_metric.cpu(),
            metric_name="ap",
            device=torch.device("cpu"),
        )

        hits_results: Dict[str, Dict[str, Union[int, torch.Tensor]]] = {}
        for k, hit_metric in self.hits_rate_metric.items():
            total_hits, batch_size_hits = self._link_metric(
                links=links,
                targets=targets,
                phase=phase,
                metric=hit_metric.cpu(),
                metric_name=f"hits_{k}",
                device=torch.device("cpu"),
            )
            hits_results[f"hits_{k}"] = {
                "score": total_hits,
                "batch_size": batch_size_hits,
            }

        return {
            "accuracy": {"score": total_acc, "batch_size": batch_size_acc},
            "auc": {"score": total_auc, "batch_size": batch_size_auc},
            "f1": {"score": total_f1, "batch_size": batch_size_f1},
            "ap": {"score": total_ap, "batch_size": batch_size_ap},
            **hits_results,
        }

    def shared_step(
        self,
        batch: Tuple[torch.Tensor, dgl.DGLHeteroGraph, dgl.DGLHeteroGraph, List[dgl.DGLHeteroGraph]],
        phase: str,
    ) -> STEP_OUTPUT:
        input_nodes, pos_graph, neg_graph, blocks = batch

        edges, target_links = self._get_edges(
            pos_graph=pos_graph,
            neg_graph=neg_graph,
            phase=phase,
        )

        predict_links, reconstructed_coefficients = self(
            input_nodes=input_nodes,
            blocks=blocks,
            coefficients=self.coefficients,
            edges=edges,
            phase=phase,
            layer_offset=self.conf.get(f"{phase}_block_layer_offset"),
        )
        self._estimate_link_metrics(
            predict_links=predict_links,
            target_links=target_links,
            phase=phase,
        )

        pos_mask = {etype: target_links[etype] > 0 for etype in sorted(predict_links)}
        pos_link_loss: torch.Tensor = self.link_criterion(
            input=torch.cat([predict_links[etype][pos_mask[etype]] for etype in sorted(predict_links)]),
            target=torch.cat([target_links[etype][pos_mask[etype]] for etype in sorted(predict_links)])
            .unsqueeze(1)
            .float(),
        )
        self.log(
            name="pos-link_loss",
            value={phase: pos_link_loss.cpu().item()},
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            enable_graph=False,
            batch_size=pos_graph.num_edges(),
        )
        neg_link_loss: torch.Tensor = self.link_criterion(
            input=torch.cat([predict_links[etype][~pos_mask[etype]] for etype in sorted(predict_links)]),
            target=torch.cat([target_links[etype][~pos_mask[etype]] for etype in sorted(predict_links)])
            .unsqueeze(1)
            .float(),
        )
        self.log(
            name="neg-link_loss",
            value={phase: neg_link_loss.cpu().item()},
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            enable_graph=False,
            batch_size=neg_graph.num_edges(),
        )
        link_batch_size = pos_graph.num_edges() + neg_graph.num_edges()
        link_loss = (pos_link_loss * pos_graph.num_edges() + neg_link_loss * neg_graph.num_edges()) / link_batch_size
        self.log(
            name="link_loss",
            value={phase: link_loss.cpu().item()},
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            enable_graph=False,
            batch_size=link_batch_size,
        )

        assert isinstance(self.coefficient_reconstruction_criterion, torch.nn.Module)
        coefficient_reconstruction_loss: torch.Tensor = self.coefficient_reconstruction_criterion(
            preds=reconstructed_coefficients,
            target=torch.cat(
                [
                    self.coefficients[etype].unsqueeze(0).repeat(1, reconstructed_coefficients.shape[0])
                    for etype in self.conf.get("etypes_input")
                ]
            ).T,
        )
        coefficient_reconstruction_loss = coefficient_reconstruction_loss * self.conf.get("coefficient")
        agg_embed_batch_size = np.product(reconstructed_coefficients.shape)
        self.log(
            name="coef_reconstruction_loss",
            value={phase: coefficient_reconstruction_loss.cpu().item()},
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            enable_graph=False,
            batch_size=agg_embed_batch_size,
        )

        total_loss = link_loss + coefficient_reconstruction_loss
        print(
            f"loss: {total_loss}, link-loss: {link_loss}, "
            f"coef-reconstruction-loss: {coefficient_reconstruction_loss}, "
            f"pos-link-loss: {pos_link_loss}, neg-link-loss: {neg_link_loss}"
        )
        return {
            "loss": total_loss,
            "link_loss": link_loss.item(),
            "coef_reconstruction_loss": coefficient_reconstruction_loss.item(),
            "link_batch_size": link_batch_size,
            "agg_embed_batch_size": agg_embed_batch_size,
        }

    def training_step(  # type: ignore # pylint: disable=arguments-differ
        # pytorch_lightning require a tensor but the dgl batch is a list type
        self,
        batch: Tuple[torch.Tensor, dgl.DGLHeteroGraph, dgl.DGLHeteroGraph, List[dgl.DGLHeteroGraph]],
        _: int,
    ) -> STEP_OUTPUT:
        return self.shared_step(
            batch=batch,
            phase="train",
        )

    def validation_step(  # type: ignore # pylint: disable=arguments-differ
        # pytorch_lightning require a tensor but the dgl batch is a list type
        self,
        batch: Tuple[torch.Tensor, dgl.DGLHeteroGraph, dgl.DGLHeteroGraph, List[dgl.DGLHeteroGraph]],
        _: int,
    ) -> Optional[STEP_OUTPUT]:
        return self.shared_step(
            batch=batch,
            phase="valid",
        )

    def test_step(  # type: ignore # pylint: disable=arguments-differ
        # pytorch_lightning require a tensor but the dgl batch is a list type
        self,
        batch: Tuple[torch.Tensor, dgl.DGLHeteroGraph, dgl.DGLHeteroGraph, List[dgl.DGLHeteroGraph]],
        _: int,
    ) -> Optional[STEP_OUTPUT]:
        return self.shared_step(
            batch=batch,
            phase="test",
        )

    @torch.no_grad()
    def shared_epoch_end(self, outputs: EPOCH_OUTPUT, phase: str) -> None:
        device = torch.device(device="cpu")
        link_batch_size, agg_embed_batch_size = torch.zeros(1, device=device), torch.zeros(1, device=device)
        total_link_loss, total_coef_reco_loss = torch.zeros(1, device=device), torch.zeros(1, device=device)
        for output in outputs:
            assert isinstance(output, dict)
            link_batch_size += output["link_batch_size"]
            total_link_loss += output["link_loss"] * output["link_batch_size"]
            agg_embed_batch_size += output["agg_embed_batch_size"]
            total_coef_reco_loss += output["coef_reconstruction_loss"] * output["agg_embed_batch_size"]
        total_link_loss /= link_batch_size
        total_coef_reco_loss /= agg_embed_batch_size
        total_loss = total_link_loss + total_coef_reco_loss
        self.log(
            name="total_loss",
            value={phase: total_loss.cpu()},
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            enable_graph=False,
            batch_size=1,
        )
        if phase == "valid":
            self.log(
                name="total_loss_valid",
                value=total_loss.cpu(),
                prog_bar=True,
                logger=True,
                on_step=False,
                on_epoch=True,
                enable_graph=False,
                batch_size=1,
            )

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.shared_epoch_end(
            outputs=outputs,
            phase="train",
        )

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.shared_epoch_end(
            outputs=outputs,
            phase="valid",
        )

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.shared_epoch_end(
            outputs=outputs,
            phase="test",
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        raise NotImplementedError(
            "`train_dataloader` must be implemented to be used with the Lightning Trainer "
            "or provide 'train_dataloader' from LightningDataModule"
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError(
            "`test_dataloader` must be implemented to be used with the Lightning Trainer "
            "or provide 'test_dataloader' from LightningDataModule"
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError(
            "`val_dataloader` must be implemented to be used with the Lightning Trainer "
            "or provide 'val_dataloader' from LightningDataModule"
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError(
            "`predict_dataloader` must be implemented to be used with the Lightning Trainer "
            "or provide 'predict_dataloader' from LightningDataModule"
        )
