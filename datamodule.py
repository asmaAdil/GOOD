from typing import Dict, List, Optional, Tuple, Type, Union

import dgl
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from tqdm import tqdm

from conf_manager import ConfManager


class ContextualNeighborSampler(dgl.dataloading.BlockSampler):
    def __init__(
        self,
        etypes_per_layer: List[List[Tuple[str, str, str]]],
        number_of_layers: List[int],
    ) -> None:
        assert len(etypes_per_layer) == len(number_of_layers)

        self.etypes_per_layer = []
        for etypes, layer in zip(etypes_per_layer, number_of_layers):
            for _ in range(layer):
                self.etypes_per_layer.append(etypes)

        self.total_num_layers = sum(number_of_layers)
        super().__init__(self.total_num_layers)

    def sample_frontier(self, block_id: int, g: dgl.DGLHeteroGraph, seed_nodes: torch.Tensor) -> dgl.DGLHeteroGraph:
        # Get all inbound edges to `seed_nodes`
        sub_graph = dgl.in_subgraph(g, seed_nodes)

        # Return a new graph with the same nodes as the original graph as a frontier
        frontier = dgl.edge_type_subgraph(graph=sub_graph, etypes=self.etypes_per_layer[block_id])
        return frontier

    def __len__(self) -> int:
        return self.total_num_layers


class NegativeSampler:
    def __init__(self, edges: Dict) -> None:
        self.edges = edges

    def __call__(
        self, graph: dgl.DGLHeteroGraph, eids_dict: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]]:
        result_dict = {}
        for etype, eids in eids_dict.items():
            src, _ = graph.find_edges(eid=eids, etype=etype)  # find all dst of src
            mask = torch.tensor(np.isin(self.edges[etype]["src"].numpy(), src.numpy()))
            ####
            src = self.edges[etype]["src"][mask]
            dst = self.edges[etype]["dst"][mask]
            ####
            result_dict[etype] = (src, dst)
        return result_dict


class ContextualDataModule(LightningDataModule):
    def __init__(
        self,
        graph: dgl.DGLHeteroGraph,
        # split_edges: Dict[str, Dict[str, Dict[Tuple[str, str, str], np.ndarray]]],
        conf: Type[ConfManager],
    ) -> None:
        super().__init__()

        self.graph = graph
        self.conf = conf

        self.neg_edges: Dict[str, Dict[Tuple[str, str, str], Dict[str, torch.Tensor]]] = {
            "train": {},
            "valid": {},
            "test": {},
        }

        self.train_graph: Optional[dgl.DGLHeteroGraph] = None
        self.valid_graph: Optional[dgl.DGLHeteroGraph] = None
        self.test_graph: Optional[dgl.DGLHeteroGraph] = None
        self.predict_graph: Optional[dgl.DGLHeteroGraph] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", "validate", None):
            self.train_graph = dgl.edge_type_subgraph(
                graph=self.graph,
                etypes=sorted(self.conf.get("train_etypes")),
            )
            self.valid_graph = dgl.edge_type_subgraph(
                graph=self.graph,
                etypes=sorted(self.conf.get("valid_etypes")),
            )
            self.neg_edges["valid"] = self.get_eval_neg_edges(self.valid_graph)

        if stage in ("test", None):
            self.test_graph = dgl.edge_type_subgraph(
                graph=self.graph,
                etypes=sorted(self.conf.get("test_etypes")),
            )
            self.neg_edges["test"] = self.get_eval_neg_edges(self.test_graph)

    @staticmethod
    def add_self_loop(graph: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        for etype in graph.canonical_etypes:
            graph = graph.add_self_loop(etype=etype)
            src, dst = graph.edges(form="uv", etype=etype)
            graph[etype].edata["feat"][src == dst] = 1
        return graph

    def get_neg_sampler(self, phase: str) -> Union[dgl.dataloading.negative_sampler.Uniform, NegativeSampler]:
        if phase == "train":
            negative_sampler = dgl.dataloading.negative_sampler.Uniform(self.conf.get("train_negative_sampler"))
            return negative_sampler
        neg_sampler = NegativeSampler(edges=self.neg_edges[phase])
        return neg_sampler

    def shared_dataloader(self, graph: dgl.DGLHeteroGraph, phase: str) -> DataLoader:
        etypes_per_layer: List[List[Tuple[str, str, str]]] = []
        for layer in self.conf.get(f"{phase}_etypes_per_layer"):
            etypes_per_layer.append([("product", etype, "product") for etype in layer])
        block_sampler = ContextualNeighborSampler(
            etypes_per_layer=etypes_per_layer,
            number_of_layers=self.conf.get("number_of_layers"),
        )

        eids: Dict[Tuple[str, str, str], torch.Tensor] = {
            ("product", etype, "product"): graph.edges(form="eid", etype=etype)
            for etype in self.conf.get(f"{phase}_etypes_target")
        }

        negative_sampler = self.get_neg_sampler(phase=phase)

        reversed_graph = dgl.add_reverse_edges(
            g=graph,
            copy_ndata=True,
            copy_edata=True,
        )
        reversed_self_loop_graph = self.add_self_loop(graph=reversed_graph)

        edge_collator = dgl.dataloading.EdgeCollator(
            block_sampler=block_sampler,
            negative_sampler=negative_sampler,
            g=reversed_self_loop_graph,
            eids=eids,
        )
        return DataLoader(
            dataset=edge_collator.dataset,
            collate_fn=edge_collator.collate,
            batch_size=self.conf.get("train_batch_size") if phase == "train" else self.conf.get("eval_batch_size"),
            shuffle=phase == "train",
            drop_last=False,
            num_workers=self.conf.get("num_workers"),
            pin_memory=self.conf.get("pin_memory"),
        )

    @staticmethod
    def get_eval_neg_edges(graph: dgl.DGLHeteroGraph) -> Dict[Tuple[str, str, str], Dict[str, torch.Tensor]]:
        pos_src: torch.Tensor
        pos_dst: torch.Tensor
        nodes: np.ndarray = np.arange(graph.num_nodes())
        dict_neg_edges: Dict[Tuple[str, str, str], Dict[str, torch.Tensor]] = {}
        for etype in graph.canonical_etypes:
            print(etype)
            total_pos: int = 0
            pos_src, pos_dst = graph.edges(form="uv", etype=etype)
            unique_src: torch.Tensor = pos_src.unique()
            neg_dst: List[int] = []
            np.random.seed(1)
            src_list: List[int] = []
            for src in unique_src:
                mask: torch.Tensor = pos_src == src
                num_negative: int = int(torch.sum(mask).item())
                total_pos += num_negative
                negative_candidates: List[int] = list(set(nodes) - set(pos_dst[mask].tolist()) - {int(src.item())})
                if len(negative_candidates) >= num_negative:
                    neg_dst_per_src: np.ndarray = np.random.choice(
                        a=np.array(negative_candidates), size=num_negative, replace=False
                    )
                    neg_dst += neg_dst_per_src.tolist()
                else:
                    neg_dst += negative_candidates
                src_list += [src.item()] * (len(neg_dst) - len(src_list))
            print(len(src_list), len(neg_dst))
            if total_pos > len(neg_dst):
                extra_num_negative: int = total_pos - len(neg_dst)
                print("Extra", extra_num_negative)
                np.random.shuffle(nodes)
                for src in tqdm(nodes):
                    mask = pos_src == src
                    existing_neg_edges = np.array(neg_dst)[np.array(src_list) == src].tolist()

                    extra_negative_candidates: np.ndarray = np.array(
                        list(set(nodes) - set(pos_dst[mask].tolist()) - {src.tolist()} - set(existing_neg_edges))
                    )
                    np.random.shuffle(extra_negative_candidates)

                    if extra_negative_candidates.shape[0] > 0:
                        src_list += [src] * len(extra_negative_candidates[: total_pos - len(neg_dst)])
                        neg_dst += extra_negative_candidates[: total_pos - len(neg_dst)].tolist()

                    if total_pos == len(neg_dst):
                        break

            assert total_pos == len(neg_dst), f"Imbalance pos/neg {total_pos}/{len(neg_dst)}"
            dict_neg_edges[etype] = {
                "src": torch.tensor(src_list),
                "dst": torch.tensor(neg_dst),
                "score": torch.zeros_like(pos_src),
            }
        return dict_neg_edges

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        assert self.train_graph is not None
        return self.shared_dataloader(
            graph=self.train_graph,
            phase="train",
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        assert self.valid_graph is not None
        return self.shared_dataloader(
            graph=self.valid_graph,
            phase="valid",
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        assert self.test_graph is not None
        return self.shared_dataloader(
            graph=self.test_graph,
            phase="test",
        )
