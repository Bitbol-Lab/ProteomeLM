"""
Feature extraction pipeline for protein-protein interaction analysis.
"""
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch

from .model import prepare_ppi
from .config import ExtractionConfig
from .data_processing import InteractionExtractor
from proteomelm.utils.io import ensure_dir


logger = logging.getLogger(__name__)


class PPIFeatureExtractor:
    """Main class for extracting PPI features from protein models."""

    def __init__(self, config: ExtractionConfig, interaction_extractor: Optional[InteractionExtractor] = None):
        self.config = config
        self.interaction_extractor = interaction_extractor

    def extract_features(self) -> Dict[str, Any]:
        """Extract PPI features using the configured model checkpoint."""
        logger.info("Starting PPI feature extraction with checkpoint: %s", self.config.checkpoint)

        if self.config.save_path is not None and str(self.config.save_path) == "":
            raise ValueError("save_path cannot be an empty string.")

        fasta_path = str(self.config.env_dir / self.config.fasta_file)
        output = prepare_ppi(
            str(self.config.checkpoint),
            fasta_path,
            encoded_genome_file=str(self.config.encoded_genome_file) if self.config.encoded_genome_file else None,
            esm_device=self.config.esm_device,
            proteomelm_device=self.config.proteomelm_device,
            include_attention=self.config.include_attention,
            include_all_hidden_states=self.config.include_all_hidden_states,
            reload_if_possible=self.config.reload_if_possible,
            orthodb_db_path=str(self.config.orthodb_db_path) if self.config.orthodb_db_path else None,
            orthodb_tsv_path=str(self.config.orthodb_tsv_path) if self.config.orthodb_tsv_path else None,
            orthodb_min_group_size=self.config.orthodb_min_group_size,
            orthodb_fetch_online=self.config.orthodb_fetch_online,
        )

        if self.interaction_extractor is not None:
            index_dict, y_dict = self.interaction_extractor.extract(
                self.config.env_dir, self.config.fasta_file
            )
        else:
            n_proteins = output["plm_representations"].size(1)
            index_dict = {
                "all": [(i, j) for i in range(n_proteins) for j in range(i + 1, n_proteins)]
            }
            y_dict = {"all": torch.zeros(len(index_dict["all"]), dtype=torch.long)}

        dump_dict = self._process_splits(output, index_dict, y_dict)

        if self.config.save_path:
            ensure_dir(str(Path(self.config.save_path).parent))
            with open(self.config.save_path, "wb") as f:
                pickle.dump(dump_dict, f)
            logger.info("Features saved to %s", self.config.save_path)

        logger.info("PPI feature extraction completed")
        return dump_dict

    def _process_splits(
        self,
        output: Dict[str, torch.Tensor],
        index_dict: Dict[str, List[Tuple[int, int]]],
        y_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        all_pairs: List[Tuple[int, int]] = []
        for pairs in index_dict.values():
            all_pairs.extend(pairs)

        attentions = output.get("plm_attentions") if self.config.include_attention else None

        attention_matrix: Optional[torch.Tensor] = None
        if attentions is not None:
            attention_matrix = self._process_attention(attentions, all_pairs)

        repr_proteomelm = output["plm_representations"]   # [1, n_proteins, dim]
        logits_proteomelm = output["plm_logits"]          # [1, n_proteins, dim]
        repr_esm = output["group_embeds"]                 # [n_proteins, d_esm]
        all_representations = (
            output.get("plm_all_representations")
            if self.config.include_all_hidden_states else None
        )

        dump_dict: Dict[str, Any] = {}
        cumsum = 0
        for split_name, pairs in index_dict.items():
            dump_dict[split_name] = self._process_single_split(
                pairs, cumsum, attention_matrix,
                repr_proteomelm, logits_proteomelm, all_representations,
                repr_esm, y_dict[split_name],
            )
            cumsum += len(pairs)

        return dump_dict

    # ------------------------------------------------------------------
    # Per-pair attention (direct)
    # ------------------------------------------------------------------

    @staticmethod
    def _process_attention(
        attentions: List[torch.Tensor],
        all_pairs: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """Extract per-pair direct attention from all layers.

        Returns tensor of shape (n_pairs, n_layers, n_heads).
        """
        if not all_pairs:
            return torch.empty(0, len(attentions), attentions[0].shape[1])

        i0 = [p[0] for p in all_pairs]
        i1 = [p[1] for p in all_pairs]

        att_list = [
            att_layer[:, :, i0, i1] + att_layer[:, :, i1, i0]
            for att_layer in attentions
        ]
        return torch.stack([a.squeeze(0) for a in att_list], dim=0).permute(2, 0, 1)

    # ------------------------------------------------------------------
    # Per-split feature assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _process_single_split(
        pairs: List[Tuple[int, int]],
        cumsum: int,
        attention_matrix: Optional[torch.Tensor],
        repr_proteomelm: torch.Tensor,
        logits_proteomelm: torch.Tensor,
        all_representations: Optional[torch.Tensor],
        repr_esm: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, Any]:
        if not pairs:
            return {
                "A": None,
                "repr_proteomelm": None, "repr_esm": None,
                "all_representations": None, "logits_proteomelm": None, "y": labels,
            }

        i0 = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        i1 = torch.tensor([p[1] for p in pairs], dtype=torch.long)

        repr_proteomelm_split = torch.cat(
            [repr_proteomelm[0, i0], repr_proteomelm[0, i1]], dim=-1
        )
        repr_esm_split = torch.cat(
            [repr_esm[i0], repr_esm[i1]], dim=-1
        )
        logits_split = torch.cat(
            [logits_proteomelm[0, i0], logits_proteomelm[0, i1]], dim=-1
        )

        all_repr_split: Optional[torch.Tensor] = None
        if all_representations is not None:
            all_repr_split = torch.cat(
                [all_representations[:, i0, :], all_representations[:, i1, :]], dim=-1
            )

        attention_split: Optional[torch.Tensor] = None
        if attention_matrix is not None:
            attention_split = attention_matrix[cumsum : cumsum + len(pairs)].float().contiguous()

        return {
            "A": attention_split,
            "repr_proteomelm": repr_proteomelm_split,
            "repr_esm": repr_esm_split,
            "all_representations": all_repr_split,
            "logits_proteomelm": logits_split,
            "y": labels,
        }

