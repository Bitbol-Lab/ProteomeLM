"""
Feature extraction pipeline for protein-protein interaction analysis.
"""
import logging
import pickle
from typing import Any, Dict, List, Optional, Tuple
import torch

from .model import prepare_ppi
from .config import ExtractionConfig
from .data_processing import InteractionExtractor


logger = logging.getLogger(__name__)


class PPIFeatureExtractor:
    """Main class for extracting PPI features from protein models."""

    def __init__(self, config: ExtractionConfig, interaction_extractor: Optional[InteractionExtractor] = None):
        """
        Initialize the PPI feature extractor.

        Args:
            config: Configuration for feature extraction
            interaction_extractor: Optional extractor for interaction data
        """
        self.config = config
        self.interaction_extractor = interaction_extractor

    def extract_features(self) -> Dict[str, Any]:
        """
        Extract PPI features using the configured model checkpoint.

        Returns:
            Dictionary containing extracted features for each dataset split
        """
        logger.info(f"Starting PPI feature extraction with checkpoint: {self.config.checkpoint}")

        # Validate configuration
        if self.config.save_path is not None and str(self.config.save_path) == "":
            raise ValueError("save_path cannot be an empty string.")

        # Prepare the PPI features
        fasta_path = str(self.config.env_dir / self.config.fasta_file)
        output = prepare_ppi(
            str(self.config.checkpoint),
            fasta_path,
            encoded_genome_file=str(self.config.encoded_genome_file) if self.config.encoded_genome_file else None,
            keep_heads=None,
            esm_device=self.config.esm_device,
            proteomelm_device=self.config.proteomelm_device,
            use_odb=False,
            include_attention=self.config.include_attention,
            include_all_hidden_states=self.config.include_all_hidden_states,
            reload_if_possible=self.config.reload_if_possible
        )

        # Extract interaction indices and labels
        if self.interaction_extractor is not None:
            index_dict, y_dict = self.interaction_extractor.extract(
                self.config.env_dir, self.config.fasta_file
            )
        else:
            # Generate all possible pairs if no specific interactions are provided
            sequence_length = output["plm_representations"].size(1)
            index_dict = {
                "all": [(i, j) for i in range(sequence_length) for j in range(i + 1, sequence_length)]
            }
            y_dict = {"all": torch.zeros(len(index_dict["all"]), dtype=torch.long)}

        # Process features for each split
        dump_dict = self._process_splits(output, index_dict, y_dict)

        # Save results if path is provided
        if self.config.save_path:
            self._save_results(dump_dict)

        logger.info("PPI feature extraction completed successfully")
        return dump_dict

    def _process_splits(
        self,
        output: Dict[str, torch.Tensor],
        index_dict: Dict[str, List[Tuple[int, int]]],
        y_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Process features for each dataset split.

        Args:
            output: Raw model output
            index_dict: Dictionary of interaction indices per split
            y_dict: Dictionary of labels per split

        Returns:
            Processed features organized by split
        """
        # Flatten all indices for attention processing
        indices_all = []
        for indices in index_dict.values():
            indices_all.extend(indices)

        # Extract attention matrices if requested
        attention_matrix = None
        if self.config.include_attention and "plm_attentions" in output:
            attention_matrix = self._process_attention(output["plm_attentions"], indices_all)

        # Extract core representations
        repr_proteomelm = output["plm_representations"]
        logits_proteomelm = output["plm_logits"]
        all_representations = output.get("plm_all_representations") if self.config.include_all_hidden_states else None
        repr_esm = output["group_embeds"]

        # Process each split
        dump_dict = {}
        cumsum = 0

        for split_name, indices in index_dict.items():
            split_data = self._process_single_split(
                indices, cumsum, attention_matrix, repr_proteomelm,
                logits_proteomelm, all_representations, repr_esm, y_dict[split_name]
            )
            dump_dict[split_name] = split_data
            cumsum += len(indices)

        return dump_dict

    def _process_attention(self, attentions: List[torch.Tensor], indices_all: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Process attention matrices for the given indices.

        Args:
            attentions: List of attention tensors from model layers
            indices_all: All interaction indices

        Returns:
            Processed attention matrix
        """
        indice_0 = [pair[0] for pair in indices_all]
        indice_1 = [pair[1] for pair in indices_all]

        # For each attention layer, extract and symmetrize the pair-wise attention
        att_list = [
            att_layer[:, :, indice_0, indice_1].clone() + att_layer[:, :, indice_1, indice_0].clone()
            for att_layer in attentions
        ]

        # Concatenate layers and permute dimensions to have shape (num_pairs, num_layers, num_heads)
        return torch.cat(att_list, dim=0).permute(2, 0, 1)

    def _process_single_split(
        self,
        indices: List[Tuple[int, int]],
        cumsum: int,
        attention_matrix: Optional[torch.Tensor],
        repr_proteomelm: torch.Tensor,
        logits_proteomelm: torch.Tensor,
        all_representations: Optional[torch.Tensor],
        repr_esm: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Process features for a single dataset split.

        Args:
            indices: Interaction indices for this split
            cumsum: Cumulative sum for attention matrix slicing
            attention_matrix: Processed attention matrix
            repr_proteomelm: ProteomeLM representations
            logits_proteomelm: ProteomeLM logits
            all_representations: All hidden state representations
            repr_esm: ESM representations
            labels: Interaction labels

        Returns:
            Dictionary containing processed features for this split
        """
        if not indices:
            return {
                "A": None,
                "repr_proteomelm": None,
                "repr_esm": None,
                "all_representations": None,
                "logits_proteomelm": None,
                "y": labels
            }

        # Concatenate features for each index pair
        repr_proteomelm_split = torch.cat([
            torch.cat([repr_proteomelm[:, i0], repr_proteomelm[:, i1]], dim=-1)
            for i0, i1 in indices
        ], dim=0)

        repr_esm_split = torch.cat([
            torch.cat([repr_esm[None, i0], repr_esm[None, i1]], dim=-1)
            for i0, i1 in indices
        ], dim=0)

        logits_split = torch.cat([
            torch.cat([logits_proteomelm[:, i0], logits_proteomelm[:, i1]], dim=-1)
            for i0, i1 in indices
        ], dim=0)

        all_representations_split = None
        if all_representations is not None:
            all_representations_split = torch.cat([
                torch.cat([all_representations[:, i0], all_representations[:, i1]], dim=-1)
                for i0, i1 in indices
            ], dim=0)

        attention_split = None
        if attention_matrix is not None:
            attention_split = attention_matrix[cumsum:cumsum + len(indices)].float().contiguous()

        return {
            "A": attention_split,
            "repr_proteomelm": repr_proteomelm_split,
            "repr_esm": repr_esm_split,
            "all_representations": all_representations_split,
            "logits_proteomelm": logits_split,
            "y": labels
        }

    def _save_results(self, dump_dict: Dict[str, Any]) -> None:
        """
        Save extraction results to pickle file.

        Args:
            dump_dict: Dictionary containing processed features
        """
        with open(self.config.save_path, "wb") as f:
            pickle.dump(dump_dict, f)
        logger.info(f"Results saved to {self.config.save_path}")


class FullProteomeExtractor:
    """Extracts full proteome PPI predictions using a trained PPI model."""

    def __init__(self, config: ExtractionConfig, ppi_checkpoint: str):
        """
        Initialize the full proteome extractor.

        Args:
            config: Configuration for feature extraction
            ppi_checkpoint: Path to trained PPI model checkpoint
        """
        self.config = config
        self.ppi_checkpoint = ppi_checkpoint

    def extract_predictions(self) -> torch.Tensor:
        """
        Extract PPI predictions for the full proteome.

        Returns:
            Tensor containing PPI prediction logits
        """
        from proteomelm.ppi.model import EnhancedPPIModel

        logger.info("Starting full proteome PPI prediction extraction")

        # Validate configuration
        if self.config.save_path is not None and str(self.config.save_path) == "":
            raise ValueError("save_path cannot be an empty string.")

        # Prepare features
        fasta_path = str(self.config.env_dir / self.config.fasta_file)
        output = prepare_ppi(
            str(self.config.checkpoint),
            fasta_path,
            encoded_genome_file=str(self.config.encoded_genome_file) if self.config.encoded_genome_file else None,
            keep_heads=None,
            esm_device=self.config.esm_device,
            proteomelm_device=self.config.proteomelm_device,
            use_odb=False,
            include_attention=self.config.include_attention,
            include_all_hidden_states=self.config.include_all_hidden_states,
            reload_if_possible=self.config.reload_if_possible
        )

        # Process attention and representations
        x = output["plm_logits"][0]
        att_list = list(output["plm_attentions"])

        # Concatenate and reshape attention matrices
        A = torch.cat(att_list, dim=0)
        A = A.permute(2, 3, 0, 1)
        A = A.reshape(A.size(0), A.size(1), A.size(-1) * A.size(-2))
        A = A + A.transpose(1, 0)  # Symmetrize

        # Load and apply PPI model
        model = EnhancedPPIModel(protein_embed_dim=x.size(-1), pair_feature_dim=A.size(-1))
        model.load_state_dict(torch.load(self.ppi_checkpoint))
        model = model.to(torch.bfloat16)

        logits = model.evaluate_full_proteome(A, x)

        # Save results if path provided
        if self.config.save_path:
            with open(self.config.save_path, "wb") as f:
                pickle.dump(logits, f)
            logger.info(f"Full proteome predictions saved to {self.config.save_path}")

        return logits
