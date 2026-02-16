import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import warnings
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import average_precision_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import optuna

from proteomelm.modeling_proteomelm import ProteomeLMForMaskedLM
from proteomelm.utils.embedding import build_genome_esmc
from proteomelm.utils.proteome import (
    load_orthodb_group_vectors,
    build_group_embeddings_for_proteome,
    download_orthodb_tsv_for_accessions,
)

# Import embedding utilities
sys.path.append(str(Path(__file__).resolve().parents[2]))
from proteomelm.utils.io import ensure_dir, parse_fasta

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------
# PPI Model
# --------------------------

class EnhancedPPIModel(nn.Module):
    def __init__(self,
                 protein_embed_dim=640,
                 pair_feature_dim=48,
                 # Protein branch parameters
                 protein_layer1_dim=512,
                 protein_layer2_dim=256,
                 dropout_protein1=0.3,
                 dropout_protein2=0.4,
                 # Pair branch parameters
                 pair_layer1_dim=64,
                 pair_layer2_dim=32,
                 dropout_pair=0.1,
                 # Interaction processor parameters (input: protein_layer2_dim*2)
                 interaction_layer1_dim=128,
                 interaction_layer2_dim=64,
                 dropout_interaction=0.3,
                 # Classifier parameters (input: attention_gate_dim + pair_layer2_dim)
                 classifier_layer1_dim=128,
                 classifier_layer2_dim=64,
                 dropout_classifier1=0.2,
                 dropout_classifier2=0.1,
                 **kwargs):
        super().__init__()
        self.protein_embed_dim = protein_embed_dim
        self.pair_feature_dim = pair_feature_dim

        # Protein branch
        self.protein_branch = nn.Sequential(
            nn.Linear(protein_embed_dim, protein_layer1_dim),
            nn.LayerNorm(protein_layer1_dim),
            nn.ReLU(),
            nn.Dropout(dropout_protein1),
            nn.Linear(protein_layer1_dim, protein_layer2_dim),
            nn.LayerNorm(protein_layer2_dim),
            nn.ReLU(),
            nn.Dropout(dropout_protein2)
        )

        # Pair branch
        self.pair_branch = nn.Sequential(
            nn.Linear(pair_feature_dim, pair_layer1_dim),
            nn.LayerNorm(pair_layer1_dim),
            nn.ReLU(),
            nn.Dropout(dropout_pair),
            nn.Linear(pair_layer1_dim, pair_layer2_dim),
            nn.LayerNorm(pair_layer2_dim),
            nn.ReLU()
        )

        # Interaction processor (combining element-wise multiplication and absolute difference)
        self.interaction_processor = nn.Sequential(
            nn.Linear(protein_layer2_dim * 4, interaction_layer1_dim),
            nn.LayerNorm(interaction_layer1_dim),
            nn.ReLU(),
            nn.Dropout(dropout_interaction),
            nn.Linear(interaction_layer1_dim, interaction_layer2_dim),
            nn.LayerNorm(interaction_layer2_dim),
            nn.ReLU()
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(interaction_layer2_dim + pair_layer2_dim, classifier_layer1_dim),
            nn.LayerNorm(classifier_layer1_dim),
            nn.ReLU(),
            nn.Dropout(dropout_classifier1),
            nn.Linear(classifier_layer1_dim, classifier_layer2_dim),
            nn.LayerNorm(classifier_layer2_dim),
            nn.ReLU(),
            nn.Dropout(dropout_classifier2),
            nn.Linear(classifier_layer2_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, F=None, E1=None, E2=None):
        if F is None:
            F = torch.zeros((E1.shape[0], self.pair_feature_dim)).to(E1.device, dtype=E1.dtype)
        if E1 is None:
            E1 = torch.zeros((F.shape[0], self.protein_embed_dim)).to(F.device)
        if E2 is None:
            E2 = torch.zeros((F.shape[0], self.protein_embed_dim)).to(F.device)

        # Process individual proteins
        E1_processed = self.protein_branch(E1)
        E2_processed = self.protein_branch(E2)

        # Create interaction features from element-wise multiplication and absolute difference
        interaction_terms = torch.cat([
            E1_processed, E2_processed,
            E1_processed * E2_processed,
            (E1_processed - E2_processed).abs()
        ], dim=1)

        # Process interactions
        interaction_features = self.interaction_processor(interaction_terms)

        # Process pair features
        pair_features = self.pair_branch(F)

        # Final classification
        final_combined = torch.cat([interaction_features, pair_features], dim=1)
        output = self.classifier(final_combined)
        return output  # torch.sigmoid(output)

    def evaluate_full_proteome(self, A=None, x=None):
        """
        Evaluate the model on the full proteome.
        Args:
            A: Pairwise features (edges).
            x: Protein embeddings.

        Returns:
            torch.Tensor: Predicted logits for the pairs.
        """
        n_nodes = x.shape[0]
        n_edges = n_nodes*n_nodes
        if A is not None:
            A = A.view(-1, self.pair_feature_dim)
            pair_features = self.pair_branch(A).reshape(n_edges, -1)
        else:
            pair_features = torch.zeros((n_edges, self.pair_feature_dim)).to(x.device, dtype=x.dtype)
        # Process individual proteins
        x_processed = self.protein_branch(x)
        # Create interaction features from element-wise multiplication and absolute difference

        interaction_terms = torch.cat([
            x_processed.unsqueeze(1).expand(-1, n_nodes, -1),
            x_processed.unsqueeze(0).expand(n_nodes, -1, -1),
            x_processed.unsqueeze(1).expand(-1, n_nodes, -1) * x_processed.unsqueeze(0).expand(n_nodes, -1, -1),
            (x_processed.unsqueeze(1).expand(-1, n_nodes, -1) - x_processed.unsqueeze(0).expand(n_nodes, -1, -1)).abs()
        ], dim=-1).view(n_edges, -1)

        # Process interactions
        interaction_features = self.interaction_processor(interaction_terms)
        # Final classification
        final_combined = torch.cat([interaction_features, pair_features], dim=-1)
        output = self.classifier(final_combined)
        return output.view(n_nodes, n_nodes)  # torch.sigmoid(output)
    

class SimpleMLP(nn.Module):
    """Simple MLP for protein-protein interaction prediction."""
    
    def __init__(self, protein_embed_dim, pair_feature_dim, hidden_dims=[64], dropout=0.3):
        super(SimpleMLP, self).__init__()
        
        layers = []
        prev_dim = 2 * protein_embed_dim + pair_feature_dim
        self.protein_embed_dim = protein_embed_dim
        self.pair_feature_dim = pair_feature_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, F=None, E1=None, E2=None):
        if F is None:
            F = torch.zeros((E1.shape[0], self.pair_feature_dim)).to(E1.device, dtype=E1.dtype)
        if E1 is None:
            E1 = torch.zeros((F.shape[0], self.protein_embed_dim)).to(F.device)
        if E2 is None:
            E2 = torch.zeros((F.shape[0], self.protein_embed_dim)).to(F.device)
        combined = torch.cat([E1, E2, F], dim=1)
        output = self.network(combined)
        return output  # torch.sigmoid(output)
    
    def evaluate_full_proteome(self, A=None, x=None):
        """
        Evaluate the model on the full proteome.
        Args:
            A: Pairwise features (edges).
            x: Protein embeddings.

        Returns:
            torch.Tensor: Predicted logits for the pairs.
        """
        n_nodes = x.shape[0]
        n_edges = n_nodes*n_nodes
        if A is not None:
            A = A.view(-1, self.pair_feature_dim)
        else:
            A = torch.zeros((n_edges, self.pair_feature_dim)).to(x.device, dtype=x.dtype)

        combined = torch.cat([
            x.unsqueeze(1).expand(-1, n_nodes, -1) * x.unsqueeze(0).expand(n_nodes, -1, -1),
            (x.unsqueeze(1).expand(-1, n_nodes, -1) - x.unsqueeze(0).expand(n_nodes, -1, -1)).abs(),
            A.view(n_nodes, n_nodes, -1)
        ], dim=-1).view(n_edges, -1)

        output = self.network(combined)
        return output.view(n_nodes, n_nodes)  # torch.sigmoid(output)

# --------------------------
# Loss Functions
# --------------------------

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification, helps improve precision-recall."""
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce_loss).mean()


class AUPRLoss(nn.Module):
    """Differentiable approximation of AUPR loss using pairwise ranking."""
    def __init__(self, num_samples=100):
        super().__init__()
        self.num_samples = num_samples
    
    def forward(self, logits, targets):
        # Separate positive and negative samples
        pos_mask = (targets == 1).squeeze()
        neg_mask = (targets == 0).squeeze()
        
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            # Fallback to BCE if only one class present
            return nn.functional.binary_cross_entropy_with_logits(logits, targets)
        
        pos_logits = logits[pos_mask]
        neg_logits = logits[neg_mask]
        
        # Sample pairs to make computation tractable
        n_pos = min(len(pos_logits), self.num_samples)
        n_neg = min(len(neg_logits), self.num_samples)
        
        if n_pos < len(pos_logits):
            pos_idx = torch.randperm(len(pos_logits))[:n_pos]
            pos_logits = pos_logits[pos_idx]
        if n_neg < len(neg_logits):
            neg_idx = torch.randperm(len(neg_logits))[:n_neg]
            neg_logits = neg_logits[neg_idx]
        
        # Compute pairwise ranking loss (approximates AUPR)
        # For each positive, it should rank higher than negatives
        pos_expanded = pos_logits.unsqueeze(1)  # [n_pos, 1]
        neg_expanded = neg_logits.unsqueeze(0)  # [1, n_neg]
        
        # Hinge loss: max(0, margin - (pos - neg))
        margin = 1.0
        ranking_loss = torch.relu(margin - (pos_expanded - neg_expanded))
        
        return ranking_loss.mean()


class CombinedLoss(nn.Module):
    """Combines BCE and AUPR loss for balanced optimization."""
    def __init__(self, bce_weight=0.5, aupr_weight=0.5, pos_weight=None, focal=False, focal_gamma=2.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.aupr_weight = aupr_weight
        self.pos_weight = pos_weight
        self.focal = focal
        if focal:
            self.bce_loss = FocalLoss(alpha=0.25, gamma=focal_gamma, pos_weight=pos_weight)
        else:
            self.bce_loss = lambda logits, targets: nn.functional.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight
            )
        self.aupr_loss = AUPRLoss()
    
    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets)
        aupr = self.aupr_loss(logits, targets)
        return self.bce_weight * bce + self.aupr_weight * aupr


# --------------------------
# PPI Model Training
# --------------------------


def train_model_cv(
    X_train, X_test, y_train, y_test,
    n_epochs=50,
    patience=5,
    model_type="simplemlp",
    model_params=None,
    verbose=True,
    replica_seed: int = 0,
    # New parameters
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
    augment_symmetric: bool = True,
    use_class_weights: bool = True,
    scheduler_patience: int = 3,
    scheduler_factor: float = 0.5,
    # Loss function selection
    loss_type: str = "bce",  # Options: "bce", "focal", "aupr", "combined"
    focal_gamma: float = 2.0,
    combined_bce_weight: float = 0.5,
    combined_aupr_weight: float = 0.5,
):
    """
    Improved PPI model training with:
    - Class imbalance handling (pos_weight)
    - Learning rate scheduling (ReduceLROnPlateau)
    - AdamW optimizer with weight decay
    - Optional symmetric data augmentation
    - Gradient clipping
    - Comprehensive seeding for reproducibility
    """
    # Comprehensive seed fixing for reproducibility
    seed = 42 + replica_seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        # Set deterministic mode for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create generator for DataLoader reproducibility
    g = torch.Generator()
    g.manual_seed(seed)
    
    def seed_worker(worker_id):
        """Seed worker for DataLoader to ensure reproducibility."""
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        import random
        random.seed(worker_seed)

    PPIModel: nn.Module = EnhancedPPIModel if model_type == "enhancedppi" else SimpleMLP

    # Convert inputs to tensors
    f_train_tensor = (
        torch.tensor(X_train["edges"], dtype=torch.float32)
        .view(X_train["edges"].shape[0], -1)
        .to(device)
    ) if X_train["edges"] is not None else torch.zeros((len(y_train), 1)).to(device)
    
    e1_train_tensor = (
        torch.tensor(X_train["x1"], dtype=torch.float32).to(device)
    ) if X_train["x1"] is not None else torch.zeros((len(y_train), 1)).to(device)
    
    e2_train_tensor = (
        torch.tensor(X_train["x2"], dtype=torch.float32).to(device)
    ) if X_train["x2"] is not None else torch.zeros((len(y_train), 1)).to(device)

    f_test_tensor = (
        torch.tensor(X_test["edges"], dtype=torch.float32)
        .view(X_test["edges"].shape[0], -1)
        .to(device)
    ) if X_test["edges"] is not None else torch.zeros((len(y_test), 1)).to(device)
    
    e1_test_tensor = (
        torch.tensor(X_test["x1"], dtype=torch.float32).to(device)
    ) if X_test["x1"] is not None else torch.zeros((len(y_test), 1)).to(device)
    
    e2_test_tensor = (
        torch.tensor(X_test["x2"], dtype=torch.float32).to(device)
    ) if X_test["x2"] is not None else torch.zeros((len(y_test), 1)).to(device)

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    # --- Data Augmentation: Swap protein pairs (interactions are symmetric) ---
    if augment_symmetric:
        e1_train_aug = torch.cat([e1_train_tensor, e2_train_tensor], dim=0)
        e2_train_aug = torch.cat([e2_train_tensor, e1_train_tensor], dim=0)
        f_train_aug = torch.cat([f_train_tensor, f_train_tensor], dim=0)
        y_train_aug = torch.cat([y_train_tensor, y_train_tensor], dim=0)
        if verbose:
            print(f"Data augmentation: {len(y_train_tensor)} -> {len(y_train_aug)} samples")
    else:
        e1_train_aug = e1_train_tensor
        e2_train_aug = e2_train_tensor
        f_train_aug = f_train_tensor
        y_train_aug = y_train_tensor

    embed_dim = e1_train_tensor.shape[-1] if e1_train_tensor is not None else 1
    edges_dim = f_train_tensor.shape[-1] if f_train_tensor is not None else 1
    if verbose:
        print(f"Embed dim: {embed_dim}, Edges dim: {edges_dim}")

    if model_params is None:
        model_params = {}
    
    lr = model_params.get("lr", 0.001)
    
    # Reset random state before model initialization to ensure reproducible weight initialization
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    model = PPIModel(protein_embed_dim=embed_dim, pair_feature_dim=edges_dim, **model_params).to(device)

    # --- Class imbalance handling and loss function selection ---
    pos_weight = None
    if use_class_weights:
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
        if verbose:
            print(f"Class distribution: {n_neg} negatives, {n_pos} positives (ratio: {n_neg/n_pos:.2f})")
            print(f"Using pos_weight: {pos_weight.item():.2f}")
    
    # Select loss function
    if loss_type == "focal":
        loss_fn = FocalLoss(alpha=0.25, gamma=focal_gamma, pos_weight=pos_weight)
        if verbose:
            print(f"Using Focal Loss with gamma={focal_gamma}")
    elif loss_type == "aupr":
        loss_fn = AUPRLoss()
        if verbose:
            print("Using AUPR Loss (pairwise ranking)")
    elif loss_type == "combined":
        loss_fn = CombinedLoss(
            bce_weight=combined_bce_weight,
            aupr_weight=combined_aupr_weight,
            pos_weight=pos_weight,
            focal=False
        )
        if verbose:
            print(f"Using Combined Loss (BCE: {combined_bce_weight}, AUPR: {combined_aupr_weight})")
    elif loss_type == "combined_focal":
        loss_fn = CombinedLoss(
            bce_weight=combined_bce_weight,
            aupr_weight=combined_aupr_weight,
            pos_weight=pos_weight,
            focal=True,
            focal_gamma=focal_gamma
        )
        if verbose:
            print(f"Using Combined Focal+AUPR Loss (Focal: {combined_bce_weight}, AUPR: {combined_aupr_weight}, gamma: {focal_gamma})")
    else:  # default "bce"
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if verbose:
            print("Using BCE Loss")

    # --- AdamW optimizer with weight decay ---
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- Learning rate scheduler ---
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=scheduler_factor, patience=scheduler_patience
    )

    # Create DataLoaders with proper seeding for reproducibility
    train_dataset = TensorDataset(f_train_aug, e1_train_aug, e2_train_aug, y_train_aug)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_dataset = TensorDataset(f_test_tensor, e1_test_tensor, e2_test_tensor, y_test_tensor)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        worker_init_fn=seed_worker
    )

    train_losses, val_losses, auprs, lrs = [], [], [], []
    best_aupr = 0
    epochs_no_improve = 0
    best_model = None

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0

        for f, e1, e2, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(f, e1, e2)
            loss = loss_fn(outputs, labels)
            loss.backward()
            
            # --- Gradient clipping ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation step
        model.eval()
        eval_loss = 0.0
        y_pred_list = []
        y_true_list = []
        with torch.no_grad():
            for f, e1, e2, labels in val_loader:
                outputs = model(f, e1, e2)
                loss = loss_fn(outputs, labels)
                eval_loss += loss.item()
                y_pred_list.extend(outputs.cpu().numpy())
                y_true_list.extend(labels.cpu().numpy())

        aupr = average_precision_score(y_true_list, y_pred_list)
        val_loss = eval_loss / len(val_loader)
        val_losses.append(val_loss)
        auprs.append(aupr)
        lrs.append(optimizer.param_groups[0]['lr'])

        if verbose:
            print(
                f"Epoch {epoch + 1}/{n_epochs}, "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"AUPR: {aupr:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

        # --- Learning rate scheduling ---
        scheduler.step(train_loss)

        # Early stopping check (based on AUPR)
        if aupr > best_aupr:
            best_aupr = aupr
            best_model = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}!")
                break

    # Load the best model for final evaluation
    model.load_state_dict(best_model)
    model.eval()
    
    with torch.no_grad():
        logits = model(f_test_tensor, e1_test_tensor, e2_test_tensor).cpu().numpy().flatten()
    
    # Convert logits to probabilities for proper thresholding
    y_pred_proba = 1 / (1 + np.exp(-logits))  # sigmoid
    y_pred_bin = (y_pred_proba >= 0.5).astype(int)
    y_test_bin = y_test_tensor.cpu().numpy().flatten().astype(int)

    print("\n" + "="*50)
    print("Final Evaluation on Test Set")
    print("="*50)
    print(classification_report(y_test_bin, y_pred_bin, target_names=["Class 0", "Class 1"]))
    
    auc_final = roc_auc_score(y_test_bin, y_pred_proba)
    aupr_final = average_precision_score(y_test_bin, y_pred_proba)
    print(f"AUC Score:  {auc_final:.4f}")
    print(f"AUPR Score: {aupr_final:.4f}")

    # Plot losses and metrics
    if verbose:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].plot(train_losses, label="Train Loss")
        axes[0].plot(val_losses, label="Validation Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].set_title("Training & Validation Loss")
        
        axes[1].plot(auprs, label="Validation AUPR", color="green")
        axes[1].axhline(y=best_aupr, color='r', linestyle='--', label=f"Best AUPR: {best_aupr:.4f}")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("AUPR")
        axes[1].legend()
        axes[1].set_title("Validation AUPR")
        
        axes[2].plot(lrs, label="Learning Rate", color="orange")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning Rate")
        axes[2].set_yscale('log')
        axes[2].legend()
        axes[2].set_title("Learning Rate Schedule")
        
        plt.tight_layout()
        plt.show()

    return model.cpu(), {"auc": auc_final, "aupr": aupr_final, "best_aupr": best_aupr}


def test_model_cv(model, X_test, y_test):
    """Test a trained model on held-out data."""
    # Convert data to PyTorch tensors
    f_test_tensor = (
        torch.tensor(X_test["edges"], dtype=torch.float32)
        .view(X_test["edges"].shape[0], -1)
        .to(device)
    ) if X_test["edges"] is not None else torch.zeros((len(y_test), 1)).to(device)
    
    e1_test_tensor = (
        torch.tensor(X_test["x1"], dtype=torch.float32).to(device)
    ) if X_test["x1"] is not None else torch.zeros((len(y_test), 1)).to(device)
    
    e2_test_tensor = (
        torch.tensor(X_test["x2"], dtype=torch.float32).to(device)
    ) if X_test["x2"] is not None else torch.zeros((len(y_test), 1)).to(device)
    
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Final evaluation
    model = model.to(device).eval()
    
    with torch.no_grad():
        logits = model(f_test_tensor, e1_test_tensor, e2_test_tensor).cpu().numpy().flatten()
    
    # Convert logits to probabilities
    y_pred_proba = 1 / (1 + np.exp(-logits))
    y_test_bin = y_test_tensor.numpy().flatten().astype(int)

    auc = roc_auc_score(y_test_bin, y_pred_proba)
    aupr = average_precision_score(y_test_bin, y_pred_proba)

    print(f"AUC Score:  {auc:.4f}")
    print(f"AUPR Score: {aupr:.4f}")

    return X_test, y_test, model, {"auc": auc, "aupr": aupr}


def prepare_ppi(checkpoint: Union[Path, str],
                fasta_file: Union[Path, str],
                encoded_genome_file: Optional[Union[Path, str]] = None,
                keep_heads: Optional[List[int]] = None,
                esm_device: str = "cuda:1",
                proteomelm_device: str = "cpu",
                include_attention: bool = False,
                include_all_hidden_states: bool = False,
                reload_if_possible: bool = False,
                use_odb: bool = False,  # TODO: use odb on the fly
                orthodb_db_path: Optional[Union[Path, str]] = None,
                orthodb_tsv_path: Optional[Union[Path, str]] = None,
                orthodb_min_group_size: int = 0,
                orthodb_fetch_online: bool = True,
                ) -> Dict[str, Any]:
    """
    Prepares input data and runs the ProteomeLM model.

    Args:
        checkpoint (Union[Path, str]): Path to the model checkpoint.
        fasta_file (Union[Path, str]): Path to the FASTA file containing the sequences.
        encoded_genome_file (Union[Path, str], optional): Path to the encoded genome file. Defaults to None.
        keep_heads (Optional[List[int]], optional): List of heads to keep. Defaults to None.
        esm_device (str, optional): Device to use for embedding generation. Defaults to "cuda:1".
        proteomelm_device (str, optional): Device to use for ProteomeLM inference. Defaults to "cpu".
        include_attention (bool, optional): Whether to include attention matrices in the output. Defaults to False.
        include_all_hidden_states (bool, optional): Whether to include all hidden states in the output. Defaults to False.
        reload_if_possible (bool, optional): Whether to reload the encoded genome file if possible. Defaults to False.
        use_odb (bool, optional): Whether to use the ODB database for sequence embedding. Defaults to False.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing original input data and model outputs.
    """
    fasta_path = Path(fasta_file)
    assert fasta_path.exists(), f"FASTA file {fasta_path} does not exist."

    # Build genome embeddings from sequences
    if encoded_genome_file is not None and Path(encoded_genome_file).exists() and reload_if_possible:
        data = torch.load(encoded_genome_file)
    else:
        with torch.no_grad():
            data = build_genome_esmc(fasta_path, device=esm_device)
        if encoded_genome_file is not None:
            # Ensure output directory exists
            ensure_dir(str(Path(encoded_genome_file).parent))
            torch.save(data, encoded_genome_file)
    assert "inputs_embeds" in data and "group_embeds" in data, "Generated data missing required keys."

    # Optional: build OrthoDB functional group embeddings
    if orthodb_db_path:
        orthodb_db_path = str(orthodb_db_path)
        if "," in orthodb_db_path:
            orthodb_db_path = orthodb_db_path.split(",")[0]
        if orthodb_db_path.endswith(".pkl"):
            orthodb_db_path = str(Path(orthodb_db_path).parent)

        tsv_path = str(orthodb_tsv_path) if orthodb_tsv_path else None
        if tsv_path is None:
            tsv_path = str(fasta_path.with_suffix("")) + "_orthodb.tsv"

        if not Path(tsv_path).exists() and orthodb_fetch_online:
            sequences = parse_fasta(str(fasta_path))
            accessions = list(sequences.keys())
            logger = logging.getLogger(__name__)
            logger.info(f"Fetching OrthoDB mappings for {len(accessions)} accessions...")
            try:
                download_orthodb_tsv_for_accessions(accessions, tsv_path)
            except RuntimeError as e:
                logger.warning(f"OrthoDB TSV download failed: {e}")

        if Path(tsv_path).exists() and Path(tsv_path).stat().st_size == 0:
            logger = logging.getLogger(__name__)
            logger.warning("OrthoDB TSV is empty; using ESM group embeddings")
            tsv_path = None

        if tsv_path and Path(tsv_path).exists() and Path(orthodb_db_path).exists():
            orthodb_means = load_orthodb_group_vectors(
                orthodb_db_path,
                min_group_size=orthodb_min_group_size,
            )
            group_embeds, mask = build_group_embeddings_for_proteome(
                fasta_path=str(fasta_path),
                orthodb_tsv_path=tsv_path,
                orthodb_group_means=orthodb_means,
                esm_embeddings=data["inputs_embeds"],
            )
            if mask.sum().item() == 0:
                logger = logging.getLogger(__name__)
                logger.warning("OrthoDB TSV has no mappings; using ESM group embeddings")
            else:
                data["group_embeds"] = group_embeds
                data["orthodb_mask"] = mask
        else:
            logger = logging.getLogger(__name__)
            logger.warning("OrthoDB TSV or group vectors missing; using ESM group embeddings")

    # Run ProteomeLM model on the generated data
    # Handle both local paths and Hugging Face model identifiers
    if "/" in str(checkpoint) and not Path(checkpoint).exists():
        # Likely a Hugging Face model identifier (e.g., "BitbolLab/ProteomeLM-S")
        checkpoint_path = str(checkpoint)
    else:
        # Local path
        checkpoint_path = Path(checkpoint)
        assert checkpoint_path.exists(), f"Checkpoint {checkpoint_path} does not exist."
        checkpoint_path = str(checkpoint_path)

    # Ensure input data has required keys
    assert "inputs_embeds" in data and "group_embeds" in data, "Data must contain 'inputs_embeds' and 'group_embeds'."
    assert isinstance(data["inputs_embeds"], torch.Tensor) and isinstance(data["group_embeds"], torch.Tensor), \
        "Inputs must be torch.Tensor arrays."

    head_mask = None  # TODO generalization
    if keep_heads is not None:
        assert isinstance(keep_heads, list), "keep_heads must be a list of integers."
        assert all(isinstance(x, int) for x in keep_heads), "keep_heads must be a list of integers."
        head_mask = torch.zeros(18 * 12, dtype=torch.bool)
        head_mask[keep_heads] = True
        head_mask = head_mask.reshape(18, 12)

    # Load model
    model = ProteomeLMForMaskedLM.from_pretrained(checkpoint_path)

    model = model.to(dtype=torch.bfloat16, device=proteomelm_device).eval()
    with torch.no_grad():
        inputs_embeds = data["inputs_embeds"][None].to(proteomelm_device, dtype=torch.bfloat16)
        group_embeds = data["group_embeds"][None].to(proteomelm_device, dtype=torch.bfloat16)

        output = model(inputs_embeds=inputs_embeds,
                       group_embeds=group_embeds,
                       head_mask=head_mask,
                       output_attentions=include_attention,
                       output_hidden_states=include_all_hidden_states)
        attentions = None
        if include_attention:
            attentions = [x.cpu() for x in output.attentions]
        representations = output.last_hidden_states.cpu()
        logits = output.logits.cpu()
        all_representations = None
        if include_all_hidden_states:
            all_representations = torch.cat([x.cpu() for x in output.hidden_states], 0)
    data["plm_attentions"] = attentions
    data["plm_representations"] = representations
    data["plm_logits"] = logits
    data["plm_all_representations"] = all_representations
    return data


# --------------------------
# Optuna Hyperparameter Optimization
# --------------------------

def objective(trial, X_train, X_test, y_train, y_test):
    # Sample training hyperparameters.
    lr = trial.suggest_loguniform('lr', 0.00025, 0.00025)

    # Sample internal model parameters.
    protein_layer1_dim = trial.suggest_categorical('protein_layer1_dim', [512, 1024, 2048])
    protein_layer2_dim = trial.suggest_categorical('protein_layer2_dim', [256, 512, 1024])
    dropout_protein1 = trial.suggest_float('dropout_protein1', 0.3, 0.3)
    dropout_protein2 = trial.suggest_float('dropout_protein2', 0.4, 0.4)

    pair_layer1_dim = trial.suggest_categorical('pair_layer1_dim', [64, 128, 256, 512])
    pair_layer2_dim = trial.suggest_categorical('pair_layer2_dim', [32, 64, 128, 256])
    dropout_pair = trial.suggest_float('dropout_pair', 0.1, 0.4)

    interaction_layer1_dim = trial.suggest_categorical('interaction_layer1_dim', [128])
    interaction_layer2_dim = trial.suggest_categorical('interaction_layer2_dim', [64])
    dropout_interaction = trial.suggest_float('dropout_interaction', 0.2, 0.5)

    classifier_layer1_dim = trial.suggest_categorical('classifier_layer1_dim', [128, 256, 512])
    classifier_layer2_dim = trial.suggest_categorical('classifier_layer2_dim', [32, 64, 128])
    dropout_classifier1 = trial.suggest_float('dropout_classifier1', 0.2, 0.2)
    dropout_classifier2 = trial.suggest_float('dropout_classifier2', 0.1, 0.1)

    model_params = {
        'lr': lr,
        'protein_layer1_dim': protein_layer1_dim,
        'protein_layer2_dim': protein_layer2_dim,
        'dropout_protein1': dropout_protein1,
        'dropout_protein2': dropout_protein2,
        'pair_layer1_dim': pair_layer1_dim,
        'pair_layer2_dim': pair_layer2_dim,
        'dropout_pair': dropout_pair,
        'interaction_layer1_dim': interaction_layer1_dim,
        'interaction_layer2_dim': interaction_layer2_dim,
        'dropout_interaction': dropout_interaction,
        'classifier_layer1_dim': classifier_layer1_dim,
        'classifier_layer2_dim': classifier_layer2_dim,
        'dropout_classifier1': dropout_classifier1,
        'dropout_classifier2': dropout_classifier2
    }

    print("Trial with parameters:")
    print(f"lr: {lr}")
    print(model_params)

    # Train the model with the sampled hyperparameters.
    # Assumes that train_model_cv and data (X_train, X_test, y_train, y_test, device) are defined.
    _, metrics = train_model_cv(
        X_train, X_test, y_train, y_test,
        n_epochs=100,
        patience=5,
        model_params=model_params,
        verbose=False,
    )

    # Since we want to maximize AUC, we return its negative value (Optuna minimizes the objective).
    return -metrics['aupr']


def main_hyperparam_optimization(X_train, X_test, y_train, y_test):
    # Create an Optuna study and optimize.
    study = optuna.create_study()
    study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test), n_trials=50)

    print("Best hyperparameters found:")
    print(study.best_params)
    print("Best AUC:", -study.best_value)
