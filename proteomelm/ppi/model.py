import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import average_precision_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import optuna

from proteomelm.modeling_proteomelm import ProteomeLMForMaskedLM
from proteomelm.utils import build_genome_esmc

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


# --------------------------
# PPI Model Training
# --------------------------


def train_model_cv(X_train, X_test, y_train, y_test, n_epochs=50, patience=5, model_params=None,
                   verbose=True, replica_seed: int = 0):
    # Fix seeds for reproducibility
    torch.manual_seed(42 + replica_seed)
    np.random.seed(42 + replica_seed)

    # Convert inputs to tensors
    f_train_tensor = (torch.tensor(X_train["edges"], dtype=torch.float32)
                      .view(X_train["edges"].shape[0], -1)
                      .to(device)) if X_train["edges"] is not None else torch.zeros((len(y_train), 1)).to(device)
    e1_train_tensor = torch.tensor(X_train["x1"], dtype=torch.float32).to(device) if X_train[
                                                                                         "x1"] is not None else torch.zeros(
        (len(y_train), 1)).to(device)
    e2_train_tensor = torch.tensor(X_train["x2"], dtype=torch.float32).to(device) if X_train[
                                                                                         "x2"] is not None else torch.zeros(
        (len(y_train), 1)).to(device)

    f_test_tensor = (torch.tensor(X_test["edges"], dtype=torch.float32)
                     .view(X_test["edges"].shape[0], -1)
                     .to(device)) if X_test["edges"] is not None else torch.zeros((len(y_test), 1)).to(device)
    e1_test_tensor = torch.tensor(X_test["x1"], dtype=torch.float32).to(device) if X_test[
                                                                                       "x1"] is not None else torch.zeros(
        (len(y_test), 1)).to(device)
    e2_test_tensor = torch.tensor(X_test["x2"], dtype=torch.float32).to(device) if X_test[
                                                                                       "x2"] is not None else torch.zeros(
        (len(y_test), 1)).to(device)

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    embed_dim = e1_train_tensor.shape[-1] if e1_train_tensor is not None else 1
    edges_dim = f_train_tensor.shape[-1] if f_train_tensor is not None else 1
    if verbose:
        print(f"Embed dim: {embed_dim}, Edges dim: {edges_dim}")

    if model_params is None:
        model_params = {}
    # Initialize the model with both input dimensions and hyperparameters for inner layers
    lr = model_params.get("lr", 0.0005)
    model = EnhancedPPIModel(protein_embed_dim=embed_dim, pair_feature_dim=edges_dim, **model_params).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create DataLoaders
    train_dataset = TensorDataset(f_train_tensor, e1_train_tensor, e2_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_dataset = TensorDataset(f_test_tensor, e1_test_tensor, e2_test_tensor, y_test_tensor)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    train_losses, val_losses, aucs = [], [], []
    best_auc = 0
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
        auc = average_precision_score(y_true_list, y_pred_list)
        val_loss = eval_loss / len(val_loader)
        val_losses.append(val_loss)
        aucs.append(auc)
        if verbose:
            print(
                f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, AUPR: {auc:.4f}")

        # Early stopping check
        if auc > best_auc:
            best_auc = auc
            best_model = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break

    # Load the best model for final evaluation
    model.load_state_dict(best_model)
    y_pred = model(f_test_tensor, e1_test_tensor, e2_test_tensor).detach().cpu().numpy().flatten()
    y_pred_bin = (y_pred >= 0.5).astype(int)
    y_test_bin = y_test_tensor.cpu().numpy().flatten().astype(int)

    print(classification_report(y_test_bin, y_pred_bin, target_names=["Class 0", "Class 1"]))
    auc_final = roc_auc_score(y_test_bin, y_pred)
    aupr_final = average_precision_score(y_test_bin, y_pred)
    print(f"AUC Score: {auc_final:.4f}")
    print(f"AUPR Score: {aupr_final:.4f}")

    # Plot losses
    if verbose:
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training & Validation Loss Curve")
        plt.show()

    return model.cpu(), {"auc": auc_final, "aupr": aupr_final}


def test_model_cv(model, X_test, y_test):
    # Convert data to PyTorch tensors
    f_test_tensor = torch.tensor(X_test["edges"], dtype=torch.float32).view(X_test["edges"].shape[0], -1).to(device) if \
        X_test["edges"] is not None else torch.zeros((len(y_test), 1)).to(device)
    e1_test_tensor = torch.tensor(X_test["x1"], dtype=torch.float32).to(device) if X_test[
                                                                                       "x1"] is not None else torch.zeros(
        (len(y_test), 1)).to(device)
    e2_test_tensor = torch.tensor(X_test["x2"], dtype=torch.float32).to(device) if X_test[
                                                                                       "x2"] is not None else torch.zeros(
        (len(y_test), 1)).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Final evaluation
    model = model.to(device).eval()
    y_pred = model(f_test_tensor.to(device), e1_test_tensor.to(device),
                   e2_test_tensor.to(device)).cpu().detach().numpy().flatten()
    y_test_bin = y_test_tensor.numpy().flatten().astype(int)

    # Generate classification report and AUC
    auc = roc_auc_score(y_test_bin, y_pred)
    aupr = average_precision_score(y_test_bin, y_pred)  # Calculate AUPR

    print(f"AUC Score: {auc:.4f}")
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
            torch.save(data, encoded_genome_file)
    assert "inputs_embeds" in data and "group_embeds" in data, "Generated data missing required keys."

    # Run ProteomeLM model on the generated data
    checkpoint_path = Path(checkpoint)
    assert checkpoint_path.exists(), f"Checkpoint {checkpoint_path} does not exist."

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
    model = ProteomeLMForMaskedLM.from_pretrained(str(checkpoint_path))

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
