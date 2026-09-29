"""Train the bundled supervised PPI models `data/interactomes/enhanced_ppi_model_<model>.pt`.

One recipe for all: ProteomeLM-S `ProteomeLM+Att` features, `EnhancedPPIModel` trained with the
original ProteomeLM 1.0 loop (Adam, lr 5e-4, plain BCE, batch 1024); the weights of the later epochs are
averaged (SWA) and BatchNorm statistics recomputed on the training pairs. Only the training pairs, the number
of epochs and, for `ecoli_yeast`, the batch size differ:

- `dscript`: D-SCRIPT human train split; 10 epochs, SWA over epochs 3-10. Training longer specializes the model
  to human and costs accuracy on distant proteomes (E. coli, yeast).
- `multispecies`: the same, plus the D-SCRIPT E. coli and yeast pairs (D-SCRIPT's test sets for these species,
  so they are training data here). Better than `dscript` on bacteria and at least as good on animals.
- `ecoli_yeast` (comparison only, not bundled): the E. coli and yeast pairs only; batch 256, 10 epochs, SWA over
  epochs 3-10. Slightly better than `multispecies` on bacteria, much worse on animals.
- `bernett`: Bernett et al. (2024) gold standard as split in the paper (train = Intra-1, validation = Intra-0,
  test = Intra-2); the number of epochs and the SWA window were chosen on validation.

    python experiments/ppi_bundled_models/train_bundled.py --model dscript --out /path/to/workdir [--device cuda:0]

Features are cached in `<out>/features/<dataset>.pt`. Extraction runs ProteomeLM on the CPU over each
proteome's full FASTA (all layers' N x N attention: ~135 GB RAM for mouse), ESM-C on --device.
"""
import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

from proteomelm.ppi.config import DATA_ROOT
from proteomelm.ppi.data_processing import create_extractor
from proteomelm.ppi.feature_extraction import PPIFeatureExtractor
from proteomelm.ppi.model import (
    EnhancedPPIModel, _seed_everything, average_state_dicts, prepare_ppi, recompute_batchnorm_stats, test_model_cv,
)

# dataset -> (directory, FASTA file, interaction extractor)
DATASETS = {
    **{f"dscript_{sp}": (DATA_ROOT / "dscript" / sp, f"{sp}.faa", "dscript")
       for sp in ["human", "mouse", "fly", "worm", "yeast", "ecoli"]},
    "bernett": (DATA_ROOT / "bernett", "human_gold.faa", "bernett"),
}
DSCRIPT_EVAL = {"human_val": ("dscript_human", "val"), **{sp: (f"dscript_{sp}", "test")
                                                         for sp in ["mouse", "fly", "worm", "yeast", "ecoli"]}}
RECIPE = {"optimizer": "Adam", "lr": 5e-4, "loss": "BCE", "batch_size": 1024, "seed": 0,
          "batchnorm": "recomputed on the training pairs after averaging"}
MODELS = {
    "dscript": {"train": [("dscript_human", "train")], "epochs": 10, "swa_epochs": [3, 10],
                "evaluate": DSCRIPT_EVAL},
    "multispecies": {"train": [("dscript_human", "train"), ("dscript_ecoli", "test"), ("dscript_yeast", "test")],
                     "epochs": 10, "swa_epochs": [3, 10],
                     "evaluate": {k: v for k, v in DSCRIPT_EVAL.items() if k not in ("ecoli", "yeast")}},
    "ecoli_yeast": {"train": [("dscript_ecoli", "test"), ("dscript_yeast", "test")], "batch_size": 256,
                    "epochs": 10, "swa_epochs": [3, 10],
                    "evaluate": {k: v for k, v in DSCRIPT_EVAL.items() if k not in ("ecoli", "yeast")}},
    "bernett": {"train": [("bernett", "train")], "epochs": 40, "swa_epochs": [20, 40],
                "evaluate": {"val": ("bernett", "val"), "test": ("bernett", "test")}},
}
logger = logging.getLogger("train_bundled")


def extract(dataset: str, backbone: str, out: Path, device: str) -> dict:
    """Pair features (a_ij + a_ji per layer/head) and ProteomeLM logits of every split, cached on disk."""
    path = out / "features" / f"{dataset}.pt"
    if not path.exists():
        env, fasta, extractor = DATASETS[dataset]
        output = prepare_ppi(backbone, str(env / fasta), encoded_genome_file=str(out / "esm" / f"{dataset}.pt"),
                             esm_device=device, proteomelm_device="cpu", include_attention=True,
                             include_all_hidden_states=False, reload_if_possible=True)
        index_dict, y_dict = create_extractor(extractor).extract(env, fasta)
        pairs = [p for split in index_dict.values() for p in split]
        attention = PPIFeatureExtractor._process_attention(output["plm_attentions"], pairs)  # a_ij + a_ji
        logits, splits, start = output["plm_logits"][0], {}, 0
        for split, split_pairs in index_dict.items():
            i0 = torch.tensor([p[0] for p in split_pairs])
            i1 = torch.tensor([p[1] for p in split_pairs])
            splits[split] = {"edges": attention[start:start + len(split_pairs)].float(),
                             "x1": logits[i0].float(), "x2": logits[i1].float(), "y": y_dict[split]}
            start += len(split_pairs)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"backbone": backbone, "n_proteins": logits.shape[0], "splits": splits}, path)
    return torch.load(path, weights_only=False)


def as_arrays(split: dict):
    return ({"edges": split["edges"].reshape(len(split["y"]), -1).numpy(), "x1": split["x1"].numpy(),
             "x2": split["x2"].numpy()}, split["y"].numpy())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backbone", default="Bitbol-Lab/ProteomeLM-S")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(args.device)
    spec = MODELS[args.model]
    recipe = {**RECIPE, **{k: spec[k] for k in ("batch_size", "epochs", "swa_epochs") if k in spec}}

    datasets = {ds for ds, _ in spec["train"]} | {ds for ds, _ in spec["evaluate"].values()}
    features = {ds: extract(ds, args.backbone, args.out, args.device) for ds in sorted(datasets)}
    parts = [features[ds]["splits"][split] for ds, split in spec["train"]]
    tensors = [torch.cat([p["edges"].reshape(len(p["y"]), -1) for p in parts]),
               torch.cat([p["x1"] for p in parts]), torch.cat([p["x2"] for p in parts]),
               torch.cat([p["y"] for p in parts]).float().unsqueeze(1)]
    tensors = [t.to(device) for t in tensors]
    n = len(tensors[3])

    _seed_everything(42 + recipe["seed"])
    model = EnhancedPPIModel(protein_embed_dim=tensors[1].shape[1], pair_feature_dim=tensors[0].shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=recipe["lr"])
    loss_fn = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(42 + recipe["seed"])
    bs, (swa_start, swa_end) = recipe["batch_size"], recipe["swa_epochs"]
    swa_states = []
    for epoch in range(1, recipe["epochs"] + 1):
        model.train()
        perm = torch.randperm(n, generator=generator).to(device)
        stop = n - 1 if n % bs == 1 else n  # BatchNorm cannot train on a single sample
        for i in range(0, stop, bs):
            f, e1, e2, y = (t[perm[i:i + bs]] for t in tensors)
            optimizer.zero_grad()
            loss_fn(model(f, e1, e2), y).backward()
            optimizer.step()
        if swa_start <= epoch <= swa_end:
            swa_states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        logger.info("epoch %d done", epoch)

    model.load_state_dict(average_state_dicts(swa_states))
    recompute_batchnorm_stats(model, ((tensors[0][i:i + 4096], tensors[1][i:i + 4096], tensors[2][i:i + 4096])
                                      for i in range(0, n, 4096)))
    model.eval()

    results = {}
    for name, (ds, split) in spec["evaluate"].items():
        metrics = test_model_cv(model, *as_arrays(features[ds]["splits"][split]), device=device)[3]
        results[name] = {"auc": float(metrics["auc"]), "aupr": float(metrics["aupr"])}
        logger.info("%s: AUC %.3f  AUPR %.3f", name, results[name]["auc"], results[name]["aupr"])

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "feature_combination": "ProteomeLM+Att",
        "model_architecture": {"protein_embed_dim": model.protein_embed_dim, "pair_feature_dim": model.pair_feature_dim},
        "training_data": [f"{ds}:{split}" for ds, split in spec["train"]],
        "training_recipe": recipe,
        "evaluation": {name: {"data": f"{ds}:{split}", **results[name]}
                       for name, (ds, split) in spec["evaluate"].items()},
        "backbone": args.backbone,
        "pair_features": "a_ij + a_ji per layer/head, layer-major (PPIFeatureExtractor._process_attention)",
        "protein_features": "ProteomeLM logits (plm_logits)",
    }, args.out / f"enhanced_ppi_model_{args.model}.pt")
    (args.out / f"results_{args.model}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
