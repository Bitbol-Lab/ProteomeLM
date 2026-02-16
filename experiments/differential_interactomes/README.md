# ProteomeLM Minimal Analysis Pipeline

Simplified pipeline for ProteomeLM benchmark evaluation addressing reviewer comments.

## Code Reduction Summary

| File | Original | Minimal | Reduction |
|------|----------|---------|-----------|
| `build_benchmark` | 1,196 lines | 275 lines | **77%** |
| `extract_attention` | 282 lines | 241 lines | **14%** |
| `analyze_benchmark` | 3,300+ lines (notebook) | 704 lines | **79%** |

## Output Tables

### Table S1: AUROC vs Random
Discriminating interaction types from random pairs using attention heads.

| Species | Direct (PDB) | Same complex (PDB) | Coexpression (STRING) |
|---------|--------------|--------------------|-----------------------|
| E. coli | | | |
| S. cerevisiae | | | |
| H. sapiens | | | |

### Table S2: Pairwise Classification
Binary classification accuracy between interaction types using logistic regression on attention heads.

| Species | Direct vs Same complex | Direct vs Coexpression | Direct vs Random |
|---------|------------------------|------------------------|------------------|
| E. coli | | | |
| S. cerevisiae | | | |
| H. sapiens | | | |

## Output Figures (PDF + SVG)

For each species:
1. `{species}_attention_by_type` - AUROC per attention head
2. `{species}_heads_vs_cosine_vs_pca` - Attention vs Cosine similarity vs PCA removal
3. `{species}_pairwise_classification_heads` - Classifier coefficient heatmaps
4. `summary_figure` - Combined visualization

## Usage

### Quick Start: Run Full Pipeline
```bash
./run_full_pipeline.sh
```

This runs all three steps for all species automatically.

### Manual Step-by-Step

#### Step 1: Build Benchmark
```bash
# For each species
python build_benchmark_minimal.py --species yeast --output-dir data/benchmarks
python build_benchmark_minimal.py --species human --output-dir data/benchmarks
python build_benchmark_minimal.py --species ecoli --output-dir data/benchmarks
```

#### Step 2: Extract Attention
```bash
# For each species (requires ProteomeLM checkpoint)
python extract_attention_minimal.py \
    --species yeast \
    --checkpoint Bitbol-Lab/ProteomeLM-M
```

#### Step 3: Run Analysis
```bash
python analyze_proteomelm_minimal.py \
    --attention-dir attention_patterns
```

## Interaction Types

| Type | Source | Description |
|------|--------|-------------|
| `pdb` | PINDER/PDB | Direct structural contacts (gold standard) |
| `pdb_physical` | PINDER/PDB | Same complex, lower quality |
| `coexpression` | STRING | Expression correlation |
| `random` | Generated | Negative control |

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
torch
requests
```

For PINDER: `pip install datasets` (HuggingFace)
