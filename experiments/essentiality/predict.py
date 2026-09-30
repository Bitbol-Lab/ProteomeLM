"""Predict essential genes of a whole proteome with ProteomeLM-ess.

    python -m experiments.essentiality.predict --fasta proteome.fasta --out scores.tsv --top-n 300

Same command line as ``python -m proteomelm.essentiality`` (see ``--help``), and
``--head`` also accepts a classifier folder written by ``train.py``
(``config.json`` + ``pytorch_model.bin``), e.g. one of the per-layer classifiers
of a training run. Output TSV columns: ``protein_id``, ``p_essential`` (softmax
probability of the essential class), ``rank`` (1 = most essential) and, with
``--top-n``/``--top-fraction``/``--threshold``,
``predicted_essential``.
"""
from proteomelm.essentiality import build_arg_parser, load_head, run_cli


def load_any_head(source, device="cpu", revision=None):
    """A ProteomeLM-ess head folder / Hugging Face repo, or a training-pipeline classifier folder."""
    from experiments.essentiality.package_head import is_classifier_checkpoint, load_classifier_checkpoint
    from proteomelm.essentiality import EssentialityHead
    import torch

    if is_classifier_checkpoint(source):
        config, state_dict = load_classifier_checkpoint(source)
        head = EssentialityHead(config)
        head.load_state_dict(state_dict)
        return config, head.to(device=device, dtype=torch.float32).eval()
    return load_head(source, device=device, revision=revision)


def main(argv=None):
    parser = build_arg_parser("python -m experiments.essentiality.predict")
    parser.description += (" --head also accepts a classifier folder written by train.py "
                           "(config.json + pytorch_model.bin).")
    return run_cli(parser.parse_args(argv), load_head_fn=load_any_head)


if __name__ == "__main__":
    main()
