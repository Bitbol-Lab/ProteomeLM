"""Driver for the essentiality experiment (replaces the original run_everything_*.sh).

    python -m experiments.essentiality.run_all STAGE [STAGE ...] [options]

Stages, in pipeline order:

* ``data``: download proteomes + OGEE labels, write labels and duplicate-free FASTAs.
* ``split``: mmseqs2 clustering at 40% identity + seeded 5-fold split (whole clusters per fold).
* ``train``: embed all genomes (ESM-C first, then each ProteomeLM), then train one
  classifier per layer for checkpoints ESMC, XS, S, M, L x seeds 42-47 x 1- and
  2-layer heads (the published grid).
* ``evaluate``: test-fold metric pickles for every trained run.
* ``figures``: Fig. 5A (2-layer; 1-layer variant too) and Fig. 5B donuts.
* ``baselines``: random / resampled ("statistics") ProteomeLM weights (seeds 42-45),
  2-layer classifiers on them, their metrics and the baseline figure.
* ``interpretability``: intrinsic dimension / entropy / PCA per layer and its figures.

Jobs are subprocesses of ``python -m experiments.essentiality.<module>``, logged to
``paths.logs_folder``; ``--gpus`` lists the GPU indices used round-robin and
``--max-jobs`` the number of concurrent jobs. ``--dry-run`` prints the commands.
"""
import argparse
import itertools
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Sequence

from experiments.essentiality.common import CHECKPOINTS, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGES = ["data", "split", "train", "evaluate", "figures", "baselines", "interpretability"]
PUBLISHED_SEEDS = [42, 43, 44, 45, 46, 47]
BASELINE_SEEDS = [42, 43, 44, 45]


class Job:
    def __init__(self, name: str, module: str, args: Sequence[str], gpu: Optional[int] = None):
        self.name, self.module, self.args, self.gpu = name, module, list(args), gpu

    def command(self) -> List[str]:
        cmd = [sys.executable, "-m", f"experiments.essentiality.{self.module}"] + self.args
        if self.gpu is not None:
            cmd += ["--gpu", str(self.gpu)]
        return cmd


class Runner:
    def __init__(self, gpus: Sequence[int], max_jobs: int, log_dir: str, dry_run: bool):
        self.gpus, self.max_jobs, self.log_dir, self.dry_run = list(gpus), max_jobs, log_dir, dry_run
        self._gpu_cycle = itertools.cycle(self.gpus)

    def next_gpu(self) -> int:
        return next(self._gpu_cycle)

    def _run_one(self, job: Job) -> int:
        cmd = job.command()
        log = os.path.join(self.log_dir, f"{job.name}.log")
        print(f"[{job.name}] {shlex.join(cmd)}  > {log}", flush=True)
        if self.dry_run:
            return 0
        os.makedirs(self.log_dir, exist_ok=True)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        with open(log, "w") as f:
            code = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        print(f"[{job.name}] {'done' if code == 0 else f'FAILED (exit {code}), see ' + log}", flush=True)
        return code

    def run(self, jobs: Sequence[Job], parallel: bool = True):
        """Run jobs (in parallel up to max_jobs); raise if any failed."""
        if not jobs:
            return
        workers = self.max_jobs if parallel else 1
        with ThreadPoolExecutor(max_workers=workers) as ex:
            codes = list(ex.map(self._run_one, jobs))
        failed = [j.name for j, c in zip(jobs, codes) if c != 0]
        if failed:
            raise RuntimeError(f"{len(failed)} job(s) failed: {failed}")


def common_args(args) -> List[str]:
    out = []
    for flag, value in [("--data-dir", args.data_dir), ("--config", args.config),
                        ("--checkpoint-dir", args.checkpoint_dir), ("--legacy-split-pkl", args.legacy_split_pkl),
                        ("--split-seed", args.split_seed), ("--classifier-dir", args.classifier_dir)]:
        if value is not None:
            out += [flag, str(value)]
    return out


def _data_args(args) -> List[str]:
    out = []
    for flag, value in [("--data-dir", args.data_dir), ("--config", args.config)]:
        if value is not None:
            out += [flag, str(value)]
    return out


def train_jobs(args, runner, checkpoints, seeds, layers, weights="trained") -> List[Job]:
    c = common_args(args) + (["--wandb"] if args.wandb else [])
    return [Job(f"train_{ck}_seed{s}_{nl}layers_weights_{weights}", "train",
                ["run", "--checkpoint", ck, "--seed", str(s), "--classifier-layers", str(nl), "--weights", weights] + c,
                runner.next_gpu())
            for s in seeds for ck in checkpoints for nl in layers if not (ck == "ESMC" and weights != "trained")]


def evaluate_jobs(args, runner, checkpoints, seeds, layers, weights="trained") -> List[Job]:
    c = common_args(args) + (["--plots-dir", args.plots_dir] if args.plots_dir else [])
    return [Job(f"metrics_{ck}_seed{s}_{nl}layers_weights_{weights}", "evaluate",
                ["--checkpoint", ck, "--seed", str(s), "--classifier-layers", str(nl), "--weights", weights] + c,
                runner.next_gpu())
            for s in seeds for ck in checkpoints for nl in layers if not (ck == "ESMC" and weights != "trained")]


def figure_job(args, name, figure, extra=(), gpu=None) -> Job:
    c = common_args(args)
    for flag, value in [("--plots-dir", args.plots_dir), ("--figures-dir", args.figures_dir)]:
        if value is not None:
            c += [flag, value]
    return Job(f"figure_{name}", "figures", [figure] + list(extra) + c, gpu)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("stages", nargs="+", choices=STAGES + ["all"])
    parser.add_argument("--data-dir", default=None, help="Root of all data paths (default DATA_ROOT/essentiality)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Local ProteomeLM weights ({dir}/ProteomeLM-{size}/checkpoint-210); default: Hugging Face")
    parser.add_argument("--legacy-split-pkl", default=None,
                        help="Train/evaluate on this fold pickle (e.g. all_sequences2_labelled_splits_40.pkl, "
                             "the published split) instead of the seeded one")
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--classifier-dir", default=None)
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--checkpoints", nargs="+", default=list(CHECKPOINTS), choices=CHECKPOINTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=PUBLISHED_SEEDS)
    parser.add_argument("--classifier-layers", nargs="+", type=int, default=[1, 2], choices=[1, 2, 3])
    parser.add_argument("--baseline-seeds", nargs="+", type=int, default=BASELINE_SEEDS)
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU indices (round-robin)")
    parser.add_argument("--max-jobs", type=int, default=2, help="Concurrent jobs")
    parser.add_argument("--mmseqs", default="mmseqs", help="mmseqs2 binary (split stage)")
    parser.add_argument("--threads", type=int, default=24, help="mmseqs threads")
    parser.add_argument("--wandb", action="store_true", help="Log training to Weights & Biases")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them")
    args = parser.parse_args(argv)

    stages = STAGES if "all" in args.stages else [s for s in STAGES if s in args.stages]
    cfg = load_config(args.config, args.data_dir)
    runner = Runner([int(g) for g in args.gpus.split(",")], args.max_jobs, cfg["paths"]["logs_folder"], args.dry_run)
    plm_checkpoints = [c for c in args.checkpoints if c != "ESMC"]

    if "data" in stages:
        runner.run([Job("data_download", "data", ["download"] + _data_args(args)),
                    Job("data_labels", "data", ["labels"] + _data_args(args))], parallel=False)
    if "split" in stages:
        extra = ["--mmseqs", args.mmseqs, "--threads", str(args.threads)]
        if args.split_seed is not None:
            extra += ["--split-seed", str(args.split_seed)]
        runner.run([Job("split", "data", ["split"] + extra + _data_args(args))])
    if "train" in stages:
        c = common_args(args)

        def embed(ck):
            return Job(f"embed_{ck}", "train", ["run", "--checkpoint", ck, "--embed-only"] + c, runner.next_gpu())
        # ESM-C first: the ProteomeLM runs reuse its per-genome ESM-C embeddings.
        if "ESMC" in args.checkpoints:
            runner.run([embed("ESMC")])
        runner.run([embed(ck) for ck in plm_checkpoints])
        runner.run(train_jobs(args, runner, args.checkpoints, args.seeds, args.classifier_layers))
    if "evaluate" in stages:
        runner.run(evaluate_jobs(args, runner, args.checkpoints, args.seeds, args.classifier_layers))
    if "figures" in stages:
        runner.run([figure_job(args, "fig5a_2layer", "fig5a", ["--classifier-layers", "2"]),
                    figure_job(args, "fig5a_1layer", "fig5a", ["--classifier-layers", "1"])])
        runner.run([figure_job(args, "fig5b", "fig5b", gpu=runner.next_gpu())])
    if "baselines" in stages:
        c = common_args(args)
        runner.run([Job("make_baselines", "train",
                        ["make-baselines", "--sizes", *plm_checkpoints, "--seeds", *map(str, args.baseline_seeds)] + c)])
        layers = [2]
        for weights in ("random", "statistics"):
            runner.run(train_jobs(args, runner, plm_checkpoints, args.baseline_seeds, layers, weights))
        for weights in ("random", "statistics"):
            runner.run(evaluate_jobs(args, runner, plm_checkpoints, args.baseline_seeds, layers, weights))
        runner.run([figure_job(args, "baselines", "baselines", ["--seeds", *map(str, args.baseline_seeds)])])
    if "interpretability" in stages:
        c = _data_args(args) + (["--checkpoint-dir", args.checkpoint_dir] if args.checkpoint_dir else [])
        runner.run([Job(f"interpretability_{ck}", "interpretability", ["--checkpoint", ck] + c, runner.next_gpu())
                    for ck in plm_checkpoints], parallel=False)
        runner.run([figure_job(args, "interpretability", "interpretability")])


if __name__ == "__main__":
    main()
