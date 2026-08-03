"""
Local-disk replacement for SN102's HF streaming training dataset.

Plugged in via the expert group config's documented extension point:

    data:
      dataset_class: "ops.shared_dataset:LocalSharedDataset"

`connito/shared/dataloader.py:get_dataloader` resolves that string and calls
`DatasetCls.get_tokenised_dataset(...)`, which is the whole dataset-construction
path. Overriding it means we never touch `load_dataset(streaming=True)`, so a
running miner makes **zero** HF dataset requests during Train. Three miners on
one box therefore stop competing for the same rate-limit bucket.

Data comes from `ops/fetch_corpus.py` output at
`$CORPUS_DIR/<expert_group>/`.

Distribution fidelity
---------------------
Proof-of-Loss scores us on `val_loss` over the validator's eval mixture, so the
training stream has to match it. Two things are preserved deliberately:

  * **Source weights.** Sources are interleaved at the weights declared in the
    group's config.yaml (c4 0.5 / Nemotron-Math 0.5 for exp_math), matching
    `interleave_datasets` upstream.
  * **Tokenisation.** Identical to
    `expert_groups/exp_math/dataset.py:StreamingTorchDataset.tokenize_and_format`
    — raw-text CPT, no chat template, truncation + pad to `sequence_length`,
    `add_special_tokens=True`.

Per-miner decorrelation
-----------------------
`rank`/`world_size` select a disjoint row stripe, and `CONNITO_DATA_SEED`
permutes shard order. Running our three UIDs with different values means they
train on different data in different orders. That is not cosmetic: the
finalizer zeroes **both** miners when two submissions produce bit-identical
`val_loss` (the duplicate-submission heuristic in
`connito/validator/evaluator.py:finalize_round_scores`). Decorrelated streams
make that collision effectively impossible between our own UIDs.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterator

import torch
from transformers import PreTrainedTokenizerBase

from connito.shared.dataloader import DefaultStreamingTorchDataset


def _corpus_dir() -> Path:
    return Path(os.environ.get("CORPUS_DIR", "/root/SN102/data/corpus"))


class _ParquetRowStream:
    """
    Endless, weighted, sharded row iterator over the local parquet corpus.

    Reads one row group at a time so memory stays flat regardless of corpus
    size. Loops forever: the Train phase is bounded by chain blocks, not by
    epochs, so the stream must never raise StopIteration mid-phase.
    """

    def __init__(
        self,
        group_dir: Path,
        rank: int,
        world_size: int,
        seed: int,
    ) -> None:
        manifest_path = group_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"no corpus manifest at {manifest_path}. "
                f"Run: python ops/fetch_corpus.py --expert-group {group_dir.name}"
            )
        manifest = json.loads(manifest_path.read_text())

        self.rank = max(0, int(rank))
        self.world_size = max(1, int(world_size))
        self.seed = int(seed)

        self.sources: list[dict[str, Any]] = []
        for entry in manifest["sources"]:
            sub = group_dir / (
                entry["path"] + ("__" + entry["name"] if entry.get("name") else "")
            ).replace("/", "__")
            shards = sorted(sub.glob("part-*.parquet"))
            if not shards:
                raise FileNotFoundError(f"corpus source {sub} has no parquet shards")
            self.sources.append(
                {"name": entry["path"], "weight": float(entry.get("weight", 1.0)), "shards": shards}
            )

        total = sum(s["weight"] for s in self.sources)
        if total <= 0:
            raise ValueError("dataset source weights sum to zero")
        for s in self.sources:
            s["weight"] /= total

    def _source_rows(self, source: dict[str, Any], rng: random.Random) -> Iterator[str]:
        """Endless shuffled row iterator for one source, striped by rank."""
        import pyarrow.parquet as pq

        shards = list(source["shards"])
        while True:
            rng.shuffle(shards)
            for shard in shards:
                pf = pq.ParquetFile(shard)
                for rg in range(pf.num_row_groups):
                    col = pf.read_row_group(rg, columns=["text"]).column("text")
                    # Stripe by rank so co-located miners never see the same rows.
                    for i in range(self.rank, len(col), self.world_size):
                        value = col[i].as_py()
                        if value:
                            yield value

    def __iter__(self) -> Iterator[str]:
        rng = random.Random(self.seed)
        iters = [self._source_rows(s, random.Random(self.seed + i)) for i, s in enumerate(self.sources)]
        weights = [s["weight"] for s in self.sources]
        while True:
            # Weighted source choice, mirroring interleave_datasets' probabilities.
            yield next(iters[rng.choices(range(len(iters)), weights=weights, k=1)[0]])


class _TokenisingIterable(torch.utils.data.IterableDataset):
    """Wraps the row stream and emits model-ready tensors."""

    def __init__(
        self,
        rows: _ParquetRowStream,
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int,
        tokenize_fn,
    ) -> None:
        super().__init__()
        self.rows = rows
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.tokenize_fn = tokenize_fn

    def __iter__(self) -> Iterator[dict[str, Any]]:
        info = torch.utils.data.get_worker_info()
        stride = info.num_workers if info else 1
        offset = info.id if info else 0
        for i, text in enumerate(self.rows):
            # Second-level striping across DataLoader workers so num_workers>1
            # does not duplicate rows.
            if stride > 1 and (i % stride) != offset:
                continue
            yield self.tokenize_fn({"text": text}, self.tokenizer, self.sequence_length)


class LocalSharedDataset(DefaultStreamingTorchDataset):
    """Drop-in `dataset_class` that reads the shared local corpus."""

    @staticmethod
    def tokenize_and_format(
        example: dict[str, Any],
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int,
    ) -> dict[str, Any]:
        # Delegate to upstream `tokenize_windowed` -- the same function
        # `DefaultStreamingTorchDataset` uses. Do not reimplement it here: any
        # divergence from how the validator tokenises its eval slice shows up
        # as a systematically worse val_loss, and a local copy would drift the
        # first time upstream touched it.
        #
        # This replaces the old prefix-truncating path copied from
        # `expert_groups/exp_math/dataset.py`. That path always trained on a
        # document's FIRST sequence_length tokens, which for templated corpora
        # is the most boilerplate-heavy region -- document bodies never entered
        # the pipeline. exp_nemotron_c4 deliberately leaves `dataset_class`
        # unset upstream to get the windowed behaviour, so a local dataset that
        # prefix-truncates would silently give up the anti-memorization
        # property in exchange for avoiding HF streaming.
        from connito.shared.dataloader import tokenize_windowed

        return tokenize_windowed(
            str(example.get("text", "")), tokenizer, sequence_length
        )

    @classmethod
    def get_tokenised_dataset(
        cls,
        config,
        tokenizer: PreTrainedTokenizerBase,
        rank: int | None = None,
        world_size: int | None = None,
        train: bool = True,
        seed: str | int | None = None,
        fraction: float | None = None,
        **_ignored: Any,
    ):
        data_cfg = config.task.exp.data
        group_name = getattr(config.task, "expert_group_name", None) or "exp_math"
        group_dir = _corpus_dir() / group_name

        # Stripe selection comes from the ENVIRONMENT, not the config.
        #
        # `WorkerConfig._update_by_task` rebuilds `task.exp` wholesale from
        # expert_groups/<group>/config.yaml on every load, so per-UID `rank` /
        # `world_size` written into a miner YAML are discarded (they come back
        # as the group default rank=1/world_size=10). And the group config is
        # shared by all three miners, so it cannot carry per-UID values either.
        #
        # An env var is per-process and survives config reload, so that is the
        # only place this can live. launch.sh sets CONNITO_DATA_RANK /
        # CONNITO_DATA_WORLD per miner. Falling back to the config values keeps
        # a bare `python -m connito.miner.train` run working.
        env_rank = os.environ.get("CONNITO_DATA_RANK")
        env_world = os.environ.get("CONNITO_DATA_WORLD")
        if env_rank is not None and env_world is not None:
            rank, world_size = int(env_rank), int(env_world)
        else:
            rank = data_cfg.rank if rank is None else rank
            world_size = data_cfg.world_size if world_size is None else world_size

        # Miners pass seed=None (only the validator eval path seeds); use the
        # per-miner env seed so our UIDs decorrelate from each other.
        if seed is None:
            seed = int(os.environ.get("CONNITO_DATA_SEED", "0"))
        elif isinstance(seed, str):
            seed = int(str(seed)[:8], 16)

        rows = _ParquetRowStream(
            group_dir=group_dir,
            rank=int(rank),
            world_size=int(world_size),
            seed=int(seed),
        )
        return _TokenisingIterable(
            rows=rows,
            tokenizer=tokenizer,
            sequence_length=int(data_cfg.sequence_length),
            tokenize_fn=cls.tokenize_and_format,
        )
