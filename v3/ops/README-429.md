# The 429 problem, and what actually fixes it

You asked for a proxy. I wired proxy support in (`PROXY_UID*` in `env.template`,
applied per-miner in `launch.sh`), but I want to be straight with you about what
it will and will not fix, because getting this wrong costs you the recency gate
and therefore 98% of the emission.

## Where the 429s come from

Two separate call sites, with different fixes.

### 1. Dataset streaming during Train — this is the big one

`connito/shared/dataloader.py:_load_streaming_split` calls
`load_dataset(..., streaming=True)`. Streaming means **every batch is an HTTP
range request to HF**, and streaming responses are *not* written to the local
datasets cache. Three miners training for 300 blocks each = three independent
request firehoses at `huggingface.co` for 60 minutes, every cycle.

This is almost certainly the bulk of your 429s, and a proxy is the wrong tool
for it. The fix is `ops/fetch_corpus.py`: pull the corpus **once** to
`$CORPUS_DIR` as local parquet, then point `dataset_class` at
`ops/shared_dataset.py:LocalSharedDataset`, which reads off local disk. After
that, all three miners make **zero** dataset requests during Train.

That single change removes roughly all steady-state HF traffic.

### 2. Checkpoint upload during MinerCommit2 — the narrow window

`connito/miner/model_io.py:_upload_checkpoint_to_hf_safe` runs inside a
**10-block (~2 minute)** phase. Three miners pushing multi-GB shards inside the
same 2-minute window is bursty by construction.

Here a proxy helps only partially, because **HF rate limits are enforced
primarily per token, and only secondarily per source IP.** If all three miners
authenticate with the same `HF_TOKEN`, routing them through three different
proxies changes the source IP but HF still sees one account making three
concurrent large writes. You will still get 429s.

The fix that actually works, in priority order:

1. **Three separate HF accounts, one token each.** This is why `env.template`
   has `HF_TOKEN_UID121/178/250` as three distinct fields rather than one shared
   value. This is also the correct model conceptually — each UID is a separate
   miner identity. Do this first.
2. **Stagger the uploads.** `launch.sh` starts the three miners 40 seconds
   apart, which offsets their phase-loop wall clocks so the uploads land at
   different points inside the MinerCommit2 window instead of all at block 0.
3. **Proxy, if you still need it.** With separate tokens this is mostly
   redundant, but it is wired and costs nothing to enable.

If a 429 does land, the consequence is specific and worth knowing:
`_upload_checkpoint_to_hf_safe` failing means the chain commit still goes out
**without HF coordinates**, so validators cannot find your checkpoint and you
are counted as missing for the round. Miss one round and you fall out of Weight
Group 1 for the whole cycle — that is the 98% share gone. Upload reliability is
worth more than any LR tuning.

## What I did not do

I did not build proxy rotation or retry-with-new-identity. If you want that,
say so and tell me what proxy pool you have, but I would push back: for this
workload it adds a failure mode (a dead proxy inside a 2-minute commit window
loses you the round) to solve a problem that three HF accounts solve outright.
