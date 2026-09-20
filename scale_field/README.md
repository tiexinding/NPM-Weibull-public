# A Mesoscopic View of Transformer Weights Through Row and Column Scale Fields

Companion data and code for the paper:

> **A Mesoscopic View of Transformer Weights Through Row and Column Scale Fields**
> Tiexin Ding (Independent Researcher). arXiv: to be added at posting.

This is the Paper #5 companion in the unified `NPM-Weibull-public` repository (Paper #1 framework
`arXiv:2605.18898`, Paper #2 training dynamics `arXiv:2606.19367`, Paper #3 data predictability
`arXiv:2608.23573`; see the top-level README).

## What is released

Every figure, table and load-bearing number in the paper traces through three layers:
raw runs and checkpoints → read-out JSON (extraction script) → figure or table (plotting script).
Layers 2 and 3 are released here; layer 1 (training checkpoints, optimizer dumps, pre-tokenized
token streams) is archived by the author and available on request.

```
scale_field/
├── DATA/                 40 read-out JSON files + DATA/external/ (4 read-outs shared with earlier studies)
├── scripts/              23 extraction scripts: checkpoints / optimizer dumps → read-out JSON
├── FS/                   14 figure and table scripts: read-out JSON → the 14 figures and 2 generated tables
├── DATA_PROVENANCE.md    map from every figure, table and quoted number to its read-out, script and raw source
└── MANIFEST.txt          MD5 and size of every released file
```

`DATA_PROVENANCE.md` is the index: §2 lists each read-out with its extraction script and raw input, §3 each
figure and table with its plotting script and inputs, §4 the release tiers, §5–6 the details and removed
numbers referenced from the appendices.

## Reproducing figures and tables

The figure scripts read only the JSON files under `DATA/` and assert the numbers quoted in the captions.
They were written against the author's working layout (`.../08_paper5_draft/data/` for the read-outs and
`.../p5_compile/figure_v2/figures/` for the output); to run them here, point the `DRAFT`/`ROOT` path
constants at this directory's `DATA/` and choose an output folder. The extraction scripts additionally
need the raw checkpoints or optimizer dumps named in `DATA_PROVENANCE.md` §1 and are included for
inspection of the exact read-out definitions. Absolute working paths recorded inside three read-outs and one
script were replaced by the placeholders `<selfavg_mechanism>` and `<hf_cache>`.

## Verify

```bash
cd scale_field && awk '!/^#/{print $1"  "$3}' MANIFEST.txt | md5sum -c --quiet && echo OK
```

## License

Same as the repository (see top-level `LICENSE`).
