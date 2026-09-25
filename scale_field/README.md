# A Mesoscopic View of Transformer Weights Through Row and Column Scale Fields

Companion data and code for the paper:

> **A Mesoscopic View of Transformer Weights Through Row and Column Scale Fields**
> Tiexin Ding. arXiv: to be added at posting.

This is the Paper #5 companion in the unified `NPM-Weibull-public` repository (Paper #1 framework
`arXiv:2605.18898`, Paper #2 training dynamics `arXiv:2606.19367`, Paper #3 data predictability
`arXiv:2608.23573`; see the top-level README).

## What is released

Every figure, table and load-bearing number in the paper traces through three layers:
raw runs and checkpoints → read-out JSON (extraction script) → figure or table (plotting script).

Released here:

- **Read-outs**: every read-out JSON behind a figure, table or quoted number (`DATA/`).
- **Extraction scripts**, checkpoints or optimizer dumps → read-out JSON (`scripts/`).
- **Figure and table scripts**, read-out JSON → the 14 figures and the generated tables (`FS/`, tables in `FS/tables/`).
- **Figures** as they appear in the paper, plus the Figure 1 SVG source (`FIG/`).
- **Training code, model configurations and run records** for every evidence stream (`train/`, one run record
  per run in `train/run_configs/`), with the script version each run recorded and CPU consistency checks of the
  released trainers against the stored records (`train/README.md`).
- **The pretokenization script**: the 1e8-token training stream is rebuilt from WikiText-103 with
  `train/ea_pretok.py` (expected SHA-256 in `train/ea_tokens.npy.meta.json`). The Pythia-series corpora and runs use
  the scripts in `train/pythia_series/`.

Not released: model checkpoints and AdamW moment dumps, because of their size. The extraction scripts therefore
cannot be re-run from this directory alone and are included so the exact read-out definitions can be inspected.
Every figure and table reproduces from the released read-outs.

```
scale_field/
├── DATA/                 39 read-out JSON files; DATA/external/ (11 read-outs that lived outside the draft
│                         folder, original names or <run>_analysis_v2.json); DATA/pythia_series/ (3 per-matrix
│                         read-outs of the Pythia-protocol size series)
├── scripts/              52 extraction scripts: checkpoints / optimizer dumps → read-out JSON
├── FS/                   14 figure and table scripts; FS/tables/ holds the 3 generated LaTeX tables
├── FIG/                  the 14 figures of the paper (PNG) and the Figure 1 SVG source
├── train/                trainers, model configurations, ea_pretok.py, Pythia-series scripts, run records (train/README.md)
├── DATA_PROVENANCE.md    map from every figure, table and quoted number to its read-out, script and raw source
└── MANIFEST.txt          MD5 and size of every released file
```

`DATA_PROVENANCE.md` is the index: §1 lists the raw sources, §2 each read-out with its extraction script and raw
input, §3 each figure and table with its plotting script and inputs, §4 the release tiers, §5–6 the details and
removed numbers referenced from the appendices. `train/README.md` gives one line per evidence stream: script and
recorded hash, flags, corpus and run records.

## Reproducing figures and tables

The figure scripts read only the JSON files under `DATA/` and assert the numbers quoted in the captions. They were
written against the author's working layout (`.../08_paper5_draft/data/` for the read-outs,
`.../p5_compile/figure_v2/figures/` for the output, `../app/` for the generated tables); to run them here, point the
`DRAFT`/`ROOT` path constants at this directory's `DATA/` (and `DATA/external/`, `DATA/pythia_series/` for the
Figure 5 script) and choose an output folder. Absolute working paths recorded inside read-outs and scripts were
replaced by the placeholders `<selfavg_mechanism>`, `<hf_cache>` and `<repo>`.

## Verify

```bash
cd scale_field && awk '!/^#/{print $1"  "$3}' MANIFEST.txt | md5sum -c --quiet && echo OK
```

## License

Same as the repository (see top-level `LICENSE`).
