# Generated tables

| file | generator | status |
|---|---|---|
| `tab_sides_body.tex` | `FS/p5v3_tabB_sides_v1.py` | Table 4 (`tab:sides`) body, as input by the paper |
| `tab_gainedit_matched.tex` | `FS/p5v3_tabB_gain_edit_matched_v2.py` | Table 5 (`tab:gainedit_matched`); the released file is the version in the paper, i.e. the script output with row labels and caption edited by hand (numbers unchanged). Re-running the script overwrites those edits. |
| `tab_gainedit_full.tex` | `FS/p5v3_tabB_gain_edit_full_v1.py` | **Data package only** (`tab:gainedit_full`): full gain-edit table, 63 conditions x two evaluation slices; not included in the PDF. |

The generators write to `../app/` relative to their working layout; in this release the outputs are kept here.
