# Publishing on Kaggle

| File | Purpose |
|---|---|
| `kaggriculture-planning-agent.ipynb` | The notebook, generated from `farm_agent/` by `build_notebook.py` |
| `kernel-metadata.json` | Settings for `kaggle kernels push`: competition attached, Internet on, private |
| `discussion_post.md` | Companion post for the competition's Discussion tab; links to the notebook and GitHub |
| `build_notebook.py` | Regenerates the notebook after any code change |

## Notebook

**Kaggle UI.** Competition page → Code → New Notebook → File → Import Notebook → choose the `.ipynb`. In the side panel, turn **Internet on** and confirm the Kaggriculture competition is attached. Then Save Version → **Save & Run All** (about five minutes on CPU) and set visibility to Public.

**Or the CLI**, from `kaggriculture/`:

```bash
python kaggle_notebook/build_notebook.py   # only if farm_agent/ changed
kaggle kernels push -p kaggle_notebook     # uploads and runs it, private
```

Then open `https://www.kaggle.com/code/wangshu/kaggriculture-planning-agent-and-evaluation`, check the outputs, and switch it to Public. To publish directly instead, set `"is_private": false` before pushing.

## Discussion post

Competition page → Discussion → New Topic. Paste `discussion_post.md`; the first line is the title. Post it after the notebook is public, so its link resolves. Optionally attach `../report/figures/fig4_measurement_resolution.png` and `fig6_flippable_losses.png`.
