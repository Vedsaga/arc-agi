# ARC-AGI-2 Workbench

This repository combines two related pieces of ARC-AGI-2 work:

1. A local visual explorer for browsing tasks and practicing hidden-answer solving.
2. Research experiments and Kaggle notebooks for symbolic ARC solving.

## Start the local explorer

The explorer is a dependency-free static site. From the repository root:

```bash
python3 -m http.server 8000
```

Open <http://localhost:8000/dist/>.

The explorer reads the local JSON files from `data/training` and `data/evaluation`; it does not upload the dataset anywhere.

### Explorer workflow

- Choose **Training** or **Public eval**, then search by task ID.
- Use `←` and `→` to move between tasks, or press `R` for a random task.
- Turn on **Challenge mode** to hide test outputs.
- Build a prediction in the **Your answer** editor by resizing the grid and painting cells.
- Use **Check guess** for private exact-match feedback without revealing the official output.
- Use **Focus view** for a larger view of any visible grid.

## Repository map

| Path | Purpose |
| --- | --- |
| [`data/`](data/) | Local ARC-AGI-2 task JSON files: 1,000 training tasks and 120 public evaluation tasks. |
| [`dist/`](dist/) | The local static explorer (`index.html`, `styles.css`, `app.js`). |
| [`docs/`](docs/) | Dataset notes and local explorer instructions. |
| [`research/experiments/`](research/experiments/) | Python research scripts for symbolic and behavioral search. |
| [`research/notebooks/`](research/notebooks/) | Kaggle/Jupyter notebooks for ARC-AGI-2 experiments. |

## Research track

The research work explores whether search over task-distinguishable behavior classes can reduce the computation required to reach correct ARC solutions compared with searching over syntactically distinct programs.

Current directions include:

- Semantic quotient search over observationally equivalent hypotheses.
- Backward edit-graph induction and structural anti-unification.
- Object/relation primitives, non-uniform priors, and information-gain scheduling.
- Exact symbolic verification with ARC pass@2 diagnostics.

The main semantic quotient notebook is [`research/notebooks/semantic_quotient_search_arc_agi2_v0.ipynb`](research/notebooks/semantic_quotient_search_arc_agi2_v0.ipynb). It produces `submission.json` and diagnostic CSVs when run in Kaggle.

## Documentation

- [ARC-AGI-2 dataset and task format](docs/arc-agi-2-dataset.md)
- [Local explorer instructions](docs/local-explorer.md)

## Data and competition note

The public training and evaluation files are included for local exploration and research. Kaggle’s private leaderboard tasks are not included in this repository; their scores can only be obtained through the competition evaluator.

The dataset and upstream materials are distributed under the included [Apache-2.0 license](LICENSE).
