# Local ARC Explorer

This document explains the local visual browser for the official ARC-AGI-2 dataset.

## Run it

From this project folder:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000/dist/>.

The explorer reads the JSON files directly from `data/training` and `data/evaluation`. It does not send the dataset anywhere.

## Explorer controls

- Select Training or Public eval and search by task ID.
- Use `←` and `→` to move between tasks, or press `R` for a random task.
- Turn on Challenge mode to hide test outputs. It defaults on for Public eval and is saved locally in your browser.
- In Challenge mode, use the Your answer panel to choose rows/columns, paint cells with ARC colors, and submit a guess. Feedback is limited to exact-match success or another-attempt-needed; the official output remains hidden.
- Use Clear to reset a draft, or Reveal official answer only when you want to stop testing yourself.
- Use Focus view on any visible grid to open a larger, scrollbar-free view. Press `Esc` or Close to return.

The static site files live in [`../dist/`](../dist/), while the task files live in [`../data/`](../data/).
