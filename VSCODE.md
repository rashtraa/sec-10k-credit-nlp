# Working in VS Code

## One-time setup

1. **Open the folder** — `File > Open Folder…` → `X:\Projects\edgar-credit-signal`.
   Open the *folder*, not individual files; the settings in `.vscode/` only
   apply when the folder is the workspace root.

2. **Install the recommended extensions** — VS Code will prompt. If it
   doesn't, open the Extensions panel and search `@recommended`. You need
   Python, Pylance, Jupyter, and debugpy.

3. **Select the interpreter** — `Ctrl+Shift+P` → *Python: Select
   Interpreter* → choose the one under
   `.\.venv\Scripts\python.exe`. It should be marked **Recommended**.

4. **Open a new terminal** (`Ctrl+` `` ` ``) and verify:

   ```powershell
   python env_check.py
   ```

   Must end with `PASS — nothing writes to C:`.

That last step is the one that matters. Read on for why.

---

## Why `env_check.py` is not optional here

VS Code activates your virtual environment automatically, which is
convenient and also a trap. It activates the venv — it does **not** run
`activate.ps1`. So the interpreter is right, everything imports, and none of
the cache redirects exist.

You would not notice. Stage 3 downloads an embedding model, HuggingFace
writes ~1 GB to `C:\Users\Rashtra Mitra\.cache\huggingface`, and your C:
drive — which had 10.1 GB free — quietly absorbs it.

`.vscode/settings.json` fixes this by setting the variables on every
integrated terminal, and `.env` covers the Run button and debugger, which
don't go through a terminal at all. Both are in the repo. `env_check.py` is
how you confirm they took effect.

**If you ever open a terminal that VS Code didn't create** (an external
PowerShell, say), run `.\activate.ps1` in it as before.

---

## Running things

Three ways, all equivalent:

| | How |
|---|---|
| **Terminal** | `python extract_sections.py` |
| **Run button** | Open the file, click ▷ top-right |
| **Debugger** | `F5`, then pick a stage from the dropdown |

`.vscode/launch.json` lists every stage in run order, so `F5` gives you a
menu rather than requiring you to remember filenames:

```
0 · status (where am I?)
0 · env_check (nothing writes to C:?)
1a · build_universe
1b · download_filings
2a · extract_sections
2b · text_features
tests · parsing + sections
```

`justMyCode` is off for the analysis stages, so you can step into pandas and
sklearn when a number looks wrong instead of guessing at it.

---

## Two settings that keep the editor usable

`data/` holds 8,800 gzipped filings. Left alone, VS Code indexes them, and
`Ctrl+Shift+F` becomes useless while the file watcher chews through a
gigabyte. `settings.json` excludes `data/`, `.cache/` and `.venv/` from
search and from watching. The folder is still there and still readable —
just not indexed.

`.gitignore` excludes the same directories. **Do not commit `data/`** — it is
a gigabyte, and every byte of it is regenerable from EDGAR by rerunning two
scripts.

---

## Where I'd like to take this next

Your PD Engine repo is a numbered notebook sequence, and it reads well —
`01_setup_and_data.ipynb` through `08_stress_and_sensitivity.ipynb`. Now that
you're in VS Code with the Jupyter extension, stages 3 and 4 can match that
shape:

```
03_retrieval_benchmark.ipynb    BM25 vs dense vs hybrid, Recall@k and MRR
04_incremental_signal.ipynb     ratios vs ratios+text, permutation null
```

Notebooks suit these two stages because the output *is* the argument — tables
and charts you'll paste into the writeup. The ingest and feature stages stay
as scripts, because they are long-running and you want them resumable rather
than tied to a kernel.

That also makes the repo consistent with the two projects already on your
CV, which is worth something when someone opens all three.
