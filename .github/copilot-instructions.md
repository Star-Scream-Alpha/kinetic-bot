**Repository Purpose**
- **Summary:**: This repo contains exploratory trading analysis and backtest notebooks that simulate option gamma-scalping and related P&L calculations (example: `delta-hedhged-gamma-scalp.ipynb`).

**Big Picture / Architecture**
- **Primary artefacts:**: notebooks in root (not a packaged app). Notebooks load local market data (CSV/parquet) and write CSV summaries.
- **Data flow:**: notebooks expect raw ticks (futures and options) and a trades windows file. Example: `FUT_FILE = "NIFTY20NOV.csv"`, `TRADES_FILE = "kinetic_full_backtest_trades.csv"`, option files named `"{OPTION_PREFIX}{Strike}{CE|PE}.parquet"`.
- **Why this shape:**: experiments are run as self-contained notebooks that read local files, reindex ticks to 1s grids, compute implied vols/deltas, simulate hedges, and write CSV outputs (e.g. `gamma_scalping_results.csv`).

**Key Files & Patterns (discoverable examples)**
- **Notebook:**: `delta-hedhged-gamma-scalp.ipynb` — canonical example of coding patterns used (config at top, data loading, helper functions, simulation loop, CSV output).
- **Data folder:**: `data/2025-11-20/` contains `FUT.csv`, `NIFTY_FUT.csv`, `kinetic_trades.csv`, and parquet/CSV outputs — treat this as the canonical input layout when testing.
- **Filename pattern:**: option files are opened with `fname = f"{OPTION_PREFIX}{K}{opt_type}.parquet"` — the agent should not change this pattern unless updating the top-level `OPTION_PREFIX` and `STRIKES` constants.
- **Expected columns:**: ticks files use `['Date','Time','LTP']`; trades file expects `['Entry_Time','Exit_Time']` (ISO/datetime parseable).

**Developer workflows & commands**
- **Interactive runs:**: open the notebook in Jupyter / VS Code notebook and run cells top→down after setting config constants at the top (FUT_FILE, OPTION_PREFIX, STRIKES, EXPIRY_DATE).
- **Non-interactive execution (when needed):**: convert to script then run (works because notebooks use top-level code with prints):
  - `jupyter nbconvert --to script delta-hedhged-gamma-scalp.ipynb`
  - `python delta-hedhged-gamma-scalp.py`
- **Parquet dependency:**: notebooks use `pd.read_parquet()` — ensure `pyarrow` or `fastparquet` is available in the environment.

**Project-specific conventions**
- **Top-of-file configuration:**: change `FUT_FILE`, `TRADES_FILE`, `OPTION_PREFIX`, `STRIKES`, `EXPIRY_DATE` in the notebook rather than editing functions; tests and outputs rely on these constants.
- **Date/time parsing:**: code expects `'%d/%m/%Y %H:%M:%S.%f'` format; when adding new data, preserve that format or update the parsing logic explicitly in the notebook.
- **Small, local outputs:**: notebooks write CSVs to the working directory (e.g., `gamma_scalping_results.csv`). Don't assume a separate `outputs/` folder exists.

**Integration points & external dependencies**
- **Python packages used (explicitly visible):**: `pandas`, `numpy`, `math`, `os` — plus parquet backend (`pyarrow` or `fastparquet`) if reading `.parquet` files.
- **Data inputs:**: local CSV and parquet files in repository root or `data/*` directories; notebook code checks `os.path.exists(fname)` and skips missing files with a warning.

**How agents should edit code**
- **Small edits only:**: prefer changing configuration constants or adding well-contained helper functions in the same notebook. For larger refactors, create a new `.py` module and keep notebooks as examples.
- **Preserve observable behavior:**: outputs (CSV headers and column names) are used for downstream analysis — keep keys like `Net_PnL_pts` and `Net_PnL_rupees` consistent.

**When to ask the user**
- **Missing data or naming ambiguity:**: ask if option files live in a different folder or if `OPTION_PREFIX` should change.
- **Runtime environment:**: ask which Python environment or parquet engine to target before changing `read_parquet` calls.

If anything above is incomplete or unclear, tell me which area (data layout, file naming, notebook-to-script flow, dependencies) you want expanded and I will iterate.
