"""
IndexBuddy configuration — reads connection settings from .env
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)

# ── MySQL connection ────────────────────────────────────────────
MYSQL_HOST: str = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER: str = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB: str = os.getenv("MYSQL_DB", "indexbuddy")

# ── Benchmark settings ──────────────────────────────────────────
BENCHMARK_TABLE: str = "orders_benchmark"
BENCHMARK_ROWS: int = 1_000_000          # configurable via env
BENCHMARK_ROWS = int(os.getenv("BENCHMARK_ROWS", str(BENCHMARK_ROWS)))

# ── Model settings ──────────────────────────────────────────────
SPEEDUP_THRESHOLD: float = float(os.getenv("SPEEDUP_THRESHOLD", "1.5"))
MODEL_PATH: str = str(Path(__file__).resolve().parent / "models" / "xgb_speedup.joblib")
SCALER_PATH: str = str(Path(__file__).resolve().parent / "models" / "scaler.joblib")
LABEL_CSV: str = str(Path(__file__).resolve().parent / "labeling" / "training_data.csv")
SHAP_PLOT_PATH: str = str(Path(__file__).resolve().parent / "evaluation" / "shap_summary.png")

# ── Labeling ────────────────────────────────────────────────────
LABELING_RUNS: int = 5          # total runs per measurement (1 warm-up discarded)
DASHBOARD_RUNS: int = 10        # total runs in the dashboard demo

# ── Candidate generation ────────────────────────────────────────
MAX_CANDIDATES_PER_QUERY: int = 15
