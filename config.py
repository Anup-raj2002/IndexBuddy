"""
IndexBuddy configuration â€” reads connection settings from .env
Zero hardcoded table or column names.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)

# â”€â”€ Local MySQL 8.0.18+ connection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
MYSQL_HOST: str = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER: str = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB: str = os.getenv("MYSQL_DB", "")

# â”€â”€ CTU Prague Relational Learning Repository (public read-only) â”€
CTU_HOST: str = os.getenv("CTU_HOST", "relational.fel.cvut.cz")
CTU_PORT: int = int(os.getenv("CTU_PORT", "3306"))
CTU_USER: str = os.getenv("CTU_USER", "guest")
CTU_PASSWORD: str = os.getenv("CTU_PASSWORD", "ctu-relational")

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
PROJECT_ROOT: Path = Path(__file__).resolve().parent
RAW_DUMP_DIR: Path = PROJECT_ROOT / "data" / "raw"
ORIGINAL_INDEXES_PATH: Path = PROJECT_ROOT / "data" / "original_indexes.json"
WORKLOAD_CSV: str = str(PROJECT_ROOT / "labeling" / "generated_workload.csv")
LABEL_CSV: str = str(PROJECT_ROOT / "labeling" / "training_data.csv")
MODEL_PATH: str = str(PROJECT_ROOT / "models" / "xgb_speedup.joblib")
SHAP_PLOT_PATH: str = str(PROJECT_ROOT / "evaluation" / "shap_summary.png")
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# â”€â”€ Model & Evaluation settings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SPEEDUP_THRESHOLD: float = float(os.getenv("SPEEDUP_THRESHOLD", "1.5"))

# â”€â”€ Labeling & Benchmarking settings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
LABELING_RUNS: int = 9          # total runs per measurement (1 warm-up discarded)
DASHBOARD_RUNS: int = 10        # total runs in the dashboard demo (1 warm-up discarded)
LARGE_TABLE_SAMPLE_THRESHOLD: int = 200_000  # use sampling when table rows exceed this
SAMPLE_SIZE_ROWS: int = 50_000
# Skip labeling queries whose unindexed runtime is below this many ms: speedups
# measured on sub-few-millisecond queries are mostly timing noise.
MIN_BASELINE_MS: float = float(os.getenv("MIN_BASELINE_MS", "3.0"))

# â”€â”€ Candidate generation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
MAX_CANDIDATES_PER_QUERY: int = 12

# â”€â”€ Selected Real CTU Prague Databases (Leave-Databases-Out) â”€â”€â”€â”€
SELECTED_CTU_DATABASES: list[str] = [
    "financial",
    "ErgastF1",
    "stats_CEB",
    "imdb_full",
    "Walmart_2014",
    "geneea",
]
TRAIN_DATABASES: list[str] = [
    "financial",
    "ErgastF1",
    "imdb_full",
    "geneea",
]
HELDOUT_TEST_DATABASES: list[str] = [
    "stats_CEB",
    "Walmart_2014",
]
