# Project Augo - EPL Betting Framework

## Overview

**Project Augo** is a fully autonomous, hybrid Premier League betting framework that fuses quantitative historical metrics with local LLM-parsed qualitative signals to find mispriced odds in English Premier League betting markets.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PROJECT AUGO                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Quantitative │  │ Qualitative  │  │   Database   │          │
│  │   Ingestion  │  │   Ingestion  │  │  (Timescale) │          │
│  │  (CSV/HTTP)  │  │ (RSS + LLM)  │  │              │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                  │                   │
│         └────────────┬────┴──────────────────┘                   │
│                      │                                          │
│              ┌───────▼────────┐                                 │
│              │ Feature Engine │                                 │
│              └───────┬────────┘                                 │
│                      │                                          │
│         ┌────────────┼────────────┐                             │
│         │            │            │                             │
│  ┌──────▼─────┐ ┌────▼────┐ ┌────▼─────┐                       │
│  │  XGBoost   │ │ Hybrid  │ │ Bankroll │                       │
│  │   Model    │ │ Fusion  │ │  Engine  │                       │
│  └──────┬─────┘ └────┬────┘ └────┬─────┘                       │
│         │            │            │                             │
│         └────────────┼────────────┘                             │
│                      │                                          │
│              ┌───────▼────────┐                                 │
│              │     GUI /      │                                 │
│              │   Telegram     │                                 │
│              └────────────────┘                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Hardware Requirements

- **CPU**: Ryzen 5 7600 or equivalent
- **GPU**: RX 7800 XT (16GB VRAM) - for local LLM inference
- **RAM**: 32GB minimum
- **Storage**: 10GB+ for historical data and models

## Software Stack

- **Python**: 3.10+
- **Database**: PostgreSQL 15+ with TimescaleDB extension
- **Local LLM**: Ollama running Qwen:14b or DeepSeek
- **Core Libraries**: Polars, Scikit-Learn, XGBoost, Pydantic, Psycopg2

## Installation & Setup

### Step 1: Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv postgresql postgresql-contrib

# Install TimescaleDB
sudo add-apt-repository ppa:timescale/timescaledb-release
sudo apt-get update
sudo apt-get install -y timescaledb-2-postgresql-15
```

### Step 2: Set Up Python Environment

```bash
cd /workspace/project_augo

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install polars pandas xgboost scikit-learn psycopg2-binary requests feedparser pydantic numpy
```

### Step 3: Configure PostgreSQL Database

```bash
# Start PostgreSQL service
sudo systemctl start postgresql

# Create database and user
sudo -u postgres psql << EOF
CREATE DATABASE project_augo;
CREATE USER augo_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE project_augo TO augo_user;
\q
EOF

# Enable TimescaleDB extension
sudo -u postgres psql -d project_augo -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"
```

### Step 4: Set Up Ollama with Local LLM

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the recommended model (Qwen 14B)
ollama pull qwen:14b

# Verify Ollama is running
curl http://localhost:11434/api/tags

# Optional: Test the model
ollama run qwen:14b "Hello, how are you?"
```

### Step 5: Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Database configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=project_augo
DB_USER=augo_user
DB_PASSWORD=your_secure_password

# Ollama configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen:14b

# Application settings
KELLY_FRACTION=0.25
MIN_EDGE_THRESHOLD=0.05
MAX_DRAWDOWN_LIMIT=0.20
```

### Step 6: Verify Installation

```bash
# Run a simple test
python main.py --help

# Test quantitative ingestion
python -c "from src.ingestion.quantitative import QuantitativeIngestionEngine; e = QuantitativeIngestionEngine(); print('Quant engine OK')"

# Test qualitative ingestion (requires Ollama running)
python -c "from src.ingestion.qualitative import QualitativeIngestionEngine; e = QualitativeIngestionEngine(); print('Qual engine OK')"
```

## Usage Guide

### First-Time Setup

1. **Initialize the Database**
   ```bash
   source venv/bin/activate
   python main.py --pipeline
   ```
   
   This will:
   - Create all database tables
   - Fetch historical EPL data (last 3 seasons)
   - Process news articles through the LLM
   - Train the XGBoost model with walk-forward validation
   - Generate initial predictions

2. **Review Model Performance**
   
   After training, check the output for:
   - **Brier Score**: Should be < 0.20 for good calibration
   - **Log Loss**: Should be < 1.0
   - **Accuracy**: Typically 45-55% for 3-class EPL prediction

### Daily Operations

#### Option A: Command Line

```bash
# Generate predictions only (uses existing model)
python main.py --predict

# Retrain model with new data
python main.py --train --seasons 2425

# Full pipeline refresh
python main.py --pipeline
```

#### Option B: GUI Interface

```bash
python main.py --gui
```

The GUI provides:
- 📊 **Dashboard**: System status overview
- 📁 **Data Ingestion**: Fetch CSV data and RSS articles
- 🧠 **Sentiment Scanner**: Analyze team-specific news
- ⚽ **Predictions**: View odds, edges, and recommendations
- ⚙️ **Settings**: Configure bankroll, thresholds, Ollama

### Exporting to Telegram

1. Generate predictions via GUI or CLI
2. Click "Export to Telegram" button in GUI
3. Message is copied to clipboard
4. Paste into your Telegram channel

Sample output:
```
🏴󠁧󠁢󠁥󠁮󠁧󠁿 PROJECT AUGO - MATCH PREDICTIONS 🏴󠁧󠁢󠁥󠁮󠁧󠁿

🏠 Arsenal vs Everton
📅 2024-12-14
💡 Recommendation: Home Win
📊 Probability: 65% / 20% / 15%
💰 Odds: 1.45 / 4.50 / 7.00
📈 Edge: +8%
💵 Stake: 3.2% of bankroll
```

## Module Reference

### Quantitative Ingestion (`src/ingestion/quantitative.py`)

```python
from src.ingestion.quantitative import QuantitativeIngestionEngine

engine = QuantitativeIngestionEngine()

# Fetch specific season
df = engine.fetch_season_data('2425')

# Fetch multiple seasons
df = engine.fetch_multiple_seasons(['2425', '2324', '2223'])

# Clean and standardize
df_clean = engine.clean_and_standardize(df)
```

### Qualitative Ingestion (`src/ingestion/qualitative.py`)

```python
from src.ingestion.qualitative import QualitativeIngestionEngine

engine = QualitativeIngestionEngine(
    ollama_base_url='http://localhost:11434',
    model_name='qwen:14b'
)

# Fetch RSS articles
articles = engine.fetch_rss_feeds(max_entries_per_feed=20)

# Process with LLM
signals = engine.process_articles(articles)

# Get team-specific signals
team_signals = engine.extract_team_signals(signals, 'Arsenal')
```

### ML Model (`src/ml/xgb_model.py`)

```python
from src.ml.xgb_model import XGBoostModel

model = XGBoostModel()

# Walk-forward validation
metrics = model.train_walk_forward(df)

# Train final model
model.fit_final_model(df)

# Predict probabilities
probs = model.predict_proba(new_matches_df)
```

### Bankroll Management (`src/ml/fusion.py`)

```python
from src.ml.fusion import BankrollEngine

engine = BankrollEngine()

# Calculate fair probability from odds
fair_probs = engine.calculate_fair_probability(1.45, 4.50, 7.00)

# Calculate edge
edge = engine.calculate_edge(model_prob=0.65, market_prob=0.55)

# Kelly stake
stake = engine.calculate_kelly_stake(edge=0.10, odds=1.45)
```

## Configuration

Edit `config/settings.py` to customize:

```python
@dataclass
class ModelConfig:
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    n_splits: int = 5  # Walk-forward folds
    test_size: int = 10  # Matches per fold

@dataclass
class BankrollConfig:
    kelly_fraction: float = 0.25  # Quarter-Kelly
    max_stake_percentage: float = 0.05  # 5% max
    min_edge_threshold: float = 0.05  # 5% minimum edge
    max_drawdown_limit: float = 0.20  # 20% circuit breaker
```

## Testing & Validation

### Unit Tests

```bash
# Run tests
pytest tests/ -v
```

### Backtesting

```python
from src.ml.xgb_model import XGBoostModel

model = XGBoostModel()
metrics = model.train_walk_forward(df)

print(f"Brier Score: {metrics['mean_brier_score']:.4f}")
print(f"Log Loss: {metrics['mean_log_loss']:.4f}")
print(f"Accuracy: {metrics['mean_accuracy']:.4f}")
```

### Expected Performance Metrics

| Metric | Good | Excellent |
|--------|------|-----------|
| Brier Score | < 0.20 | < 0.15 |
| Log Loss | < 1.0 | < 0.85 |
| Accuracy | > 45% | > 50% |
| ROI (simulated) | > 5% | > 10% |

## Troubleshooting

### Ollama Connection Failed

```bash
# Check if Ollama is running
systemctl status ollama

# Restart Ollama
ollama serve

# Test connection
curl http://localhost:11434/api/tags
```

### Database Connection Error

```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Verify credentials
psql -h localhost -U augo_user -d project_augo
```

### No Predictions Generated

- Check if edge threshold is too high (default 5%)
- Verify odds data is present in matches
- Ensure model is trained (`--train` flag)

## Risk Warnings

⚠️ **IMPORTANT**: This software is for educational and research purposes only.

- Past performance does not guarantee future results
- Sports betting involves significant financial risk
- Never bet more than you can afford to lose
- The Kelly Criterion can still result in substantial drawdowns
- Always use responsible gambling practices

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please read CONTRIBUTING.md first.

## Support

For issues and feature requests, please open a GitHub issue.
