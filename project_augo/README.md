# Project Augo 🎯

**EPL Betting Framework with ML + LLM Fusion**

A complete Windows-native framework for English Premier League match prediction, combining quantitative machine learning with qualitative LLM-powered sentiment analysis.

## Features

### 🔢 Quantitative Analysis
- **Data Source**: football-data.co.uk (historical EPL match data)
- **Features**: Goals, shots, corners, cards, odds
- **ML Model**: XGBoost with TimeSeriesSplit validation
- **Calibration**: Isotonic regression for reliable probabilities

### 📰 Qualitative Analysis  
- **Data Sources**: BBC Sport, Guardian Football, Sky Sports RSS feeds
- **LLM Processing**: Local Ollama instance (qwen:14b or deepseek-r1:14b)
- **Extracted Metrics**:
  - Key absences impact (0-10)
  - Fatigue/rotation risk (0-10)
  - Morale sentiment score (-5 to +5)
  - Tactical summary

### 🔗 Hybrid Fusion Engine
- Combines XGBoost probabilities with LLM qualitative signals
- Vig removal from market odds (Shin method)
- Edge calculation and Kelly Criterion staking
- Risk controls: max stake %, exposure limits, drawdown circuit breaker

### 🖥️ Windows GUI
- Single-window Tkinter application
- Tabs: Dashboard, Data Ingestion, Sentiment Scan, Predictions, Telegram Export
- Real-time status monitoring
- One-click data fetching and model training

### 📤 Telegram Integration
- Formatted prediction messages
- Clipboard copy functionality
- Direct bot API integration

## Windows Installation

### Prerequisites

1. **Python 3.10+** (from [python.org](https://python.org))
   - ✅ Check "Add Python to PATH" during installation
   - ✅ Ensure "tcl/tk and IDLE" is selected (for GUI)

2. **PostgreSQL + TimescaleDB**
   - Download from [postgresql.org](https://www.postgresql.org/download/windows/)
   - Install TimescaleDB extension

3. **Ollama** (local LLM runtime)
   - Download from [ollama.ai](https://ollama.ai)
   - Pull model: `ollama pull qwen:14b`

### Setup Steps

```powershell
# 1. Clone repository
cd C:\Users\YourName\Documents
git clone <repo-url> project_augo
cd project_augo

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure database
# Edit config/settings.py with your PostgreSQL credentials

# 5. Initialize database
python -c "from src.database.schema import DatabaseManager; db = DatabaseManager(host='localhost', database='project_augo', user='postgres', password='YOUR_PASSWORD'); db.initialize_schema()"

# 6. Launch GUI
python main.py --gui
```

## Usage

### Command Line

```powershell
# Run full pipeline
python main.py --pipeline

# Launch GUI
python main.py --gui

# Fetch latest match data only
python main.py --fetch

# Run tests
python main.py --test
```

### GUI Workflow

1. **Dashboard**: Check system status (database, Ollama connection)
2. **Data Ingestion**: 
   - Click "Fetch Match Data" for historical results
   - Click "Fetch & Parse Articles" for qualitative signals
3. **Sentiment Scan**: Paste article text for instant LLM analysis
4. **Predictions**: View gameweek predictions with probabilities and stakes
5. **Telegram Export**: Generate and send formatted predictions

## Project Structure

```
project_augo/
├── config/
│   └── settings.py          # Configuration dataclasses
├── schemas/
│   └── qualitative.py       # Pydantic validation schemas
├── src/
│   ├── ingestion/
│   │   ├── quantitative.py  # football-data.co.uk ingestion
│   │   └── qualitative.py   # RSS + Ollama processing
│   ├── ml/
│   │   ├── xgb_model.py     # XGBoost predictor
│   │   └── fusion.py        # Hybrid fusion + bankroll
│   ├── database/
│   │   └── schema.py        # PostgreSQL schema + CRUD
│   └── gui/
│       └── app.py           # Tkinter GUI
├── tests/
│   └── test_core.py         # Unit tests
├── main.py                  # CLI entry point
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Configuration

Edit `config/settings.py`:

```python
# Database
DatabaseConfig(
    host="localhost",
    port=5432,
    database="project_augo",
    user="postgres",
    password="YOUR_PASSWORD"
)

# Ollama
OllamaConfig(
    base_url="http://localhost:11434",
    model="qwen:14b"
)

# Bankroll
BankrollConfig(
    kelly_fraction=0.25,      # Quarter-Kelly
    max_stake_percent=0.05,   # 5% max per bet
    drawdown_limit=0.20       # 20% circuit breaker
)
```

## Testing

```powershell
# Activate venv first
.\venv\Scripts\Activate.ps1

# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/test_core.py::TestFusionEngine -v
```

## Disclaimer

⚠️ **This software is for educational and research purposes only.**

- Past performance does not guarantee future results
- Always gamble responsibly
- Never bet more than you can afford to lose
- This is not financial advice

## License

MIT License - See LICENSE file for details

## Support

For issues or questions:
1. Check that Ollama is running: `http://localhost:11434`
2. Verify PostgreSQL connection
3. Review logs in `%LOCALAPPDATA%\ProjectAugo\logs\`
