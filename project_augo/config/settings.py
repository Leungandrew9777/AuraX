"""
Project Augo - Configuration Module
Centralized configuration for database, API endpoints, and model parameters.
"""
from dataclasses import dataclass
from typing import List, Optional
import os


@dataclass
class DatabaseConfig:
    """PostgreSQL/TimescaleDB connection settings."""
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", 5432))
    database: str = os.getenv("DB_NAME", "project_augo")
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "postgres")
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class OllamaConfig:
    """Local LLM configuration."""
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name: str = os.getenv("OLLAMA_MODEL", "qwen:14b")
    timeout: int = 120  # seconds


@dataclass
class DataSourcesConfig:
    """External data source URLs."""
    football_data_base_url: str = "https://www.football-data.co.uk/mmz4281"
    rss_feeds: List[str] = None
    
    def __post_init__(self):
        if self.rss_feeds is None:
            self.rss_feeds = [
                "https://feeds.bbci.co.uk/sport/football/rss.xml",
                "https://www.theguardian.com/football/rss",
                "https://www.skysports.com/rss/12040"
            ]


@dataclass
class ModelConfig:
    """ML model hyperparameters."""
    # XGBoost params
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_random_state: int = 42
    
    # Walk-forward validation
    n_splits: int = 5
    test_size: int = 10  # matches per fold
    
    # Calibration
    calibration_method: str = "isotonic"
    
    # Meta-learner
    meta_learner_alpha: float = 0.01  # Ridge regularization


@dataclass
class BankrollConfig:
    """Kelly criterion and risk management."""
    kelly_fraction: float = 0.25  # Quarter-Kelly
    max_stake_percentage: float = 0.05  # 5% of bankroll per bet
    min_edge_threshold: float = 0.05  # 5% edge minimum
    max_drawdown_limit: float = 0.20  # 20% drawdown circuit breaker
    initial_bankroll: float = 1000.0


@dataclass
class AppConfig:
    """Main application configuration."""
    database: DatabaseConfig = None
    ollama: OllamaConfig = None
    data_sources: DataSourcesConfig = None
    model: ModelConfig = None
    bankroll: BankrollConfig = None
    
    def __post_init__(self):
        if self.database is None:
            self.database = DatabaseConfig()
        if self.ollama is None:
            self.ollama = OllamaConfig()
        if self.data_sources is None:
            self.data_sources = DataSourcesConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.bankroll is None:
            self.bankroll = BankrollConfig()


# Global config instance
config = AppConfig()
