"""
Project Augo Configuration Settings
Windows-optimized configuration dataclasses
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os

@dataclass
class DatabaseConfig:
    """PostgreSQL/TimescaleDB connection settings for Windows"""
    host: str = "localhost"
    port: int = 5432
    database: str = "project_augo"
    user: str = "postgres"
    password: str = "your_password_here"  # Change this!
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class OllamaConfig:
    """Local LLM configuration for Windows"""
    base_url: str = "http://localhost:11434"
    model: str = "qwen:14b"  # or "deepseek-r1:14b"
    timeout: int = 120
    
@dataclass
class DataPaths:
    """File paths optimized for Windows"""
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("APPDATA", ".")) / "ProjectAugo" / "data")
    cache_dir: Path = field(default_factory=lambda: Path(os.environ.get("LOCALAPPDATA", ".")) / "ProjectAugo" / "cache")
    
    def __post_init__(self):
        # Create directories if they don't exist (Windows-compatible)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class RSSFeeds:
    """RSS feed endpoints for qualitative data"""
    bbc_sport: str = "https://feeds.bbci.co.uk/sport/football/rss.xml"
    guardian_football: str = "https://www.theguardian.com/football/rss"
    sky_sports_news: str = "https://www.skysports.com/rss/12040"
    
    @property
    def all_feeds(self) -> List[str]:
        return [self.bbc_sport, self.guardian_football, self.sky_sports_news]

@dataclass
class MLConfig:
    """Machine Learning hyperparameters"""
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    n_splits: int = 5  # TimeSeriesSplit folds
    calibration_method: str = "isotonic"
    random_state: int = 42

@dataclass
class BankrollConfig:
    """Kelly criterion and risk management settings"""
    kelly_fraction: float = 0.25  # Quarter-Kelly
    max_stake_percent: float = 0.05  # Max 5% per bet
    max_exposure: float = 0.20  # Max 20% total exposure
    drawdown_limit: float = 0.20  # 20% drawdown circuit breaker
    initial_bankroll: float = 1000.0  # Starting units

@dataclass
class Config:
    """Master configuration container"""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    paths: DataPaths = field(default_factory=DataPaths)
    rss: RSSFeeds = field(default_factory=RSSFeeds)
    ml: MLConfig = field(default_factory=MLConfig)
    bankroll: BankrollConfig = field(default_factory=BankrollConfig)

# Global config instance
config = Config()
