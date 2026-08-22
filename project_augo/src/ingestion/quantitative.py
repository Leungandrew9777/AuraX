"""
Project Augo - Quantitative Data Ingestion Engine
Fetches and parses historical EPL data from Football-Data.co.uk
"""
import polars as pl
from io import StringIO
import requests
from typing import List, Dict, Optional
from datetime import datetime
import logging

from config.settings import config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantitativeIngestionEngine:
    """
    Engine for fetching, parsing, and cleaning EPL historical data.
    
    Handles multiple seasons and standardizes column names across
    different CSV formats from football-data.co.uk
    """
    
    # Core columns we always want to extract
    CORE_COLUMNS = [
        # Match metadata
        'Date', 'HomeTeam', 'AwayTeam', 'FTR',  # Full-time result
        # Goals
        'FTHG', 'FTAG', 'HTHG', 'HTAG',  # Half-time/Full-time goals
        # Shots
        'HS', 'AS', 'HST', 'AST',  # Shots / Shots on target
        # Corners
        'HC', 'AC',
        # Cards
        'HY', 'AY', 'HR', 'AR',  # Yellow/Red cards
        # Betting odds (Bet365)
        'B365H', 'B365D', 'B365A',
        # Additional odds if available
        'PSH', 'PSD', 'PSA',  # Pinnacle
        'WHH', 'WHD', 'WHA'   # William Hill
    ]
    
    SEASON_MAP = {
        '2425': '2024-2025',
        '2324': '2023-2024',
        '2223': '2022-2023',
        '2122': '2021-2022',
        '2021': '2020-2021',
        '1920': '2019-2020',
    }
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or config.data_sources.football_data_base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Project Augo - Sports Analytics)'
        })
    
    def fetch_season_data(self, season_code: str) -> Optional[pl.DataFrame]:
        """
        Fetch CSV data for a specific season.
        
        Args:
            season_code: Season identifier (e.g., '2425' for 2024-2025)
            
        Returns:
            Polars DataFrame with cleaned data, or None if fetch fails
        """
        url = f"{self.base_url}/{season_code}/E0.csv"
        logger.info(f"Fetching season {season_code} from {url}")
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse CSV with Polars
            csv_content = response.text
            df = pl.read_csv(StringIO(csv_content))
            
            # Add season identifier
            df = df.with_columns(
                pl.lit(season_code).alias('season_code'),
                pl.lit(self.SEASON_MAP.get(season_code, 'Unknown')).alias('season_name'),
                pl.lit(datetime.now()).alias('ingested_at')
            )
            
            logger.info(f"Successfully fetched {len(df)} matches for season {season_code}")
            return df
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch season {season_code}: {e}")
            return None
        except pl.exceptions.PolarsError as e:
            logger.error(f"Failed to parse CSV for season {season_code}: {e}")
            return None
    
    def fetch_multiple_seasons(self, season_codes: List[str]) -> pl.DataFrame:
        """
        Fetch and concatenate data from multiple seasons.
        
        Args:
            season_codes: List of season identifiers
            
        Returns:
            Combined Polars DataFrame
        """
        dfs = []
        for code in season_codes:
            df = self.fetch_season_data(code)
            if df is not None:
                dfs.append(df)
        
        if not dfs:
            raise ValueError("No seasons were successfully fetched")
        
        # Concatenate all seasons
        combined = pl.concat(dfs, how='vertical_relaxed')
        logger.info(f"Combined {len(combined)} total matches from {len(season_codes)} seasons")
        return combined
    
    def clean_and_standardize(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Clean and standardize the raw dataframe.
        
        - Handle missing values
        - Convert date formats
        - Standardize team names
        - Create derived features
        """
        df_clean = df.clone()
        
        # Convert Date column to proper datetime (UK format DD/MM/YYYY)
        if 'Date' in df_clean.columns:
            df_clean = df_clean.with_columns(
                pl.col('Date').str.strptime(pl.Date, '%d/%m/%Y').alias('match_date')
            )
        
        # Standardize result column
        if 'FTR' in df_clean.columns:
            df_clean = df_clean.with_columns(
                pl.col('FTR').map_elements(
                    lambda x: {'H': 'home_win', 'D': 'draw', 'A': 'away_win'}.get(x, 'unknown'),
                    return_dtype=pl.Utf8
                ).alias('result_encoded')
            )
        
        # Fill missing numerical columns with 0 (common for older seasons)
        numeric_cols = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean = df_clean.with_columns(
                    pl.col(col).fill_null(0).alias(col)
                )
        
        # Create goal difference feature
        if 'FTHG' in df_clean.columns and 'FTAG' in df_clean.columns:
            df_clean = df_clean.with_columns(
                (pl.col('FTHG') - pl.col('FTAG')).alias('goal_difference')
            )
        
        # Create total goals feature
        if 'FTHG' in df_clean.columns and 'FTAG' in df_clean.columns:
            df_clean = df_clean.with_columns(
                (pl.col('FTHG') + pl.col('FTAG')).alias('total_goals')
            )
        
        # Calculate implied probabilities from odds (if available)
        if all(col in df_clean.columns for col in ['B365H', 'B365D', 'B365A']):
            df_clean = self._calculate_implied_probabilities(df_clean)
        
        logger.info(f"Cleaned dataframe shape: {df_clean.shape}")
        return df_clean
    
    def _calculate_implied_probabilities(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate fair implied probabilities by removing bookmaker margin.
        Uses the Power method for normalization.
        """
        df_prob = df.clone()
        
        # Calculate raw implied probabilities
        df_prob = df_prob.with_columns(
            (1.0 / pl.col('B365H')).alias('raw_prob_home'),
            (1.0 / pl.col('B365D')).alias('raw_prob_draw'),
            (1.0 / pl.col('B365A')).alias('raw_prob_away')
        )
        
        # Sum of probabilities (includes vig/margin)
        df_prob = df_prob.with_columns(
            (pl.col('raw_prob_home') + pl.col('raw_prob_draw') + pl.col('raw_prob_away'))
            .alias('total_implied_prob')
        )
        
        # Normalize to remove vig (Power method)
        df_prob = df_prob.with_columns(
            (pl.col('raw_prob_home') / pl.col('total_implied_prob')).alias('fair_prob_home'),
            (pl.col('raw_prob_draw') / pl.col('total_implied_prob')).alias('fair_prob_draw'),
            (pl.col('raw_prob_away') / pl.col('total_implied_prob')).alias('fair_prob_away')
        )
        
        # Calculate bookmaker margin
        df_prob = df_prob.with_columns(
            (pl.col('total_implied_prob') - 1.0).alias('bookmaker_margin')
        )
        
        return df_prob
    
    def select_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Select only the core features needed for modeling.
        """
        available_cols = df.columns
        cols_to_select = [col for col in self.CORE_COLUMNS if col in available_cols]
        
        # Always include these metadata columns
        meta_cols = ['match_date', 'HomeTeam', 'AwayTeam', 'season_code', 'result_encoded']
        meta_cols = [col for col in meta_cols if col in available_cols]
        
        selected = df.select(meta_cols + cols_to_select)
        logger.info(f"Selected {len(selected.columns)} features for modeling")
        return selected
    
    def ingest_latest_season(self) -> pl.DataFrame:
        """
        Convenience method to fetch and process the latest available season.
        """
        df_raw = self.fetch_season_data('2425')  # Latest season
        if df_raw is None:
            # Fall back to previous season
            df_raw = self.fetch_season_data('2324')
        
        if df_raw is None:
            raise RuntimeError("Could not fetch any season data")
        
        df_clean = self.clean_and_standardize(df_raw)
        df_final = self.select_features(df_clean)
        
        return df_final
    
    def save_to_csv(self, df: pl.DataFrame, filepath: str) -> None:
        """Save processed dataframe to CSV."""
        df.write_csv(filepath)
        logger.info(f"Saved dataframe to {filepath}")
    
    def load_from_csv(self, filepath: str) -> pl.DataFrame:
        """Load previously saved dataframe from CSV."""
        df = pl.read_csv(filepath)
        logger.info(f"Loaded dataframe from {filepath} with {len(df)} rows")
        return df
