"""
Quantitative Data Ingestion Engine
Fetches and parses EPL data from football-data.co.uk using Polars
Windows-optimized with robust error handling
"""
import polars as pl
import requests
from io import StringIO
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantitativeIngestionEngine:
    """
    Automated ingestion of EPL historical data from football-data.co.uk
    
    Seasons format: "2425" for 2024/25, "2324" for 2023/24, etc.
    """
    
    BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
    
    # Column mappings from source to standardized names
    COLUMN_MAPPING = {
        # Core results
        'FTHG': 'ft_home_goals',
        'FTAG': 'ft_away_goals',
        'HTHG': 'ht_home_goals',
        'HTAG': 'ht_away_goals',
        'FTR': 'ft_result',  # H/D/A
        
        # Shots
        'HS': 'home_shots',
        'AS': 'away_shots',
        'HST': 'home_shots_on_target',
        'AST': 'away_shots_on_target',
        
        # Corners
        'HC': 'home_corners',
        'AC': 'away_corners',
        
        # Cards
        'HY': 'home_yellow_cards',
        'AY': 'away_yellow_cards',
        'HR': 'home_red_cards',
        'AR': 'away_red_cards',
        
        # Odds (Bet365)
        'B365H': 'odds_home',
        'B365D': 'odds_draw',
        'B365A': 'odds_away',
        
        # Metadata
        'Date': 'match_date',
        'HomeTeam': 'home_team',
        'AwayTeam': 'away_team',
        'Div': 'division',
    }
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the ingestion engine.
        
        Args:
            cache_dir: Directory to cache downloaded CSVs. Defaults to local cache.
        """
        if cache_dir is None:
            # Windows-compatible cache directory
            import os
            cache_dir = Path(os.environ.get("LOCALAPPDATA", ".")) / "ProjectAugo" / "cache" / "quantitative"
        
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_season_code(self, year_start: int) -> str:
        """Convert year to season code (e.g., 2024 -> '2425')"""
        year_end = year_start + 1
        return f"{str(year_start)[-2:]}{str(year_end)[-2:]}"
    
    def fetch_season_data(
        self, 
        season_years: List[int],
        force_refresh: bool = False
    ) -> pl.DataFrame:
        """
        Fetch and combine data for multiple seasons.
        
        Args:
            season_years: List of starting years (e.g., [2024, 2023, 2022])
            force_refresh: If True, re-download even if cached
            
        Returns:
            Polars DataFrame with all seasons combined
        """
        all_dataframes = []
        
        for year in season_years:
            season_code = self._get_season_code(year)
            df = self._fetch_single_season(season_code, force_refresh)
            if df is not None:
                all_dataframes.append(df)
                logger.info(f"✓ Loaded season {year}/{year+1}: {len(df)} matches")
        
        if not all_dataframes:
            raise ValueError("No data fetched for any requested season")
        
        # Combine all seasons
        combined = pl.concat(all_dataframes, how="vertical_relaxed")
        logger.info(f"✓ Total matches loaded: {len(combined)}")
        
        return combined
    
    def _fetch_single_season(
        self, 
        season_code: str, 
        force_refresh: bool = False
    ) -> Optional[pl.DataFrame]:
        """Fetch data for a single season"""
        url = self.BASE_URL.format(season=season_code)
        cache_file = self.cache_dir / f"E0_{season_code}.csv"
        
        # Check cache first
        if cache_file.exists() and not force_refresh:
            logger.info(f"Loading from cache: {cache_file}")
            try:
                return self._parse_csv(cache_file.read_text(encoding='utf-8'))
            except Exception as e:
                logger.warning(f"Cache read failed, re-downloading: {e}")
        
        # Download from source
        logger.info(f"Downloading season {season_code} from {url}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save to cache
            csv_content = response.text
            cache_file.write_text(csv_content, encoding='utf-8')
            logger.info(f"✓ Cached to {cache_file}")
            
            return self._parse_csv(csv_content)
            
        except requests.RequestException as e:
            logger.error(f"Download failed for season {season_code}: {e}")
            # Try to use stale cache if available
            if cache_file.exists():
                logger.warning("Using stale cache as fallback")
                return self._parse_csv(cache_file.read_text(encoding='utf-8'))
            return None
    
    def _parse_csv(self, csv_content: str) -> pl.DataFrame:
        """Parse CSV content into Polars DataFrame with type conversions"""
        # Use StringIO for in-memory parsing
        csv_io = StringIO(csv_content)
        
        df = pl.read_csv(csv_io)
        
        # Rename columns to standardized names
        rename_map = {k: v for k, v in self.COLUMN_MAPPING.items() if k in df.columns}
        df = df.rename(rename_map)
        
        # Convert date column (format: DD/MM/YYYY)
        if 'match_date' in df.columns:
            df = df.with_columns(
                pl.col('match_date').str.strptime(pl.Date, "%d/%m/%Y")
            )
        
        # Convert odds columns to float (handle non-numeric values)
        odds_cols = ['odds_home', 'odds_draw', 'odds_away']
        for col in odds_cols:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col).cast(pl.Float64, strict=False)
                )
        
        # Add derived columns
        df = self._add_derived_features(df)
        
        return df
    
    def _add_derived_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add computed features to the dataframe"""
        # Goal difference
        if 'ft_home_goals' in df.columns and 'ft_away_goals' in df.columns:
            df = df.with_columns(
                (pl.col('ft_home_goals') - pl.col('ft_away_goals')).alias('goal_difference')
            )
        
        # Total goals
        if 'ft_home_goals' in df.columns and 'ft_away_goals' in df.columns:
            df = df.with_columns(
                (pl.col('ft_home_goals') + pl.col('ft_away_goals')).alias('total_goals')
            )
        
        # Shot accuracy (shots on target / total shots)
        if 'home_shots' in df.columns and 'home_shots_on_target' in df.columns:
            df = df.with_columns(
                (pl.col('home_shots_on_target') / pl.col('home_shots').clip(lower_bound=1)).alias('home_shot_accuracy')
            )
        if 'away_shots' in df.columns and 'away_shots_on_target' in df.columns:
            df = df.with_columns(
                (pl.col('away_shots_on_target') / pl.col('away_shots').clip(lower_bound=1)).alias('away_shot_accuracy')
            )
        
        return df
    
    def get_latest_season(self) -> pl.DataFrame:
        """Fetch only the current/latest season data"""
        current_year = datetime.now().year
        # Adjust for season that spans two years
        if datetime.now().month < 8:  # Before August, previous season is current
            current_year -= 1
        
        return self.fetch_season_data([current_year])
    
    def validate_data_quality(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Run data quality checks and return report"""
        report = {
            'total_matches': len(df),
            'missing_values': {},
            'date_range': None,
            'teams_count': 0,
            'odds_coverage': 0.0,
        }
        
        # Missing values
        for col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                report['missing_values'][col] = f"{null_count} ({100*null_count/len(df):.1f}%)"
        
        # Date range
        if 'match_date' in df.columns:
            report['date_range'] = {
                'start': str(df['match_date'].min()),
                'end': str(df['match_date'].max())
            }
        
        # Unique teams
        if 'home_team' in df.columns:
            teams = set(df['home_team'].to_list()) | set(df['away_team'].to_list())
            report['teams_count'] = len(teams)
        
        # Odds coverage
        if 'odds_home' in df.columns:
            odds_available = df['odds_home'].is_not_null().sum()
            report['odds_coverage'] = float(odds_available) / len(df) * 100
        
        return report


# Example usage
if __name__ == "__main__":
    engine = QuantitativeIngestionEngine()
    
    # Fetch last 3 seasons
    df = engine.fetch_season_data([2024, 2023, 2022])
    
    print(f"\nLoaded {len(df)} matches")
    print(df.head())
    
    # Quality report
    report = engine.validate_data_quality(df)
    print(f"\nData Quality Report:")
    for key, value in report.items():
        print(f"  {key}: {value}")
