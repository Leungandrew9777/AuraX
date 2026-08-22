"""
Project Augo - Database Schema and Connection Management
PostgreSQL + TimescaleDB setup for time-series match data.
"""
import psycopg2
from psycopg2 import sql, extras
from typing import Optional, List, Dict
import logging
from contextlib import contextmanager

from config.settings import config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages PostgreSQL/TimescaleDB connections and schema operations.
    
    Provides idempotent table creation and efficient batch operations
    for both match data and qualitative signals.
    """
    
    # Table creation SQL statements
    CREATE_MATCHES_TABLE = """
    CREATE TABLE IF NOT EXISTS matches (
        match_id SERIAL PRIMARY KEY,
        match_date DATE NOT NULL,
        season_code VARCHAR(10) NOT NULL,
        home_team VARCHAR(50) NOT NULL,
        away_team VARCHAR(50) NOT NULL,
        
        -- Half-time stats
        ht_home_goals INTEGER DEFAULT 0,
        ht_away_goals INTEGER DEFAULT 0,
        
        -- Full-time stats
        ft_home_goals INTEGER DEFAULT 0,
        ft_away_goals INTEGER DEFAULT 0,
        result VARCHAR(20),  -- home_win, draw, away_win
        
        -- Match statistics
        home_shots INTEGER DEFAULT 0,
        away_shots INTEGER DEFAULT 0,
        home_shots_on_target INTEGER DEFAULT 0,
        away_shots_on_target INTEGER DEFAULT 0,
        home_corners INTEGER DEFAULT 0,
        away_corners INTEGER DEFAULT 0,
        home_yellow_cards INTEGER DEFAULT 0,
        away_yellow_cards INTEGER DEFAULT 0,
        home_red_cards INTEGER DEFAULT 0,
        away_red_cards INTEGER DEFAULT 0,
        
        -- Derived features
        goal_difference INTEGER DEFAULT 0,
        total_goals INTEGER DEFAULT 0,
        
        -- Timestamps
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW(),
        
        -- Unique constraint to prevent duplicates
        UNIQUE(match_date, home_team, away_team)
    );
    
    -- Create index on match_date for time-series queries
    CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);
    CREATE INDEX IF NOT EXISTS idx_matches_season ON matches(season_code);
    CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team, away_team);
    """
    
    CREATE_ODDS_TABLE = """
    CREATE TABLE IF NOT EXISTS odds (
        odds_id SERIAL PRIMARY KEY,
        match_id INTEGER REFERENCES matches(match_id) ON DELETE CASCADE,
        
        -- Bookmaker identifiers
        bookmaker VARCHAR(50) NOT NULL,  -- e.g., 'B365', 'Pinnacle', 'WilliamHill'
        
        -- Decimal odds
        home_win_odds DECIMAL(10, 4),
        draw_odds DECIMAL(10, 4),
        away_win_odds DECIMAL(10, 4),
        
        -- Implied probabilities (fair, vig removed)
        fair_prob_home DECIMAL(10, 6),
        fair_prob_draw DECIMAL(10, 6),
        fair_prob_away DECIMAL(10, 6),
        
        -- Bookmaker margin
        bookmaker_margin DECIMAL(10, 6),
        
        -- Timestamps
        odds_timestamp TIMESTAMPTZ DEFAULT NOW(),
        created_at TIMESTAMPTZ DEFAULT NOW(),
        
        UNIQUE(match_id, bookmaker, odds_timestamp)
    );
    
    -- Index for quick odds lookups
    CREATE INDEX IF NOT EXISTS idx_odds_match ON odds(match_id);
    CREATE INDEX IF NOT EXISTS idx_odds_bookmaker ON odds(bookmaker);
    """
    
    CREATE_LLM_SIGNALS_TABLE = """
    CREATE TABLE IF NOT EXISTS llm_qualitative_signals (
        signal_id SERIAL PRIMARY KEY,
        article_id VARCHAR(100) UNIQUE NOT NULL,
        
        -- Article metadata
        source VARCHAR(100) NOT NULL,
        title TEXT,
        published_at TIMESTAMPTZ NOT NULL,
        link TEXT,
        
        -- Teams mentioned
        teams_mentioned TEXT[],  -- Array of team names
        
        -- Quantitative metrics from LLM
        key_absences_impact DECIMAL(5, 2) CHECK (key_absences_impact BETWEEN 0 AND 10),
        fatigue_rotation_risk DECIMAL(5, 2) CHECK (fatigue_rotation_risk BETWEEN 0 AND 10),
        morale_sentiment_score DECIMAL(5, 2) CHECK (morale_sentiment_score BETWEEN -5 AND 5),
        
        -- Qualitative summary
        tactical_summary TEXT,
        
        -- Quality metrics
        confidence_score DECIMAL(5, 4) CHECK (confidence_score BETWEEN 0 AND 1),
        
        -- Raw payload for audit
        raw_json_payload JSONB,
        
        -- Timestamps
        ingested_at TIMESTAMPTZ DEFAULT NOW(),
        
        -- Index on teams for fast filtering
        INDEX idx_llm_teams USING GIN (teams_mentioned)
    );
    
    -- Time-based index for recent signals
    CREATE INDEX IF NOT EXISTS idx_llm_published ON llm_qualitative_signals(published_at);
    CREATE INDEX IF NOT EXISTS idx_llm_source ON llm_qualitative_signals(source);
    """
    
    CREATE_MODEL_PREDICTIONS_TABLE = """
    CREATE TABLE IF NOT EXISTS model_predictions (
        prediction_id SERIAL PRIMARY KEY,
        match_id INTEGER REFERENCES matches(match_id) ON DELETE CASCADE,
        
        -- Model version tracking
        model_version VARCHAR(50) NOT NULL,
        model_type VARCHAR(50) NOT NULL,  -- 'xgb_base', 'hybrid_meta'
        
        -- Probability outputs
        prob_home DECIMAL(10, 6) NOT NULL,
        prob_draw DECIMAL(10, 6) NOT NULL,
        prob_away DECIMAL(10, 6) NOT NULL,
        
        -- Calibration info
        calibration_method VARCHAR(50),
        is_calibrated BOOLEAN DEFAULT FALSE,
        
        -- Edge calculation
        edge_home DECIMAL(10, 6),
        edge_draw DECIMAL(10, 6),
        edge_away DECIMAL(10, 6),
        
        -- Recommended bet (if any)
        recommended_outcome VARCHAR(10),
        recommended_stake DECIMAL(10, 4),
        kelly_fraction DECIMAL(5, 4),
        
        -- Timestamps
        predicted_at TIMESTAMPTZ DEFAULT NOW(),
        
        UNIQUE(match_id, model_version, model_type)
    );
    
    CREATE INDEX IF NOT EXISTS idx_predictions_match ON model_predictions(match_id);
    CREATE INDEX IF NOT EXISTS idx_predictions_model ON model_predictions(model_type);
    """
    
    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string or config.database.connection_string
        self._initialize_timescale_extension()
    
    def _initialize_timescale_extension(self):
        """Ensure TimescaleDB extension is available."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
                    conn.commit()
                    logger.info("TimescaleDB extension initialized")
        except Exception as e:
            logger.warning(f"Could not initialize TimescaleDB extension: {e}")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = psycopg2.connect(self.connection_string)
            yield conn
        finally:
            if conn is not None:
                conn.close()
    
    @contextmanager
    def get_cursor(self, commit: bool = False):
        """Context manager for database cursors with optional commit."""
        with self.get_connection() as conn:
            cur = None
            try:
                cur = conn.cursor(cursor_factory=extras.RealDictCursor)
                yield cur
                if commit:
                    conn.commit()
            finally:
                if cur is not None:
                    cur.close()
    
    def initialize_schema(self):
        """Create all tables if they don't exist (idempotent)."""
        logger.info("Initializing database schema...")
        
        with self.get_cursor(commit=True) as cur:
            # Create tables in order (respecting foreign keys)
            cur.execute(self.CREATE_MATCHES_TABLE)
            logger.info("Created matches table")
            
            cur.execute(self.CREATE_ODDS_TABLE)
            logger.info("Created odds table")
            
            cur.execute(self.CREATE_LLM_SIGNALS_TABLE)
            logger.info("Created llm_qualitative_signals table")
            
            cur.execute(self.CREATE_MODEL_PREDICTIONS_TABLE)
            logger.info("Created model_predictions table")
        
        logger.info("Database schema initialization complete")
    
    def insert_match(self, match_data: Dict) -> Optional[int]:
        """
        Insert a single match record.
        
        Returns:
            match_id if inserted, None if duplicate
        """
        query = """
        INSERT INTO matches (
            match_date, season_code, home_team, away_team,
            ht_home_goals, ht_away_goals, ft_home_goals, ft_away_goals,
            result, home_shots, away_shots, home_shots_on_target,
            away_shots_on_target, home_corners, away_corners,
            home_yellow_cards, away_yellow_cards, home_red_cards,
            away_red_cards, goal_difference, total_goals
        ) VALUES (
            %(match_date)s, %(season_code)s, %(home_team)s, %(away_team)s,
            %(ht_home_goals)s, %(ht_away_goals)s, %(ft_home_goals)s, %(ft_away_goals)s,
            %(result)s, %(home_shots)s, %(away_shots)s, %(home_shots_on_target)s,
            %(away_shots_on_target)s, %(home_corners)s, %(away_corners)s,
            %(home_yellow_cards)s, %(away_yellow_cards)s, %(home_red_cards)s,
            %(away_red_cards)s, %(goal_difference)s, %(total_goals)s
        )
        ON CONFLICT (match_date, home_team, away_team) 
        DO UPDATE SET
            ft_home_goals = EXCLUDED.ft_home_goals,
            ft_away_goals = EXCLUDED.ft_away_goals,
            result = EXCLUDED.result,
            updated_at = NOW()
        RETURNING match_id;
        """
        
        with self.get_cursor(commit=True) as cur:
            cur.execute(query, match_data)
            result = cur.fetchone()
            return result['match_id'] if result else None
    
    def insert_matches_batch(self, matches: List[Dict]) -> int:
        """
        Batch insert multiple matches.
        
        Returns:
            Number of successfully inserted matches
        """
        query = """
        INSERT INTO matches (
            match_date, season_code, home_team, away_team,
            ht_home_goals, ht_away_goals, ft_home_goals, ft_away_goals,
            result, home_shots, away_shots, home_shots_on_target,
            away_shots_on_target, home_corners, away_corners,
            goal_difference, total_goals
        ) VALUES %s
        ON CONFLICT (match_date, home_team, away_team) DO NOTHING;
        """
        
        values = [
            (
                m.get('match_date'), m.get('season_code'),
                m.get('home_team'), m.get('away_team'),
                m.get('ht_home_goals', 0), m.get('ht_away_goals', 0),
                m.get('ft_home_goals', 0), m.get('ft_away_goals', 0),
                m.get('result'), m.get('home_shots', 0), m.get('away_shots', 0),
                m.get('home_shots_on_target', 0), m.get('away_shots_on_target', 0),
                m.get('home_corners', 0), m.get('away_corners', 0),
                m.get('goal_difference', 0), m.get('total_goals', 0)
            )
            for m in matches
        ]
        
        with self.get_cursor(commit=True) as cur:
            extras.execute_values(cur, query, values)
            return len(values)
    
    def insert_odds(self, match_id: int, odds_data: Dict) -> Optional[int]:
        """Insert odds record for a match."""
        query = """
        INSERT INTO odds (
            match_id, bookmaker, home_win_odds, draw_odds, away_win_odds,
            fair_prob_home, fair_prob_draw, fair_prob_away, bookmaker_margin
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (match_id, bookmaker, odds_timestamp) DO NOTHING
        RETURNING odds_id;
        """
        
        values = (
            match_id,
            odds_data.get('bookmaker'),
            odds_data.get('home_win_odds'),
            odds_data.get('draw_odds'),
            odds_data.get('away_win_odds'),
            odds_data.get('fair_prob_home'),
            odds_data.get('fair_prob_draw'),
            odds_data.get('fair_prob_away'),
            odds_data.get('bookmaker_margin')
        )
        
        with self.get_cursor(commit=True) as cur:
            cur.execute(query, values)
            result = cur.fetchone()
            return result['odds_id'] if result else None
    
    def insert_llm_signal(self, signal_data: Dict) -> Optional[int]:
        """Insert LLM qualitative signal."""
        query = """
        INSERT INTO llm_qualitative_signals (
            article_id, source, title, published_at, link,
            teams_mentioned, key_absences_impact, fatigue_rotation_risk,
            morale_sentiment_score, tactical_summary, confidence_score,
            raw_json_payload
        ) VALUES (
            %(article_id)s, %(source)s, %(title)s, %(published_at)s, %(link)s,
            %(teams_mentioned)s, %(key_absences_impact)s, %(fatigue_rotation_risk)s,
            %(morale_sentiment_score)s, %(tactical_summary)s, %(confidence_score)s,
            %(raw_json_payload)s
        )
        ON CONFLICT (article_id) DO UPDATE SET
            confidence_score = EXCLUDED.confidence_score,
            ingested_at = NOW()
        RETURNING signal_id;
        """
        
        with self.get_cursor(commit=True) as cur:
            cur.execute(query, signal_data)
            result = cur.fetchone()
            return result['signal_id'] if result else None
    
    def insert_llm_signals_batch(self, signals: List[Dict]) -> int:
        """Batch insert multiple LLM signals."""
        query = """
        INSERT INTO llm_qualitative_signals (
            article_id, source, title, published_at, link,
            teams_mentioned, key_absences_impact, fatigue_rotation_risk,
            morale_sentiment_score, tactical_summary, confidence_score
        ) VALUES %s
        ON CONFLICT (article_id) DO NOTHING;
        """
        
        values = [
            (
                s.get('article_id'), s.get('source'), s.get('title'),
                s.get('published_at'), s.get('link'),
                s.get('teams_mentioned', []),
                s.get('key_absences_impact'), s.get('fatigue_rotation_risk'),
                s.get('morale_sentiment_score'), s.get('tactical_summary'),
                s.get('confidence_score')
            )
            for s in signals
        ]
        
        with self.get_cursor(commit=True) as cur:
            extras.execute_values(cur, query, values)
            return len(values)
    
    def insert_prediction(self, prediction_data: Dict) -> Optional[int]:
        """Insert model prediction."""
        query = """
        INSERT INTO model_predictions (
            match_id, model_version, model_type,
            prob_home, prob_draw, prob_away,
            calibration_method, is_calibrated,
            edge_home, edge_draw, edge_away,
            recommended_outcome, recommended_stake, kelly_fraction
        ) VALUES (
            %(match_id)s, %(model_version)s, %(model_type)s,
            %(prob_home)s, %(prob_draw)s, %(prob_away)s,
            %(calibration_method)s, %(is_calibrated)s,
            %(edge_home)s, %(edge_draw)s, %(edge_away)s,
            %(recommended_outcome)s, %(recommended_stake)s, %(kelly_fraction)s
        )
        ON CONFLICT (match_id, model_version, model_type) DO UPDATE SET
            prob_home = EXCLUDED.prob_home,
            prob_draw = EXCLUDED.prob_draw,
            prob_away = EXCLUDED.prob_away,
            predicted_at = NOW()
        RETURNING prediction_id;
        """
        
        with self.get_cursor(commit=True) as cur:
            cur.execute(query, prediction_data)
            result = cur.fetchone()
            return result['prediction_id'] if result else None
    
    def get_recent_matches(self, limit: int = 100) -> List[Dict]:
        """Fetch recent matches for analysis."""
        query = """
        SELECT * FROM matches
        ORDER BY match_date DESC
        LIMIT %s;
        """
        
        with self.get_cursor() as cur:
            cur.execute(query, (limit,))
            return list(cur.fetchall())
    
    def get_upcoming_matches(self, days_ahead: int = 7) -> List[Dict]:
        """Fetch upcoming matches without results."""
        query = """
        SELECT * FROM matches
        WHERE match_date >= CURRENT_DATE
        AND match_date <= CURRENT_DATE + INTERVAL '%s days'
        AND result IS NULL
        ORDER BY match_date ASC;
        """
        
        with self.get_cursor() as cur:
            cur.execute(query, (days_ahead,))
            return list(cur.fetchall())
    
    def get_team_recent_signals(self, team_name: str, days_back: int = 7) -> List[Dict]:
        """Get recent LLM signals mentioning a specific team."""
        query = """
        SELECT * FROM llm_qualitative_signals
        WHERE %s = ANY(teams_mentioned)
        AND published_at >= CURRENT_DATE - INTERVAL '%s days'
        ORDER BY published_at DESC;
        """
        
        with self.get_cursor() as cur:
            cur.execute(query, (team_name, days_back))
            return list(cur.fetchall())
    
    def get_latest_odds_for_match(self, match_id: int) -> Optional[Dict]:
        """Get the most recent odds for a specific match."""
        query = """
        SELECT * FROM odds
        WHERE match_id = %s
        ORDER BY odds_timestamp DESC
        LIMIT 1;
        """
        
        with self.get_cursor() as cur:
            cur.execute(query, (match_id,))
            return cur.fetchone()
    
    def truncate_all_tables(self):
        """WARNING: Delete all data from all tables. Use with caution."""
        logger.warning("Truncating all tables!")
        
        with self.get_cursor(commit=True) as cur:
            cur.execute("TRUNCATE TABLE model_predictions, llm_qualitative_signals, odds, matches RESTART IDENTITY CASCADE;")
        
        logger.info("All tables truncated")
