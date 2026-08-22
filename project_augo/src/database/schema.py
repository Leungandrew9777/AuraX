"""
Database Schema and Connection Management
PostgreSQL + TimescaleDB setup for Project Augo
Windows-optimized with connection pooling
"""
import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import RealDictCursor
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages PostgreSQL/TimescaleDB connections and schema operations.
    
    Tables:
    - matches: Core match data
    - odds: Historical bookmaker odds
    - llm_qualitative_signals: LLM-parsed metrics
    - model_predictions: ML model outputs
    """
    
    # Table creation SQL statements
    SCHEMA_SQL = """
    -- Enable TimescaleDB extension (if not already enabled)
    CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
    
    -- Matches table (hypertable for time-series)
    CREATE TABLE IF NOT EXISTS matches (
        id SERIAL PRIMARY KEY,
        match_date DATE NOT NULL,
        home_team VARCHAR(100) NOT NULL,
        away_team VARCHAR(100) NOT NULL,
        ft_home_goals INTEGER,
        ft_away_goals INTEGER,
        ht_home_goals INTEGER,
        ht_away_goals INTEGER,
        ft_result CHAR(1),  -- H, D, A
        home_shots INTEGER,
        away_shots INTEGER,
        home_shots_on_target INTEGER,
        away_shots_on_target INTEGER,
        home_corners INTEGER,
        away_corners INTEGER,
        home_yellow_cards INTEGER,
        away_yellow_cards INTEGER,
        home_red_cards INTEGER,
        away_red_cards INTEGER,
        division VARCHAR(10),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Create hypertable for matches (time-series optimization)
    SELECT create_hypertable('matches', 'match_date', if_not_exists => TRUE);
    
    -- Indexes for common queries
    CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team, away_team);
    CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date DESC);
    
    -- Odds table
    CREATE TABLE IF NOT EXISTS odds (
        id SERIAL PRIMARY KEY,
        match_id INTEGER REFERENCES matches(id) ON DELETE CASCADE,
        bookmaker VARCHAR(50) DEFAULT 'Bet365',
        odds_home DECIMAL(5,2),
        odds_draw DECIMAL(5,2),
        odds_away DECIMAL(5,2),
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(match_id, bookmaker, recorded_at)
    );
    
    CREATE INDEX IF NOT EXISTS idx_odds_match ON odds(match_id);
    CREATE INDEX IF NOT EXISTS idx_odds_bookmaker ON odds(bookmaker);
    
    -- LLM Qualitative Signals table
    CREATE TABLE IF NOT EXISTS llm_qualitative_signals (
        id SERIAL PRIMARY KEY,
        article_title TEXT NOT NULL,
        source VARCHAR(100),
        published_date TIMESTAMP,
        key_absences_impact DECIMAL(3,1) CHECK (key_absences_impact BETWEEN 0 AND 10),
        fatigue_rotation_risk DECIMAL(3,1) CHECK (fatigue_rotation_risk BETWEEN 0 AND 10),
        morale_sentiment_score DECIMAL(3,1) CHECK (morale_sentiment_score BETWEEN -5 AND 5),
        tactical_summary TEXT,
        teams_mentioned TEXT[],  -- Array of team names
        confidence_score DECIMAL(3,2) CHECK (confidence_score BETWEEN 0 AND 1),
        raw_json JSONB,  -- Full LLM response for debugging
        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_llm_teams ON llm_qualitative_signals USING GIN(teams_mentioned);
    CREATE INDEX IF NOT EXISTS idx_llm_published ON llm_qualitative_signals(published_date DESC);
    
    -- Model Predictions table
    CREATE TABLE IF NOT EXISTS model_predictions (
        id SERIAL PRIMARY KEY,
        match_id INTEGER REFERENCES matches(id),
        model_version VARCHAR(50) NOT NULL,
        prob_home DECIMAL(5,4),
        prob_draw DECIMAL(5,4),
        prob_away DECIMAL(5,4),
        predicted_outcome CHAR(1),  -- H, D, A
        edge_home DECIMAL(7,4),
        edge_draw DECIMAL(7,4),
        edge_away DECIMAL(7,4),
        recommended_stake DECIMAL(5,4),
        kelly_fraction DECIMAL(5,4),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_predictions_match ON model_predictions(match_id);
    CREATE INDEX IF NOT EXISTS idx_predictions_model ON model_predictions(model_version);
    CREATE INDEX IF NOT EXISTS idx_predictions_created ON model_predictions(created_at DESC);
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "project_augo",
        user: str = "postgres",
        password: str = "your_password_here",
        min_connections: int = 2,
        max_connections: int = 10
    ):
        """
        Initialize database manager with connection pool.
        
        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            database: Database name
            user: Database user
            password: Database password
            min_connections: Minimum pool size
            max_connections: Maximum pool size
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        
        self.connection_pool: Optional[pool.SimpleConnectionPool] = None
        self._initialize_pool(min_connections, max_connections)
    
    def _initialize_pool(self, min_conn: int, max_conn: int):
        """Initialize the connection pool"""
        try:
            self.connection_pool = pool.SimpleConnectionPool(
                minconn=min_conn,
                maxconn=max_conn,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            logger.info(f"✓ Database connection pool initialized ({min_conn}-{max_conn} connections)")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        if self.connection_pool is None:
            raise RuntimeError("Connection pool not initialized")
        
        conn = self.connection_pool.getconn()
        try:
            yield conn
        finally:
            self.connection_pool.putconn(conn)
    
    @contextmanager
    def get_cursor(self, dict_cursor: bool = True):
        """Context manager for database cursors"""
        with self.get_connection() as conn:
            cursor_type = RealDictCursor if dict_cursor else psycopg2.Cursor
            cursor = conn.cursor(cursor_factory=cursor_type)
            try:
                yield cursor
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Database transaction failed: {e}")
                raise
            finally:
                cursor.close()
    
    def initialize_schema(self):
        """Create all tables and indexes (idempotent)"""
        logger.info("Initializing database schema...")
        
        try:
            with self.get_cursor() as cursor:
                # Execute each statement separately
                statements = [s.strip() for s in self.SCHEMA_SQL.split(';') if s.strip()]
                
                for stmt in statements:
                    if stmt:
                        cursor.execute(stmt)
                
                logger.info("✓ Database schema initialized successfully")
                
        except Exception as e:
            logger.error(f"Schema initialization failed: {e}")
            raise
    
    # ==================== MATCHES ====================
    
    def insert_matches(self, matches_data: List[Dict[str, Any]]) -> int:
        """
        Insert multiple matches (upsert on conflict).
        
        Returns number of inserted/updated rows.
        """
        if not matches_data:
            return 0
        
        query = """
        INSERT INTO matches (
            match_date, home_team, away_team,
            ft_home_goals, ft_away_goals, ht_home_goals, ht_away_goals, ft_result,
            home_shots, away_shots, home_shots_on_target, away_shots_on_target,
            home_corners, away_corners,
            home_yellow_cards, away_yellow_cards, home_red_cards, away_red_cards,
            division
        ) VALUES (
            %(match_date)s, %(home_team)s, %(away_team)s,
            %(ft_home_goals)s, %(ft_away_goals)s, %(ht_home_goals)s, %(ht_away_goals)s, %(ft_result)s,
            %(home_shots)s, %(away_shots)s, %(home_shots_on_target)s, %(away_shots_on_target)s,
            %(home_corners)s, %(away_corners)s,
            %(home_yellow_cards)s, %(away_yellow_cards)s, %(home_red_cards)s, %(away_red_cards)s,
            %(division)s
        )
        ON CONFLICT DO NOTHING
        RETURNING id
        """
        
        inserted = 0
        with self.get_cursor() as cursor:
            for match in matches_data:
                try:
                    cursor.execute(query, match)
                    inserted += 1
                except Exception as e:
                    logger.warning(f"Failed to insert match: {e}")
        
        logger.info(f"✓ Inserted {inserted} matches")
        return inserted
    
    def get_matches(
        self,
        limit: int = 100,
        offset: int = 0,
        team: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch matches with optional filtering"""
        if team:
            query = """
            SELECT * FROM matches
            WHERE home_team = %(team)s OR away_team = %(team)s
            ORDER BY match_date DESC
            LIMIT %(limit)s OFFSET %(offset)s
            """
        else:
            query = """
            SELECT * FROM matches
            ORDER BY match_date DESC
            LIMIT %(limit)s OFFSET %(offset)s
            """
        
        with self.get_cursor(dict_cursor=True) as cursor:
            cursor.execute(query, {'limit': limit, 'offset': offset, 'team': team})
            return cursor.fetchall()
    
    # ==================== ODDS ====================
    
    def insert_odds(self, match_id: int, odds_data: Dict[str, Any]) -> int:
        """Insert odds for a match"""
        query = """
        INSERT INTO odds (match_id, odds_home, odds_draw, odds_away, bookmaker)
        VALUES (%(match_id)s, %(odds_home)s, %(odds_draw)s, %(odds_away)s, %(bookmaker)s)
        ON CONFLICT (match_id, bookmaker, recorded_at) DO UPDATE SET
            odds_home = EXCLUDED.odds_home,
            odds_draw = EXCLUDED.odds_draw,
            odds_away = EXCLUDED.odds_away,
            recorded_at = CURRENT_TIMESTAMP
        RETURNING id
        """
        
        odds_data['match_id'] = match_id
        
        with self.get_cursor() as cursor:
            cursor.execute(query, odds_data)
            result = cursor.fetchone()
            return result['id'] if result else 0
    
    def get_odds_for_match(self, match_id: int) -> Optional[Dict[str, Any]]:
        """Get latest odds for a specific match"""
        query = """
        SELECT * FROM odds
        WHERE match_id = %(match_id)s
        ORDER BY recorded_at DESC
        LIMIT 1
        """
        
        with self.get_cursor(dict_cursor=True) as cursor:
            cursor.execute(query, {'match_id': match_id})
            return cursor.fetchone()
    
    # ==================== LLM SIGNALS ====================
    
    def insert_llm_signal(self, signal_data: Dict[str, Any]) -> int:
        """Insert a qualitative signal from LLM parsing"""
        query = """
        INSERT INTO llm_qualitative_signals (
            article_title, source, published_date,
            key_absences_impact, fatigue_rotation_risk, morale_sentiment_score,
            tactical_summary, teams_mentioned, confidence_score, raw_json
        ) VALUES (
            %(article_title)s, %(source)s, %(published_date)s,
            %(key_absences_impact)s, %(fatigue_rotation_risk)s, %(morale_sentiment_score)s,
            %(tactical_summary)s, %(teams_mentioned)s, %(confidence_score)s, %(raw_json)s
        )
        RETURNING id
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(query, signal_data)
            result = cursor.fetchone()
            return result['id'] if result else 0
    
    def get_signals_for_team(
        self,
        team_name: str,
        days_back: int = 7
    ) -> List[Dict[str, Any]]:
        """Get recent LLM signals mentioning a specific team"""
        query = """
        SELECT * FROM llm_qualitative_signals
        WHERE %(team)s = ANY(teams_mentioned)
        AND published_date >= NOW() - INTERVAL '%(days)s days'
        ORDER BY published_date DESC
        """
        
        with self.get_cursor(dict_cursor=True) as cursor:
            cursor.execute(query, {'team': team_name, 'days': days_back})
            return cursor.fetchall()
    
    def get_average_signals_for_match(
        self,
        home_team: str,
        away_team: str,
        days_back: int = 7
    ) -> Dict[str, float]:
        """
        Get averaged qualitative signals for both teams in a match.
        Returns aggregated metrics for fusion with ML model.
        """
        query = """
        SELECT 
            AVG(key_absences_impact) as avg_absences,
            AVG(fatigue_rotation_risk) as avg_fatigue,
            AVG(morale_sentiment_score) as avg_morale
        FROM llm_qualitative_signals
        WHERE (%(home)s = ANY(teams_mentioned) OR %(away)s = ANY(teams_mentioned))
        AND published_date >= NOW() - INTERVAL '%(days)s days'
        """
        
        with self.get_cursor(dict_cursor=True) as cursor:
            cursor.execute(query, {'home': home_team, 'away': away_team, 'days': days_back})
            result = cursor.fetchone()
            
            return {
                'avg_absences_impact': float(result['avg_absences']) if result['avg_absences'] else 5.0,
                'avg_fatigue_risk': float(result['avg_fatigue']) if result['avg_fatigue'] else 5.0,
                'avg_morale_score': float(result['avg_morale']) if result['avg_morale'] else 0.0,
            }
    
    # ==================== PREDICTIONS ====================
    
    def save_prediction(self, prediction_data: Dict[str, Any]) -> int:
        """Save model prediction to database"""
        query = """
        INSERT INTO model_predictions (
            match_id, model_version,
            prob_home, prob_draw, prob_away, predicted_outcome,
            edge_home, edge_draw, edge_away,
            recommended_stake, kelly_fraction
        ) VALUES (
            %(match_id)s, %(model_version)s,
            %(prob_home)s, %(prob_draw)s, %(prob_away)s, %(predicted_outcome)s,
            %(edge_home)s, %(edge_draw)s, %(edge_away)s,
            %(recommended_stake)s, %(kelly_fraction)s
        )
        RETURNING id
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(query, prediction_data)
            result = cursor.fetchone()
            return result['id'] if result else 0
    
    def get_recent_predictions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get most recent predictions"""
        query = """
        SELECT p.*, m.home_team, m.away_team, m.match_date
        FROM model_predictions p
        JOIN matches m ON p.match_id = m.id
        ORDER BY p.created_at DESC
        LIMIT %(limit)s
        """
        
        with self.get_cursor(dict_cursor=True) as cursor:
            cursor.execute(query, {'limit': limit})
            return cursor.fetchall()
    
    def close(self):
        """Close all connections in the pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("✓ Database connections closed")


# Example usage
if __name__ == "__main__":
    # Configure these for your Windows PostgreSQL installation
    db = DatabaseManager(
        host="localhost",
        port=5432,
        database="project_augo",
        user="postgres",
        password="your_password_here"
    )
    
    # Initialize schema (creates tables)
    db.initialize_schema()
    
    print("\n✓ Database ready!")
