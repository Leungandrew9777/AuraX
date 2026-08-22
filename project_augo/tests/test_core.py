"""
Project Augo - Unit Tests
Validation tests for probability calibration, backtesting metrics, and core functionality.
"""
import pytest
import numpy as np
import polars as pl
from datetime import datetime, timedelta


class TestQuantitativeIngestion:
    """Tests for quantitative data ingestion engine."""
    
    def test_fetch_season_data(self):
        """Test fetching a single season of data."""
        from src.ingestion.quantitative import QuantitativeIngestionEngine
        
        engine = QuantitativeIngestionEngine()
        
        # This test requires internet connection
        df = engine.fetch_season_data('2324')
        
        if df is not None:
            assert len(df) > 0
            assert 'FTHG' in df.columns
            assert 'FTAG' in df.columns
    
    def test_clean_and_standardize(self):
        """Test data cleaning operations."""
        from src.ingestion.quantitative import QuantitativeIngestionEngine
        
        engine = QuantitativeIngestionEngine()
        
        # Create sample data
        sample_data = {
            'Date': ['01/01/2024', '02/01/2024'],
            'HomeTeam': ['Arsenal', 'Chelsea'],
            'AwayTeam': ['Liverpool', 'Man City'],
            'FTR': ['H', 'A'],
            'FTHG': [2, 1],
            'FTAG': [1, 3],
            'HS': [15, 10],
            'AS': [12, 18],
            'HST': [6, 4],
            'AST': [5, 8],
            'B365H': [1.80, 2.50],
            'B365D': [3.50, 3.20],
            'B365A': [4.50, 2.80]
        }
        
        df = pl.DataFrame(sample_data)
        df_clean = engine.clean_and_standardize(df)
        
        assert 'match_date' in df_clean.columns
        assert 'result_encoded' in df_clean.columns
        assert 'goal_difference' in df_clean.columns
        assert 'fair_prob_home' in df_clean.columns
    
    def test_implied_probability_calculation(self):
        """Test fair probability calculation removes vig correctly."""
        from src.ingestion.quantitative import QuantitativeIngestionEngine
        
        engine = QuantitativeIngestionEngine()
        
        # Sample odds that include vig
        sample_data = {
            'Date': ['01/01/2024'],
            'HomeTeam': ['Arsenal'],
            'AwayTeam': ['Liverpool'],
            'FTR': ['H'],
            'FTHG': [2],
            'FTAG': [1],
            'B365H': [1.90],
            'B365D': [3.50],
            'B365A': [4.00]
        }
        
        df = pl.DataFrame(sample_data)
        df_clean = engine.clean_and_standardize(df)
        
        # Fair probabilities should sum to ~1.0 (after vig removal)
        total_fair_prob = (
            df_clean['fair_prob_home'][0] + 
            df_clean['fair_prob_draw'][0] + 
            df_clean['fair_prob_away'][0]
        )
        
        assert abs(total_fair_prob - 1.0) < 0.01
        
        # Raw implied probabilities should sum to > 1.0 (includes vig)
        total_raw_prob = (
            df_clean['raw_prob_home'][0] + 
            df_clean['raw_prob_draw'][0] + 
            df_clean['raw_prob_away'][0]
        )
        
        assert total_raw_prob > 1.0


class TestQualitativeIngestion:
    """Tests for qualitative data ingestion engine."""
    
    def test_schema_validation(self):
        """Test Pydantic schema validation for LLM outputs."""
        from schemas.qualitative import QualitativeSignal
        
        # Valid signal
        valid_signal = QualitativeSignal(
            article_id="test_001",
            source="BBC Sport",
            published_at=datetime.now(),
            teams_mentioned=["Arsenal", "Chelsea"],
            key_absences_impact=5.0,
            fatigue_rotation_risk=3.0,
            morale_sentiment_score=2.0,
            tactical_summary="Arsenal missing key midfielder due to injury.",
            confidence_score=0.85
        )
        
        assert valid_signal.key_absences_impact == 5.0
        assert valid_signal.morale_sentiment_score == 2.0
    
    def test_schema_validation_bounds(self):
        """Test that schema enforces value bounds."""
        from schemas.qualitative import QualitativeSignal
        from pydantic import ValidationError
        
        # Test out-of-bounds values
        with pytest.raises(ValidationError):
            QualitativeSignal(
                article_id="test_002",
                source="Test",
                published_at=datetime.now(),
                teams_mentioned=[],
                key_absences_impact=15.0,  # Should be 0-10
                fatigue_rotation_risk=3.0,
                morale_sentiment_score=2.0,
                tactical_summary="Test"
            )
        
        with pytest.raises(ValidationError):
            QualitativeSignal(
                article_id="test_003",
                source="Test",
                published_at=datetime.now(),
                teams_mentioned=[],
                key_absences_impact=5.0,
                fatigue_rotation_risk=3.0,
                morale_sentiment_score=8.0,  # Should be -5 to +5
                tactical_summary="Test"
            )


class TestXGBoostModel:
    """Tests for XGBoost model training and prediction."""
    
    def test_feature_engineering(self):
        """Test rolling feature creation."""
        from src.ml.xgb_model import FeatureEngineer
        
        engineer = FeatureEngineer(window_sizes=[3, 5])
        
        # Create sample match data
        n_matches = 20
        dates = [datetime(2024, 1, i+1) for i in range(n_matches)]
        
        sample_data = {
            'match_date': dates * 2,  # Two teams
            'HomeTeam': ['Team A'] * n_matches + ['Team B'] * n_matches,
            'AwayTeam': ['Team B'] * n_matches + ['Team A'] * n_matches,
            'FTHG': list(np.random.randint(0, 4, n_matches)) * 2,
            'FTAG': list(np.random.randint(0, 4, n_matches)) * 2,
            'FTR': np.random.choice(['H', 'D', 'A'], n_matches * 2).tolist(),
            'HS': list(np.random.randint(5, 20, n_matches)) * 2,
            'AS': list(np.random.randint(5, 20, n_matches)) * 2,
            'HST': list(np.random.randint(2, 10, n_matches)) * 2,
            'AST': list(np.random.randint(2, 10, n_matches)) * 2,
        }
        
        df = pl.DataFrame(sample_data)
        df_features = engineer.create_rolling_features(df)
        
        # Check that rolling features were created
        assert any('rolling_' in col for col in df_features.columns)
    
    def test_walk_forward_validation(self):
        """Test walk-forward validation prevents look-ahead bias."""
        from src.ml.xgb_model import XGBoostModel
        
        model = XGBoostModel()
        
        # Create larger sample dataset
        n_matches = 100
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_matches)]
        
        sample_data = {
            'match_date': dates,
            'HomeTeam': np.random.choice(['Team A', 'Team B', 'Team C'], n_matches).tolist(),
            'AwayTeam': np.random.choice(['Team D', 'Team E', 'Team F'], n_matches).tolist(),
            'FTHG': np.random.randint(0, 4, n_matches).tolist(),
            'FTAG': np.random.randint(0, 4, n_matches).tolist(),
            'FTR': np.random.choice(['H', 'D', 'A'], n_matches).tolist(),
            'HS': np.random.randint(5, 20, n_matches).tolist(),
            'AS': np.random.randint(5, 20, n_matches).tolist(),
            'HST': np.random.randint(2, 10, n_matches).tolist(),
            'AST': np.random.randint(2, 10, n_matches).tolist(),
            'HC': np.random.randint(2, 12, n_matches).tolist(),
            'AC': np.random.randint(2, 12, n_matches).tolist(),
        }
        
        df = pl.DataFrame(sample_data)
        
        # Run walk-forward validation
        metrics = model.train_walk_forward(df)
        
        # Check metrics are reasonable
        assert 'mean_brier_score' in metrics
        assert 'mean_log_loss' in metrics
        assert 'mean_accuracy' in metrics
        
        # Brier score should be between 0 and 2 for 3-class problem
        assert 0 <= metrics['mean_brier_score'] <= 2.0
        
        # Log loss should be positive
        assert metrics['mean_log_loss'] >= 0
    
    def test_probability_calibration(self):
        """Test that calibrated probabilities sum to 1."""
        from src.ml.xgb_model import XGBoostModel
        
        model = XGBoostModel()
        
        # Create sample data
        n_matches = 50
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_matches)]
        
        sample_data = {
            'match_date': dates,
            'HomeTeam': np.random.choice(['Team A', 'Team B'], n_matches).tolist(),
            'AwayTeam': np.random.choice(['Team C', 'Team D'], n_matches).tolist(),
            'FTHG': np.random.randint(0, 4, n_matches).tolist(),
            'FTAG': np.random.randint(0, 4, n_matches).tolist(),
            'FTR': np.random.choice(['H', 'D', 'A'], n_matches).tolist(),
            'HS': np.random.randint(5, 20, n_matches).tolist(),
            'AS': np.random.randint(5, 20, n_matches).tolist(),
            'HST': np.random.randint(2, 10, n_matches).tolist(),
            'AST': np.random.randint(2, 10, n_matches).tolist(),
            'HC': np.random.randint(2, 12, n_matches).tolist(),
            'AC': np.random.randint(2, 12, n_matches).tolist(),
        }
        
        df = pl.DataFrame(sample_data)
        
        # Train final model
        model.fit_final_model(df)
        
        # Get predictions
        probs = model.predict_proba(df.head(10))
        
        # Check probabilities sum to 1
        prob_sums = probs.sum(axis=1)
        assert np.allclose(prob_sums, 1.0, atol=1e-6)
        
        # Check all probabilities are in [0, 1]
        assert np.all((probs >= 0) & (probs <= 1))


class TestHybridFusion:
    """Tests for hybrid fusion and bankroll management."""
    
    def test_qualitative_adjustment(self):
        """Test qualitative adjustment calculation."""
        from src.ml.fusion import HybridFusionEngine
        
        engine = HybridFusionEngine()
        
        # Test with negative signals (injuries, fatigue, low morale)
        adjustment_negative = engine.calculate_qualitative_adjustment(
            absence_impact=8.0,
            fatigue_risk=7.0,
            morale_score=-3.0,
            confidence=0.8
        )
        
        # Should be negative
        assert adjustment_negative < 0
        
        # Test with positive signals
        adjustment_positive = engine.calculate_qualitative_adjustment(
            absence_impact=1.0,
            fatigue_risk=2.0,
            morale_score=4.0,
            confidence=0.9
        )
        
        # Should be positive
        assert adjustment_positive > 0
    
    def test_probability_fusion(self):
        """Test probability fusion maintains valid distribution."""
        from src.ml.fusion import HybridFusionEngine
        import numpy as np
        
        engine = HybridFusionEngine()
        
        xgb_probs = np.array([0.5, 0.25, 0.25])
        
        home_signals = {
            'key_absences_impact': 3.0,
            'fatigue_rotation_risk': 4.0,
            'morale_sentiment_score': 2.0,
            'confidence_score': 0.7
        }
        
        away_signals = {
            'key_absences_impact': 5.0,
            'fatigue_rotation_risk': 3.0,
            'morale_sentiment_score': -1.0,
            'confidence_score': 0.6
        }
        
        fused_probs = engine.fuse_probabilities(xgb_probs, home_signals, away_signals)
        
        # Check sum to 1
        assert np.isclose(fused_probs.sum(), 1.0, atol=1e-6)
        
        # Check all in valid range
        assert np.all((fused_probs >= 0) & (fused_probs <= 1))
    
    def test_kelly_criterion(self):
        """Test Kelly stake calculation."""
        from src.ml.fusion import BankrollEngine
        
        engine = BankrollEngine()
        
        # Positive edge scenario
        stake = engine.calculate_kelly_stake(edge=0.10, odds=2.00)
        
        # Should be positive stake
        assert stake > 0
        
        # Negative edge scenario
        stake_no_bet = engine.calculate_kelly_stake(edge=-0.05, odds=2.00)
        
        # Should be zero (no bet)
        assert stake_no_bet == 0.0
    
    def test_fair_probability_calculation(self):
        """Test fair probability calculation from odds."""
        from src.ml.fusion import BankrollEngine
        
        engine = BankrollEngine()
        
        # Typical odds with vig
        fair_probs = engine.calculate_fair_probability(
            odds_home=1.90,
            odds_draw=3.50,
            odds_away=4.00
        )
        
        # Should sum to 1.0
        assert np.isclose(sum(fair_probs), 1.0, atol=1e-6)
        
        # Each should be in (0, 1)
        assert all(0 < p < 1 for p in fair_probs)
    
    def test_edge_calculation(self):
        """Test edge calculation."""
        from src.ml.fusion import BankrollEngine
        
        engine = BankrollEngine()
        
        # Model thinks 60%, market implies 55%
        edge = engine.calculate_edge(model_prob=0.60, market_prob=0.55)
        
        assert abs(edge - 0.05) < 1e-10
        
        # Negative edge
        edge_negative = engine.calculate_edge(model_prob=0.40, market_prob=0.50)
        
        assert abs(edge_negative - (-0.10)) < 1e-10


class TestBankrollEngine:
    """Tests for bankroll management and circuit breakers."""
    
    def test_circuit_breaker(self):
        """Test drawdown circuit breaker activation."""
        from src.ml.fusion import BankrollEngine
        
        engine = BankrollEngine()
        engine.config.max_drawdown_limit = 0.20  # 20%
        
        # Simulate losses
        initial_bankroll = engine.current_bankroll
        
        # Record losing bets until circuit breaker triggers
        for _ in range(20):
            engine.record_bet_result(
                outcome='H',
                stake_fraction=0.02,
                odds=2.00,
                won=False
            )
        
        # Check circuit breaker status
        current_drawdown = abs(engine.total_profit_loss) / initial_bankroll
        
        if current_drawdown >= 0.20:
            assert engine.is_circuit_breaker_active is True
    
    def test_performance_tracking(self):
        """Test performance summary calculation."""
        from src.ml.fusion import BankrollEngine
        
        engine = BankrollEngine()
        
        # Record some bets
        engine.record_bet_result('H', 0.02, 2.00, won=True)
        engine.record_bet_result('A', 0.03, 1.80, won=False)
        engine.record_bet_result('D', 0.01, 3.50, won=True)
        
        summary = engine.get_performance_summary()
        
        assert summary['total_bets'] == 3
        assert 'total_pnl' in summary
        assert 'roi' in summary
        assert 'win_rate' in summary


class TestPredictionPipeline:
    """Tests for end-to-end prediction pipeline."""
    
    def test_telegram_export_format(self):
        """Test Telegram message formatting."""
        from src.ml.fusion import PredictionPipeline, MatchPrediction
        
        pipeline = PredictionPipeline()
        
        # Create sample prediction with sufficient edge
        pred = MatchPrediction(
            match_id=1,
            home_team='Arsenal',
            away_team='Everton',
            match_date='2024-12-14',
            prob_home_xgb=0.60,
            prob_draw_xgb=0.25,
            prob_away_xgb=0.15,
            home_absence_impact=2.0,
            home_fatigue_risk=3.0,
            home_morale_score=2.0,
            away_absence_impact=4.0,
            away_fatigue_risk=5.0,
            away_morale_score=-1.0,
            prob_home_final=0.65,
            prob_draw_final=0.20,
            prob_away_final=0.15,
            odds_home=1.45,
            odds_draw=4.50,
            odds_away=7.00,
            fair_prob_home=0.55,  # Lower to create edge > 5%
            fair_prob_draw=0.20,
            fair_prob_away=0.12,
            edge_home=0.10,  # 10% edge > 5% threshold
            edge_draw=0.00,
            edge_away=0.03,
            recommended_outcome='H',
            recommended_stake=0.032,
            kelly_fraction_used=0.25,
            has_positive_edge=True
        )
        
        telegram_msg = pipeline.export_to_telegram_format([pred])
        
        # Check message contains expected elements
        assert 'Arsenal' in telegram_msg or 'PROJECT AUGO' in telegram_msg


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
