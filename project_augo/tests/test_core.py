"""
Project Augo - Unit Tests
Windows-compatible test suite for core functionality
"""
import pytest
import numpy as np
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestQualitativeSchema:
    """Test Pydantic schema validation for LLM outputs"""
    
    def test_valid_signal(self):
        from schemas.qualitative import QualitativeSignal
        
        signal = QualitativeSignal(
            article_title="Test Article",
            source="BBC Sport",
            published_date=datetime.now(),
            key_absences_impact=7.5,
            fatigue_rotation_risk=4.0,
            morale_sentiment_score=2.5,
            tactical_summary="Team in good form",
            teams_mentioned=["Arsenal"],
            confidence_score=0.92
        )
        
        assert signal.key_absences_impact == 7.5
        assert signal.morale_sentiment_score == 2.5
    
    def test_bounds_validation_absences(self):
        from schemas.qualitative import QualitativeSignal
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            QualitativeSignal(
                article_title="Test",
                source="BBC",
                published_date=datetime.now(),
                key_absences_impact=15.0,  # Out of bounds (max 10)
                fatigue_rotation_risk=5.0,
                morale_sentiment_score=0.0,
                tactical_summary="Test"
            )
    
    def test_bounds_validation_morale(self):
        from schemas.qualitative import QualitativeSignal
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            QualitativeSignal(
                article_title="Test",
                source="BBC",
                published_date=datetime.now(),
                key_absences_impact=5.0,
                fatigue_rotation_risk=5.0,
                morale_sentiment_score=-10.0,  # Out of bounds (min -5)
                tactical_summary="Test"
            )


class TestFusionEngine:
    """Test probability fusion and Kelly criterion"""
    
    def test_probability_fusion_sum(self):
        from src.ml.fusion import HybridFusionEngine
        
        engine = HybridFusionEngine()
        xgb_probs = np.array([0.55, 0.25, 0.20])
        llm_signals = {
            'absences_impact': 7.0,
            'fatigue_risk': 3.0,
            'morale_score': 2.5
        }
        
        fused = engine.fuse_probabilities(xgb_probs, llm_signals)
        
        # Probabilities must sum to 1.0
        assert abs(fused.sum() - 1.0) < 0.001
    
    def test_kelly_positive_edge(self):
        from src.ml.fusion import HybridFusionEngine
        
        engine = HybridFusionEngine()
        
        # Positive edge scenario
        stake = engine.kelly_criterion(probability=0.55, decimal_odds=2.0)
        
        assert stake > 0  # Should recommend positive stake
    
    def test_kelly_negative_edge(self):
        from src.ml.fusion import HybridFusionEngine
        
        engine = HybridFusionEngine()
        
        # Negative edge scenario (probability too low for odds)
        stake = engine.kelly_criterion(probability=0.40, decimal_odds=2.0)
        
        assert stake == 0  # Should not bet
    
    def test_vig_removal(self):
        from src.ml.fusion import HybridFusionEngine
        
        engine = HybridFusionEngine()
        
        # Typical bookmaker odds with ~5% vig
        fair_probs = engine.remove_vig(2.0, 3.5, 4.0)
        
        # Fair probs should sum to ~1.0 (no vig)
        assert 0.99 <= sum(fair_probs) <= 1.01


class TestBankrollManager:
    """Test bankroll management and risk controls"""
    
    def test_stake_calculation(self):
        from src.ml.fusion import BankrollManager
        
        bm = BankrollManager(initial_bankroll=1000)
        stake = bm.calculate_stake(edge=0.08, odds=2.0, probability=0.55)
        
        assert stake > 0
        assert stake <= 1000 * 0.05  # Max 5% of bankroll
    
    def test_circuit_breaker(self):
        from src.ml.fusion import BankrollManager
        
        bm = BankrollManager(initial_bankroll=1000, drawdown_limit=0.20)
        
        # Simulate 25% drawdown
        bm.peak_bankroll = 1000
        bm.current_bankroll = 750
        
        bm._check_circuit_breaker()
        
        assert bm.circuit_breaker_active is True
        
        # No stakes allowed when circuit breaker active
        stake = bm.calculate_stake(edge=0.10, odds=2.0, probability=0.60)
        assert stake == 0
    
    def test_performance_tracking(self):
        from src.ml.fusion import BankrollManager
        
        bm = BankrollManager(initial_bankroll=1000)
        
        # Record and settle some bets
        bm.record_bet({'home': 'A', 'away': 'B'}, 10, 'H', 2.0)
        bm.settle_bet(0, won=True, return_amount=20)
        
        perf = bm.get_performance_summary()
        
        assert perf['current_bankroll'] == 1010
        assert perf['bets_placed'] == 1


class TestQuantitativeIngestion:
    """Test quantitative data ingestion"""
    
    def test_season_code_generation(self):
        from src.ingestion.quantitative import QuantitativeIngestionEngine
        
        engine = QuantitativeIngestionEngine()
        
        assert engine._get_season_code(2024) == "2425"
        assert engine._get_season_code(2023) == "2324"
        assert engine._get_season_code(2022) == "2223"


class TestMLEngine:
    """Test ML model components"""
    
    def test_feature_engineering_structure(self):
        """Verify feature engineering produces expected columns"""
        from src.ml.xgb_model import EPLMatchPredictor
        import polars as pl
        
        predictor = EPLMatchPredictor()
        
        # Check feature columns are defined
        assert len(predictor.FEATURE_COLUMNS) > 0
        assert 'home_rolling_goals_avg' in predictor.FEATURE_COLUMNS
        assert 'away_rolling_points_avg' in predictor.FEATURE_COLUMNS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
