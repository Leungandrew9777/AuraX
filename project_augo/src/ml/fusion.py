"""
Hybrid Fusion Engine and Bankroll Management
Combines XGBoost probabilities with LLM qualitative signals
Implements Kelly Criterion staking with risk controls
Windows-optimized implementation
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MatchPrediction:
    """Container for complete match prediction with edge calculation"""
    home_team: str
    away_team: str
    match_date: datetime
    
    # Raw XGBoost probabilities
    prob_xgb_home: float
    prob_xgb_draw: float
    prob_xgb_away: float
    
    # LLM qualitative adjustments
    llm_absences_impact: float  # 0-10
    llm_fatigue_risk: float     # 0-10
    llm_morale_score: float     # -5 to +5
    
    # Final fused probabilities
    prob_final_home: float
    prob_final_draw: float
    prob_final_away: float
    
    # Market odds
    odds_home: float
    odds_draw: float
    odds_away: float
    
    # Fair probabilities (vig removed)
    fair_prob_home: float
    fair_prob_draw: float
    fair_prob_away: float
    
    # Edge calculations
    edge_home: float
    edge_draw: float
    edge_away: float
    
    # Recommended stake (Kelly)
    kelly_stake_home: float
    kelly_stake_draw: float
    kelly_stake_away: float
    
    # Best bet recommendation
    recommended_outcome: str  # 'H', 'D', 'A', or 'NONE'
    recommended_stake: float
    confidence_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling datetime serialization"""
        d = asdict(self)
        d['match_date'] = self.match_date.isoformat()
        return d


class HybridFusionEngine:
    """
    Combines quantitative ML probabilities with qualitative LLM signals.
    """
    
    def __init__(
        self,
        meta_learner_alpha: float = 0.1,
        xgb_weight: float = 0.7,
        llm_weight: float = 0.3
    ):
        self.xgb_weight = xgb_weight
        self.llm_weight = llm_weight
        self.meta_learner_alpha = meta_learner_alpha
        
        self._meta_coefficients = {
            'absences': -0.02,
            'fatigue': -0.015,
            'morale': 0.04,
        }
    
    def fuse_probabilities(
        self,
        xgb_probs: np.ndarray,
        llm_signals: Dict[str, float],
        team_context: Optional[Dict[str, str]] = None
    ) -> np.ndarray:
        """Fuse XGBoost probabilities with LLM qualitative signals."""
        p_home, p_draw, p_away = xgb_probs
        
        absences = llm_signals.get('absences_impact', 5.0)
        fatigue = llm_signals.get('fatigue_risk', 5.0)
        morale = llm_signals.get('morale_score', 0.0)
        
        absences_adjustment = self._meta_coefficients['absences'] * (absences - 5.0)
        fatigue_adjustment = self._meta_coefficients['fatigue'] * (fatigue - 5.0)
        morale_adjustment = self._meta_coefficients['morale'] * morale
        
        p_home_adjusted = p_home + absences_adjustment + morale_adjustment
        p_away_adjusted = p_away + fatigue_adjustment * 0.5
        p_draw_adjusted = p_draw - (absences_adjustment + morale_adjustment) * 0.3
        
        total = p_home_adjusted + p_draw_adjusted + p_away_adjusted
        
        if total <= 0 or total > 2:
            logger.warning(f"Probability normalization issue: {total}")
            return xgb_probs
        
        fused_probs = np.array([
            p_home_adjusted / total,
            p_draw_adjusted / total,
            p_away_adjusted / total
        ])
        
        fused_probs = np.clip(fused_probs, 0.001, 0.999)
        fused_probs = fused_probs / fused_probs.sum()
        
        return fused_probs
    
    def create_match_prediction(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime,
        xgb_probs: np.ndarray,
        llm_signals: Dict[str, float],
        market_odds: Dict[str, float]
    ) -> MatchPrediction:
        """Create complete match prediction with edge and stake calculations."""
        fused_probs = self.fuse_probabilities(xgb_probs, llm_signals, {
            'home_team': home_team,
            'away_team': away_team
        })
        
        fair_probs = self.remove_vig(
            market_odds.get('home', 2.0),
            market_odds.get('draw', 3.5),
            market_odds.get('away', 3.0)
        )
        
        edge_home = fused_probs[0] - fair_probs[0]
        edge_draw = fused_probs[1] - fair_probs[1]
        edge_away = fused_probs[2] - fair_probs[2]
        
        kelly_home = self.kelly_criterion(fused_probs[0], market_odds.get('home', 2.0))
        kelly_draw = self.kelly_criterion(fused_probs[1], market_odds.get('draw', 3.5))
        kelly_away = self.kelly_criterion(fused_probs[2], market_odds.get('away', 3.0))
        
        edges = {'H': edge_home, 'D': edge_draw, 'A': edge_away}
        best_outcome = max(edges, key=edges.get)
        best_edge = edges[best_outcome]
        
        if best_edge > 0.02:
            recommended_outcome = best_outcome
            stakes = {'H': kelly_home, 'D': kelly_draw, 'A': kelly_away}
            recommended_stake = stakes[best_outcome]
            confidence = self._assess_confidence(best_edge, fused_probs)
        else:
            recommended_outcome = 'NONE'
            recommended_stake = 0.0
            confidence = 'LOW'
        
        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            prob_xgb_home=float(xgb_probs[0]),
            prob_xgb_draw=float(xgb_probs[1]),
            prob_xgb_away=float(xgb_probs[2]),
            llm_absences_impact=llm_signals.get('absences_impact', 5.0),
            llm_fatigue_risk=llm_signals.get('fatigue_risk', 5.0),
            llm_morale_score=llm_signals.get('morale_score', 0.0),
            prob_final_home=float(fused_probs[0]),
            prob_final_draw=float(fused_probs[1]),
            prob_final_away=float(fused_probs[2]),
            odds_home=market_odds.get('home', 0),
            odds_draw=market_odds.get('draw', 0),
            odds_away=market_odds.get('away', 0),
            fair_prob_home=float(fair_probs[0]),
            fair_prob_draw=float(fair_probs[1]),
            fair_prob_away=float(fair_probs[2]),
            edge_home=float(edge_home),
            edge_draw=float(edge_draw),
            edge_away=float(edge_away),
            kelly_stake_home=float(kelly_home),
            kelly_stake_draw=float(kelly_draw),
            kelly_stake_away=float(kelly_away),
            recommended_outcome=recommended_outcome,
            recommended_stake=float(recommended_stake),
            confidence_level=confidence
        )
    
    def remove_vig(
        self,
        odds_home: float,
        odds_draw: float,
        odds_away: float,
        method: str = 'shin'
    ) -> Tuple[float, float, float]:
        """Remove bookmaker margin (vig) from odds to get fair probabilities."""
        implied_home = 1 / odds_home
        implied_draw = 1 / odds_draw
        implied_away = 1 / odds_away
        
        total_implied = implied_home + implied_draw + implied_away
        
        if method == 'proportional':
            fair_home = implied_home / total_implied
            fair_draw = implied_draw / total_implied
            fair_away = implied_away / total_implied
        elif method == 'shin':
            fair_home, fair_draw, fair_away = self._shin_method(
                implied_home, implied_draw, implied_away, total_implied
            )
        else:
            power = 1 / total_implied
            sum_power = sum(implied ** power for implied in [implied_home, implied_draw, implied_away])
            fair_home = (implied_home ** power) / sum_power
            fair_draw = (implied_draw ** power) / sum_power
            fair_away = (implied_away ** power) / sum_power
        
        return fair_home, fair_draw, fair_away
    
    def _shin_method(
        self,
        p1: float, p2: float, p3: float,
        total: float,
        iterations: int = 100
    ) -> Tuple[float, float, float]:
        """Shin's method for removing favorite-longshot bias."""
        q1 = p1 / total
        q2 = p2 / total
        q3 = p3 / total
        
        Z = total - 1
        
        for _ in range(iterations):
            denom1 = (1 - Z) * (1 - Z + Z * np.sqrt(1 / q1)) if q1 > 0 else 1
            q1_new = (1 - Z) * q1 / denom1
            
            denom2 = (1 - Z) * (1 - Z + Z * np.sqrt(1 / q2)) if q2 > 0 else 1
            q2_new = (1 - Z) * q2 / denom2
            
            denom3 = (1 - Z) * (1 - Z + Z * np.sqrt(1 / q3)) if q3 > 0 else 1
            q3_new = (1 - Z) * q3 / denom3
            
            sum_q = q1_new + q2_new + q3_new
            if sum_q > 0:
                q1, q2, q3 = q1_new / sum_q, q2_new / sum_q, q3_new / sum_q
        
        return q1, q2, q3
    
    def kelly_criterion(
        self,
        probability: float,
        decimal_odds: float,
        kelly_fraction: float = 0.25
    ) -> float:
        """Calculate optimal stake using Kelly Criterion."""
        if decimal_odds <= 1.0:
            return 0.0
        
        b = decimal_odds - 1
        p = probability
        q = 1 - p
        
        kelly_full = (p * b - q) / b
        kelly_stake = kelly_full * kelly_fraction
        
        return max(0.0, kelly_stake)
    
    def _assess_confidence(self, edge: float, probs: np.ndarray) -> str:
        """Assess confidence level based on edge and probability concentration."""
        max_prob = probs.max()
        
        if edge > 0.10 and max_prob > 0.60:
            return 'HIGH'
        elif edge > 0.05 and max_prob > 0.45:
            return 'MEDIUM'
        else:
            return 'LOW'


class BankrollManager:
    """Manages betting bankroll with strict risk controls."""
    
    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        max_stake_percent: float = 0.05,
        max_exposure: float = 0.20,
        drawdown_limit: float = 0.20,
        kelly_fraction: float = 0.25
    ):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.max_stake_percent = max_stake_percent
        self.max_exposure = max_exposure
        self.drawdown_limit = drawdown_limit
        self.kelly_fraction = kelly_fraction
        
        self.bet_history: List[Dict[str, Any]] = []
        self.total_staked = 0.0
        self.total_returned = 0.0
        
        self.circuit_breaker_active = False
        self.peak_bankroll = initial_bankroll
    
    def calculate_stake(
        self,
        edge: float,
        odds: float,
        probability: float
    ) -> float:
        """Calculate stake with all risk controls applied."""
        if self.circuit_breaker_active:
            logger.warning("Circuit breaker active - no bets allowed")
            return 0.0
        
        base_stake = self._kelly_with_edge(edge, odds, probability)
        
        max_stake = self.current_bankroll * self.max_stake_percent
        stake = min(base_stake, max_stake)
        
        current_exposure = self.get_current_exposure()
        if current_exposure + stake > self.current_bankroll * self.max_exposure:
            stake = max(0, self.current_bankroll * self.max_exposure - current_exposure)
        
        return round(stake, 2)
    
    def _kelly_with_edge(self, edge: float, odds: float, probability: float) -> float:
        """Kelly calculation incorporating edge directly."""
        if edge <= 0:
            return 0.0
        
        base_kelly = edge * self.kelly_fraction
        confidence_multiplier = min(1.0, probability * 2)
        stake = self.current_bankroll * base_kelly * confidence_multiplier
        
        return stake
    
    def get_current_exposure(self) -> float:
        """Calculate total current exposure from pending bets."""
        pending = [b for b in self.bet_history if b.get('status') == 'pending']
        return sum(b.get('stake', 0) for b in pending)
    
    def record_bet(
        self,
        match_info: Dict[str, Any],
        stake: float,
        outcome: str,
        odds: float
    ):
        """Record a placed bet for tracking."""
        self.bet_history.append({
            'timestamp': datetime.now(),
            'match': match_info,
            'stake': stake,
            'outcome_type': outcome,
            'odds': odds,
            'status': 'pending'
        })
        self.total_staked += stake
    
    def settle_bet(self, bet_index: int, won: bool, return_amount: float = 0.0):
        """Settle a bet and update bankroll."""
        if bet_index >= len(self.bet_history):
            raise ValueError("Invalid bet index")
        
        bet = self.bet_history[bet_index]
        bet['status'] = 'won' if won else 'lost'
        bet['return'] = return_amount
        
        if won:
            self.current_bankroll += return_amount
            self.total_returned += return_amount
            
            if self.current_bankroll > self.peak_bankroll:
                self.peak_bankroll = self.current_bankroll
        
        self._check_circuit_breaker()
    
    def _check_circuit_breaker(self):
        """Check if drawdown limit is breached."""
        drawdown = (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll
        
        if drawdown >= self.drawdown_limit:
            self.circuit_breaker_active = True
            logger.critical(
                f"CIRCUIT BREAKER ACTIVATED! Drawdown: {drawdown:.2%}"
            )
    
    def reset_circuit_breaker(self):
        """Manually reset circuit breaker after review."""
        self.circuit_breaker_active = False
        logger.info("Circuit breaker manually reset")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        total_profit = self.current_bankroll - self.initial_bankroll
        roi = total_profit / self.total_staked if self.total_staked > 0 else 0
        
        won_bets = sum(1 for b in self.bet_history if b.get('status') == 'won')
        lost_bets = sum(1 for b in self.bet_history if b.get('status') == 'lost')
        win_rate = won_bets / (won_bets + lost_bets) if (won_bets + lost_bets) > 0 else 0
        
        return {
            'initial_bankroll': self.initial_bankroll,
            'current_bankroll': self.current_bankroll,
            'total_profit': total_profit,
            'roi': roi,
            'total_staked': self.total_staked,
            'win_rate': win_rate,
            'bets_placed': len(self.bet_history),
            'circuit_breaker_active': self.circuit_breaker_active,
            'current_exposure': self.get_current_exposure()
        }


if __name__ == "__main__":
    fusion = HybridFusionEngine()
    xgb_probs = np.array([0.55, 0.25, 0.20])
    llm_signals = {'absences_impact': 7.0, 'fatigue_risk': 3.0, 'morale_score': 2.5}
    fused = fusion.fuse_probabilities(xgb_probs, llm_signals)
    print(f"Fused Probs: {fused}")
