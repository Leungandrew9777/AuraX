"""
Project Augo - Hybrid Fusion Meta-Learner and Bankroll Engine
Combines XGBoost probabilities with LLM qualitative signals,
calculates edge, and sizes stakes using Kelly Criterion.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from config.settings import config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MatchPrediction:
    """Container for match prediction outputs."""
    match_id: int
    home_team: str
    away_team: str
    match_date: str
    
    # Model probabilities
    prob_home_xgb: float
    prob_draw_xgb: float
    prob_away_xgb: float
    
    # Qualitative adjustments (per team)
    home_absence_impact: float
    home_fatigue_risk: float
    home_morale_score: float
    away_absence_impact: float
    away_fatigue_risk: float
    away_morale_score: float
    
    # Final fused probabilities
    prob_home_final: float
    prob_draw_final: float
    prob_away_final: float
    
    # Market odds and implied probs
    odds_home: float
    odds_draw: float
    odds_away: float
    fair_prob_home: float
    fair_prob_draw: float
    fair_prob_away: float
    
    # Edge calculations
    edge_home: float
    edge_draw: float
    edge_away: float
    
    # Betting recommendation
    recommended_outcome: Optional[str]  # 'H', 'D', 'A', or None
    recommended_stake: float  # Fraction of bankroll
    kelly_fraction_used: float
    
    # Metadata
    model_version: str = "v1.0"
    has_positive_edge: bool = False


class HybridFusionEngine:
    """
    Combines base XGBoost probabilities with LLM-derived qualitative signals.
    
    Uses a meta-learner (Logistic Regression / Ridge) to learn optimal
    weighting between quantitative and qualitative inputs.
    """
    
    def __init__(self, alpha: float = None):
        self.alpha = alpha or config.model.meta_learner_alpha
        self.meta_model = None
        self.feature_weights = {
            'absence_impact_weight': -0.02,  # Negative: absences hurt
            'fatigue_weight': -0.015,  # Negative: fatigue hurts
            'morale_weight': 0.03,  # Positive: morale helps
            'confidence_weight': 0.1  # Weight for LLM confidence
        }
    
    def calculate_qualitative_adjustment(self, absence_impact: float,
                                          fatigue_risk: float,
                                          morale_score: float,
                                          confidence: float = 0.5) -> float:
        """
        Calculate probability adjustment based on qualitative factors.
        
        The adjustment is scaled to be a small perturbation to avoid
        overwhelming the quantitative base model.
        
        Args:
            absence_impact: 0-10 scale (higher = more key players out)
            fatigue_risk: 0-10 scale (higher = more fatigued)
            morale_score: -5 to +5 scale
            confidence: LLM confidence in extraction (0-1)
            
        Returns:
            Probability adjustment factor (-0.15 to +0.15 typical range)
        """
        # Normalize inputs to 0-1 scale
        absence_normalized = absence_impact / 10.0
        fatigue_normalized = fatigue_risk / 10.0
        morale_normalized = (morale_score + 5.0) / 10.0  # Shift from [-5,5] to [0,1]
        
        # Calculate weighted adjustment
        raw_adjustment = (
            self.feature_weights['absence_impact_weight'] * absence_normalized +
            self.feature_weights['fatigue_weight'] * fatigue_normalized +
            self.feature_weights['morale_weight'] * morale_normalized
        )
        
        # Scale by confidence (low confidence = less adjustment)
        adjusted = raw_adjustment * confidence
        
        # Clip to reasonable bounds
        return np.clip(adjusted, -0.15, 0.15)
    
    def fuse_probabilities(self, xgb_probs: np.ndarray,
                          home_signals: Dict[str, float],
                          away_signals: Dict[str, float]) -> np.ndarray:
        """
        Fuse XGBoost probabilities with qualitative signals.
        
        Args:
            xgb_probs: Array of [P(home), P(draw), P(away)] from XGBoost
            home_signals: Dict with home team qualitative metrics
            away_signals: Dict with away team qualitative metrics
            
        Returns:
            Adjusted probability array
        """
        p_home, p_draw, p_away = xgb_probs
        
        # Get signal values with defaults
        home_absence = home_signals.get('key_absences_impact', 0.0)
        home_fatigue = home_signals.get('fatigue_rotation_risk', 0.0)
        home_morale = home_signals.get('morale_sentiment_score', 0.0)
        home_confidence = home_signals.get('confidence_score', 0.5)
        
        away_absence = away_signals.get('key_absences_impact', 0.0)
        away_fatigue = away_signals.get('fatigue_rotation_risk', 0.0)
        away_morale = away_signals.get('morale_sentiment_score', 0.0)
        away_confidence = away_signals.get('confidence_score', 0.5)
        
        # Calculate adjustments for each outcome
        home_adjustment = self.calculate_qualitative_adjustment(
            home_absence, home_fatigue, home_morale, home_confidence
        )
        
        away_adjustment = self.calculate_qualitative_adjustment(
            away_absence, away_fatigue, away_morale, away_confidence
        )
        
        # Apply adjustments
        # Home advantage increases if home team has better signals
        p_home_adj = p_home + home_adjustment - (away_adjustment * 0.5)
        
        # Away advantage increases if away team has better signals
        p_away_adj = p_away + away_adjustment - (home_adjustment * 0.5)
        
        # Draw probability adjusts inversely (more uncertainty = more draw likelihood)
        total_signal_strength = abs(home_adjustment) + abs(away_adjustment)
        p_draw_adj = p_draw - (home_adjustment + away_adjustment) * 0.3
        
        # Ensure probabilities sum to 1
        total = p_home_adj + p_draw_adj + p_away_adj
        if total > 0:
            p_home_adj /= total
            p_draw_adj /= total
            p_away_adj /= total
        
        # Clip to valid probability range
        p_home_adj = np.clip(p_home_adj, 0.01, 0.98)
        p_draw_adj = np.clip(p_draw_adj, 0.01, 0.98)
        p_away_adj = np.clip(p_away_adj, 0.01, 0.98)
        
        # Re-normalize
        total = p_home_adj + p_draw_adj + p_away_adj
        p_home_adj /= total
        p_draw_adj /= total
        p_away_adj /= total
        
        return np.array([p_home_adj, p_draw_adj, p_away_adj])
    
    def train_meta_learner(self, historical_data: List[Dict]) -> None:
        """
        Train meta-learner on historical predictions and outcomes.
        
        This would use logistic regression to learn optimal weights
        for combining XGBoost and qualitative signals.
        
        For now, uses hand-tuned weights (above).
        """
        logger.info("Meta-learner training not yet implemented. Using default weights.")
        # TODO: Implement proper meta-learner training
        pass


class BankrollEngine:
    """
    Calculates betting edges and sizes stakes using Kelly Criterion.
    
    Implements risk management features including:
    - Fractional Kelly (e.g., Quarter-Kelly)
    - Maximum stake limits
    - Drawdown circuit breakers
    """
    
    def __init__(self, bankroll_config=None):
        self.config = bankroll_config or config.bankroll
        self.current_bankroll = self.config.initial_bankroll
        self.total_profit_loss = 0.0
        self.bet_history: List[Dict] = []
        self.max_drawdown = 0.0
        self.is_circuit_breaker_active = False
    
    def calculate_fair_probability(self, odds_home: float, odds_draw: float,
                                    odds_away: float, method: str = 'power') -> Tuple[float, float, float]:
        """
        Convert decimal odds to fair implied probabilities by removing vig.
        
        Uses the Power method (proportional normalization) or Shin method.
        
        Args:
            odds_home, odds_draw, odds_away: Decimal odds from bookmaker
            method: 'power' or 'shin'
            
        Returns:
            Tuple of (fair_prob_home, fair_prob_draw, fair_prob_away)
        """
        # Raw implied probabilities
        raw_home = 1.0 / odds_home
        raw_draw = 1.0 / odds_draw
        raw_away = 1.0 / odds_away
        
        # Total implied probability (includes vig)
        total_implied = raw_home + raw_draw + raw_away
        vig = total_implied - 1.0
        
        if method == 'power':
            # Power method: proportional normalization
            fair_home = raw_home / total_implied
            fair_draw = raw_draw / total_implied
            fair_away = raw_away / total_implied
            
        elif method == 'shin':
            # Shin method: more sophisticated margin removal
            # Simplified implementation
            margin_factor = 1.0 / total_implied
            fair_home = raw_home * margin_factor
            fair_draw = raw_draw * margin_factor
            fair_away = raw_away * margin_factor
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return fair_home, fair_draw, fair_away
    
    def calculate_edge(self, model_prob: float, market_prob: float) -> float:
        """
        Calculate betting edge.
        
        Edge = Model Probability - Market Implied Probability
        
        Positive edge indicates a potentially valuable bet.
        """
        return model_prob - market_prob
    
    def calculate_kelly_stake(self, edge: float, odds: float) -> float:
        """
        Calculate Kelly Criterion stake.
        
        Kelly Stake = (bp - q) / b
        where:
            b = odds - 1 (decimal odds minus 1)
            p = model probability of winning
            q = 1 - p (probability of losing)
        
        Simplified: Kelly = Edge / (odds - 1)
        
        Returns:
            Fraction of bankroll to stake (can be negative for no bet)
        """
        if odds <= 1.0:
            return 0.0
        
        b = odds - 1.0
        p = edge + (1.0 / odds)  # Recover model probability
        q = 1.0 - p
        
        if b == 0:
            return 0.0
        
        kelly = (b * p - q) / b
        
        # Apply fractional Kelly (e.g., Quarter-Kelly)
        fractional_kelly = kelly * self.config.kelly_fraction
        
        # Only bet if positive edge
        if fractional_kelly <= 0:
            return 0.0
        
        # Cap at maximum stake percentage
        max_stake = self.config.max_stake_percentage
        return min(fractional_kelly, max_stake)
    
    def check_circuit_breaker(self) -> bool:
        """
        Check if drawdown circuit breaker should be activated.
        
        Returns:
            True if betting should stop due to excessive drawdown
        """
        current_drawdown = abs(min(0, self.total_profit_loss)) / self.config.initial_bankroll
        
        if current_drawdown >= self.config.max_drawdown_limit:
            self.is_circuit_breaker_active = True
            logger.warning(f"Circuit breaker activated! Drawdown: {current_drawdown:.2%}")
            return True
        
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        return False
    
    def evaluate_bet(self, match_prediction: MatchPrediction) -> MatchPrediction:
        """
        Evaluate a match prediction and generate betting recommendation.
        
        Applies edge threshold and Kelly sizing.
        """
        # Check circuit breaker first
        if self.check_circuit_breaker():
            match_prediction.recommended_outcome = None
            match_prediction.recommended_stake = 0.0
            logger.info("Betting suspended due to circuit breaker")
            return match_prediction
        
        # Find best edge
        edges = {
            'H': match_prediction.edge_home,
            'D': match_prediction.edge_draw,
            'A': match_prediction.edge_away
        }
        
        best_outcome = max(edges, key=edges.get)
        best_edge = edges[best_outcome]
        
        # Check if edge exceeds minimum threshold
        if best_edge < self.config.min_edge_threshold:
            match_prediction.recommended_outcome = None
            match_prediction.recommended_stake = 0.0
            match_prediction.has_positive_edge = False
            logger.debug(f"No bet: best edge {best_edge:.4f} below threshold {self.config.min_edge_threshold}")
            return match_prediction
        
        # Get odds for best outcome
        odds_map = {'H': match_prediction.odds_home, 'D': match_prediction.odds_draw, 'A': match_prediction.odds_away}
        prob_map = {'H': match_prediction.prob_home_final, 'D': match_prediction.prob_draw_final, 'A': match_prediction.prob_away_final}
        fair_map = {'H': match_prediction.fair_prob_home, 'D': match_prediction.fair_prob_draw, 'A': match_prediction.fair_prob_away}
        
        best_odds = odds_map[best_outcome]
        best_prob = prob_map[best_outcome]
        best_fair = fair_map[best_outcome]
        
        # Calculate Kelly stake
        stake_fraction = self.calculate_kelly_stake(best_edge, best_odds)
        
        # Record recommendation
        match_prediction.recommended_outcome = best_outcome
        match_prediction.recommended_stake = stake_fraction
        match_prediction.kelly_fraction_used = self.config.kelly_fraction
        match_prediction.has_positive_edge = True
        
        logger.info(f"Bet recommended: {best_outcome} with stake {stake_fraction:.4f} ({stake_fraction*100:.2f}% of bankroll)")
        
        return match_prediction
    
    def record_bet_result(self, outcome: str, stake_fraction: float, 
                          odds: float, won: bool) -> None:
        """
        Record the result of a placed bet.
        
        Updates bankroll and checks circuit breaker.
        """
        if won:
            profit = stake_fraction * self.current_bankroll * (odds - 1)
        else:
            profit = -stake_fraction * self.current_bankroll
        
        self.total_profit_loss += profit
        self.current_bankroll += profit
        
        bet_record = {
            'outcome': outcome,
            'stake_fraction': stake_fraction,
            'odds': odds,
            'won': won,
            'profit': profit,
            'bankroll_after': self.current_bankroll
        }
        
        self.bet_history.append(bet_record)
        
        logger.info(f"Bet result: {'WIN' if won else 'LOSS'}, P/L: {profit:.2f}, Bankroll: {self.current_bankroll:.2f}")
        
        # Check circuit breaker after update
        self.check_circuit_breaker()
    
    def get_performance_summary(self) -> Dict:
        """Get summary statistics of betting performance."""
        if not self.bet_history:
            return {
                'total_bets': 0,
                'total_pnl': 0.0,
                'roi': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'current_bankroll': self.current_bankroll
            }
        
        wins = sum(1 for b in self.bet_history if b['won'])
        total_staked = sum(b['stake_fraction'] * self.config.initial_bankroll for b in self.bet_history)
        
        return {
            'total_bets': len(self.bet_history),
            'total_pnl': self.total_profit_loss,
            'roi': self.total_profit_loss / total_staked if total_staked > 0 else 0.0,
            'win_rate': wins / len(self.bet_history),
            'max_drawdown': self.max_drawdown,
            'current_bankroll': self.current_bankroll,
            'circuit_breaker_active': self.is_circuit_breaker_active
        }


class PredictionPipeline:
    """
    End-to-end pipeline that combines all components.
    
    Orchestrates:
    1. XGBoost probability generation
    2. LLM signal retrieval
    3. Hybrid fusion
    4. Edge calculation
    5. Stake sizing
    """
    
    def __init__(self, fusion_engine: HybridFusionEngine = None,
                 bankroll_engine: BankrollEngine = None):
        self.fusion_engine = fusion_engine or HybridFusionEngine()
        self.bankroll_engine = bankroll_engine or BankrollEngine()
    
    def generate_predictions(self, matches: List[Dict],
                            xgb_probs: np.ndarray,
                            llm_signals: Dict[str, Dict]) -> List[MatchPrediction]:
        """
        Generate full predictions for a batch of matches.
        
        Args:
            matches: List of match dictionaries with metadata and odds
            xgb_probs: Numpy array of XGBoost probabilities [n_matches, 3]
            llm_signals: Dict mapping team names to their qualitative signals
            
        Returns:
            List of MatchPrediction objects with full analysis
        """
        predictions = []
        
        for i, match in enumerate(matches):
            # Extract XGBoost probabilities
            xgb_prob_array = xgb_probs[i] if len(xgb_probs.shape) > 1 else xgb_probs
            
            # Get team signals
            home_team = match.get('home_team', match.get('HomeTeam'))
            away_team = match.get('away_team', match.get('AwayTeam'))
            
            home_signals = llm_signals.get(home_team, {})
            away_signals = llm_signals.get(away_team, {})
            
            # Fuse probabilities
            fused_probs = self.fusion_engine.fuse_probabilities(
                xgb_prob_array, home_signals, away_signals
            )
            
            # Get market odds
            odds_home = match.get('odds_home', match.get('B365H', 0))
            odds_draw = match.get('odds_draw', match.get('B365D', 0))
            odds_away = match.get('odds_away', match.get('B365A', 0))
            
            # Calculate fair probabilities from odds
            fair_probs = self.bankroll_engine.calculate_fair_probability(
                odds_home, odds_draw, odds_away
            )
            
            # Calculate edges
            edge_home = fused_probs[0] - fair_probs[0]
            edge_draw = fused_probs[1] - fair_probs[1]
            edge_away = fused_probs[2] - fair_probs[2]
            
            # Create prediction object
            pred = MatchPrediction(
                match_id=match.get('match_id', i),
                home_team=home_team,
                away_team=away_team,
                match_date=str(match.get('match_date', '')),
                prob_home_xgb=xgb_prob_array[0],
                prob_draw_xgb=xgb_prob_array[1],
                prob_away_xgb=xgb_prob_array[2],
                home_absence_impact=home_signals.get('key_absences_impact', 0),
                home_fatigue_risk=home_signals.get('fatigue_rotation_risk', 0),
                home_morale_score=home_signals.get('morale_sentiment_score', 0),
                away_absence_impact=away_signals.get('key_absences_impact', 0),
                away_fatigue_risk=away_signals.get('fatigue_rotation_risk', 0),
                away_morale_score=away_signals.get('morale_sentiment_score', 0),
                prob_home_final=fused_probs[0],
                prob_draw_final=fused_probs[1],
                prob_away_final=fused_probs[2],
                odds_home=odds_home,
                odds_draw=odds_draw,
                odds_away=odds_away,
                fair_prob_home=fair_probs[0],
                fair_prob_draw=fair_probs[1],
                fair_prob_away=fair_probs[2],
                edge_home=edge_home,
                edge_draw=edge_draw,
                edge_away=edge_away,
                recommended_outcome=None,
                recommended_stake=0.0,
                kelly_fraction_used=self.bankroll_engine.config.kelly_fraction
            )
            
            # Evaluate bet
            pred = self.bankroll_engine.evaluate_bet(pred)
            predictions.append(pred)
        
        return predictions
    
    def export_to_telegram_format(self, predictions: List[MatchPrediction]) -> str:
        """Format predictions for Telegram message."""
        messages = []
        messages.append("🏴󠁧󠁢󠁥󠁮󠁧󠁿 **PROJECT AUGO - MATCH PREDICTIONS** 🏴󠁧󠁢󠁥󠁮󠁧󠁿\n")
        
        for pred in predictions:
            if pred.has_positive_edge:
                emoji = {'H': '🏠', 'D': '🤝', 'A': '✈️'}
                outcome_names = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}
                
                message = f"""
{emoji.get(pred.recommended_outcome, '⚽')} *{pred.home_team} vs {pred.away_team}*
📅 {pred.match_date}

💡 *Recommendation:* {outcome_names.get(pred.recommended_outcome)}
📊 *Probability:* {pred.prob_home_final:.1%} / {pred.prob_draw_final:.1%} / {pred.prob_away_final:.1%}
💰 *Odds:* {pred.odds_home:.2f} / {pred.odds_draw:.2f} / {pred.odds_away:.2f}
📈 *Edge:* {pred.edge_home:.1%} / {pred.edge_draw:.1%} / {pred.edge_away:.1%}

💵 *Stake:* {pred.recommended_stake*100:.2f}% of bankroll
🎯 *Confidence:* {'High' if pred.recommended_stake > 0.03 else 'Medium' if pred.recommended_stake > 0.01 else 'Low'}
---
"""
                messages.append(message)
        
        if len(messages) == 1:
            messages.append("\n⚠️ No bets meeting edge threshold for this gameweek.")
        
        return "\n".join(messages)
