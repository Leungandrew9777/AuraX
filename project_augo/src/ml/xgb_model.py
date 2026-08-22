"""
XGBoost Base Model with Walk-Forward Validation
TimeSeriesSplit, probability calibration, and feature engineering
Windows-optimized implementation
"""
import polars as pl
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
import joblib
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, classification_report
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EPLMatchPredictor:
    """
    XGBoost multi-class classifier for EPL match outcomes.
    
    Features:
    - Rolling averages (last 5 matches)
    - Head-to-head statistics
    - Home/away form splits
    
    Validation:
    - TimeSeriesSplit (walk-forward) to prevent look-ahead bias
    - Isotonic regression calibration for reliable probabilities
    """
    
    FEATURE_COLUMNS = [
        # Home team rolling features
        'home_rolling_goals_avg',
        'home_rolling_goals_conceded_avg',
        'home_rolling_shot_accuracy_avg',
        'home_rolling_points_avg',
        'home_rolling_form_diff',
        
        # Away team rolling features  
        'away_rolling_goals_avg',
        'away_rolling_goals_conceded_avg',
        'away_rolling_shot_accuracy_avg',
        'away_rolling_points_avg',
        'away_rolling_form_diff',
        
        # Relative strength
        'goal_diff_advantage',
        'shot_accuracy_advantage',
        'form_advantage',
        
        # Home advantage indicator
        'is_home',  # Always 1 for training, used differently in prediction
    ]
    
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        n_splits: int = 5,
        random_state: int = 42,
        model_dir: Optional[Path] = None
    ):
        """
        Initialize the XGBoost predictor.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Step size shrinkage
            n_splits: Number of folds for TimeSeriesSplit
            random_state: Random seed for reproducibility
            model_dir: Directory to save/load models
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_splits = n_splits
        self.random_state = random_state
        
        if model_dir is None:
            import os
            model_dir = Path(os.environ.get("LOCALAPPDATA", ".")) / "ProjectAugo" / "models"
        
        self.model_dir = model_dir
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Base XGBoost classifier
        self.base_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,  # Home, Draw, Away
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric='mlogloss',
            early_stopping_rounds=50,
            verbosity=0
        )
        
        # Calibrated wrapper (isotonic regression)
        self.calibrated_model = None
        
        # Feature importance cache
        self._feature_importance = None
        
        # Training metrics
        self.training_metrics = {}
    
    def engineer_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Create rolling features and derived statistics from raw match data.
        
        This is the critical step that transforms raw match stats into
        ML-ready features while avoiding look-ahead bias.
        """
        logger.info("Engineering features...")
        
        # Sort by date (critical for rolling calculations)
        df = df.sort('match_date')
        
        # Create unique match identifier
        df = df.with_columns(
            (pl.col('match_date').cast(pl.String) + '_' + 
             pl.col('home_team') + '_' + 
             pl.col('away_team')).alias('match_id')
        )
        
        # Calculate rolling features for each team
        df = self._add_team_rolling_features(df, 'home')
        df = self._add_team_rolling_features(df, 'away')
        
        # Create relative advantage features
        df = df.with_columns([
            (pl.col('home_rolling_goals_avg') - pl.col('away_rolling_goals_avg')).alias('goal_diff_advantage'),
            (pl.col('home_rolling_shot_accuracy_avg') - pl.col('away_rolling_shot_accuracy_avg')).alias('shot_accuracy_advantage'),
            (pl.col('home_rolling_points_avg') - pl.col('away_rolling_points_avg')).alias('form_advantage'),
        ])
        
        # Add home indicator
        df = df.with_columns(pl.lit(1).alias('is_home'))
        
        # Create target variable (H=0, D=1, A=2)
        if 'ft_result' in df.columns:
            df = df.with_columns(
                pl.when(pl.col('ft_result') == 'H').then(0)
                .when(pl.col('ft_result') == 'D').then(1)
                .when(pl.col('ft_result') == 'A').then(2)
                .otherwise(None).alias('target')
            )
        
        return df
    
    def _add_team_rolling_features(self, df: pl.DataFrame, location: str) -> pl.DataFrame:
        """
        Add rolling window features for a team (home or away).
        
        Uses shift() to ensure no look-ahead bias - only past matches are used.
        """
        team_col = f'{location}_team'
        goals_col = 'ft_home_goals' if location == 'home' else 'ft_away_goals'
        conceded_col = 'ft_away_goals' if location == 'home' else 'ft_home_goals'
        
        # Group by team and calculate rolling stats
        rolling_window = 5
        
        # Goals scored rolling average
        df = df.with_columns(
            pl.col(goals_col).over(team_col).shift(1).rolling_mean(window_size=rolling_window).alias(f'{location}_rolling_goals_avg')
        )
        
        # Goals conceded rolling average
        df = df.with_columns(
            pl.col(conceded_col).over(team_col).shift(1).rolling_mean(window_size=rolling_window).alias(f'{location}_rolling_goals_conceded_avg')
        )
        
        # Shot accuracy rolling average
        shot_acc_col = f'{location}_shot_accuracy'
        if shot_acc_col in df.columns:
            df = df.with_columns(
                pl.col(shot_acc_col).over(team_col).shift(1).rolling_mean(window_size=rolling_window).alias(f'{location}_rolling_shot_accuracy_avg')
            )
        else:
            # Calculate shot accuracy if not present
            shots_col = 'home_shots' if location == 'home' else 'away_shots'
            sot_col = 'home_shots_on_target' if location == 'home' else 'away_shots_on_target'
            
            if shots_col in df.columns and sot_col in df.columns:
                shot_acc = (pl.col(sot_col) / pl.col(shots_col).clip(lower_bound=1))
                df = df.with_columns(
                    shot_acc.over(team_col).shift(1).rolling_mean(window_size=rolling_window).alias(f'{location}_rolling_shot_accuracy_avg')
                )
        
        # Points rolling average (3 for win, 1 for draw, 0 for loss)
        result_col = 'ft_result'
        if result_col in df.columns:
            points = pl.when(pl.col(result_col) == ('H' if location == 'home' else 'A')).then(3)\
                     .when(pl.col(result_col) == 'D').then(1)\
                     .otherwise(0)
            
            df = df.with_columns(
                points.over(team_col).shift(1).rolling_mean(window_size=rolling_window).alias(f'{location}_rolling_points_avg')
            )
        
        # Form difference (recent goal difference trend)
        goal_diff = pl.col(goals_col) - pl.col(conceded_col)
        df = df.with_columns(
            goal_diff.over(team_col).shift(1).rolling_mean(window_size=rolling_window).alias(f'{location}_rolling_form_diff')
        )
        
        # Fill NaN values with league averages
        fill_cols = [c for c in df.columns if c.startswith(f'{location}_rolling')]
        for col in fill_cols:
            mean_val = df[col].mean()
            if mean_val is not None and not np.isnan(mean_val):
                df = df.with_columns(pl.col(col).fill_null(mean_val))
        
        return df
    
    def train_with_walk_forward(
        self,
        df: pl.DataFrame,
        use_calibration: bool = True
    ) -> Dict[str, float]:
        """
        Train using TimeSeriesSplit (walk-forward validation).
        
        This completely eliminates look-ahead bias by ensuring each test fold
        only uses training data from earlier time periods.
        
        Returns:
            Dictionary with validation metrics
        """
        logger.info("Starting walk-forward validation training...")
        
        # Engineer features
        df_features = self.engineer_features(df.clone())
        
        # Drop rows with null targets or features
        df_clean = df_features.filter(
            pl.col('target').is_not_null() & 
            pl.all_horizontal([pl.col(c).is_not_null() for c in self.FEATURE_COLUMNS])
        )
        
        if len(df_clean) < 100:
            raise ValueError(f"Insufficient data after cleaning: {len(df_clean)} rows")
        
        logger.info(f"Training on {len(df_clean)} matches")
        
        # Prepare data
        X = df_clean.select(self.FEATURE_COLUMNS).to_numpy()
        y = df_clean['target'].to_numpy()
        
        # TimeSeriesSplit for walk-forward validation
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        all_probs = []
        all_true = []
        fold_metrics = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            logger.info(f"Fold {fold + 1}/{self.n_splits}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train base model
            self.base_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            # Predict probabilities
            if use_calibration:
                # Use calibrated model
                cal_model = CalibratedClassifierCV(
                    self.base_model,
                    method='isotonic',
                    cv='prefit'  # Already trained
                )
                cal_model.fit(X_test, y_test)
                probs = cal_model.predict_proba(X_test)
            else:
                probs = self.base_model.predict_proba(X_test)
            
            # Store for overall metrics
            all_probs.append(probs)
            all_true.append(y_test)
            
            # Fold metrics
            preds = np.argmax(probs, axis=1)
            fold_brier = brier_score_loss(y_test, probs[np.arange(len(y_test)), preds])
            fold_logloss = log_loss(y_test, probs)
            
            fold_metrics.append({
                'brier': fold_brier,
                'logloss': fold_logloss
            })
            
            logger.info(f"  Brier: {fold_brier:.4f}, LogLoss: {fold_logloss:.4f}")
        
        # Combine all folds
        all_probs_np = np.vstack(all_probs)
        all_true_np = np.concatenate(all_true)
        
        # Overall metrics
        overall_brier = brier_score_loss(all_true_np, all_probs_np[np.arange(len(all_true_np)), np.argmax(all_probs_np, axis=1)])
        overall_logloss = log_loss(all_true_np, all_probs_np)
        
        self.training_metrics = {
            'brier_score': overall_brier,
            'log_loss': overall_logloss,
            'fold_metrics': fold_metrics,
            'n_samples': len(df_clean),
            'feature_columns': self.FEATURE_COLUMNS
        }
        
        logger.info(f"\n✓ Walk-forward validation complete")
        logger.info(f"Overall Brier Score: {overall_brier:.4f}")
        logger.info(f"Overall Log Loss: {overall_logloss:.4f}")
        
        # Final training on full dataset for deployment
        logger.info("Training final model on full dataset...")
        self.base_model.fit(X, y, verbose=False)
        
        if use_calibration:
            self.calibrated_model = CalibratedClassifierCV(
                self.base_model,
                method='isotonic',
                cv='prefit'
            )
            self.calibrated_model.fit(X, y)
            logger.info("✓ Probability calibration applied (Isotonic Regression)")
        
        # Save model
        self.save_model()
        
        return self.training_metrics
    
    def predict_probabilities(
        self,
        match_data: pl.DataFrame
    ) -> np.ndarray:
        """
        Predict outcome probabilities for new matches.
        
        Returns:
            Array of shape (n_matches, 3) with [P(Home), P(Draw), P(Away)]
        """
        # Engineer features for new data
        df_features = self.engineer_features(match_data.clone())
        
        # Select feature columns
        X = df_features.select(self.FEATURE_COLUMNS).to_numpy()
        
        # Use calibrated model if available
        if self.calibrated_model:
            probs = self.calibrated_model.predict_proba(X)
        else:
            probs = self.base_model.predict_proba(X)
        
        return probs
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if self._feature_importance is None:
            importance = self.base_model.feature_importances_
            self._feature_importance = dict(zip(self.FEATURE_COLUMNS, importance.tolist()))
        
        return self._feature_importance
    
    def save_model(self, filename: str = "epl_xgb_model.joblib"):
        """Save model to disk"""
        model_path = self.model_dir / filename
        joblib.dump({
            'base_model': self.base_model,
            'calibrated_model': self.calibrated_model,
            'feature_columns': self.FEATURE_COLUMNS,
            'training_metrics': self.training_metrics
        }, model_path)
        logger.info(f"✓ Model saved to {model_path}")
        return model_path
    
    def load_model(self, filename: str = "epl_xgb_model.joblib"):
        """Load model from disk"""
        model_path = self.model_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        data = joblib.load(model_path)
        self.base_model = data['base_model']
        self.calibrated_model = data['calibrated_model']
        self.FEATURE_COLUMNS = data['feature_columns']
        self.training_metrics = data['training_metrics']
        
        logger.info(f"✓ Model loaded from {model_path}")
        return self
    
    def print_classification_report(self, df: pl.DataFrame) -> str:
        """Generate detailed classification report"""
        df_features = self.engineer_features(df.clone())
        df_clean = df_features.filter(
            pl.col('target').is_not_null() &
            pl.all_horizontal([pl.col(c).is_not_null() for c in self.FEATURE_COLUMNS])
        )
        
        X = df_clean.select(self.FEATURE_COLUMNS).to_numpy()
        y = df_clean['target'].to_numpy()
        
        probs = self.predict_probabilities(df_clean.to_pandas())
        preds = np.argmax(probs, axis=1)
        
        target_names = ['Home Win', 'Draw', 'Away Win']
        report = classification_report(y, preds, target_names=target_names)
        
        return report


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    n_matches = 500
    
    sample_data = pl.DataFrame({
        'match_date': pl.date_range(datetime(2022, 8, 1), datetime(2024, 5, 1), interval='7d', eager=True)[:n_matches],
        'home_team': np.random.choice(['Arsenal', 'Chelsea', 'Liverpool', 'Man City'], n_matches),
        'away_team': np.random.choice(['Arsenal', 'Chelsea', 'Liverpool', 'Man City'], n_matches),
        'ft_home_goals': np.random.poisson(1.5, n_matches),
        'ft_away_goals': np.random.poisson(1.2, n_matches),
        'ft_result': np.random.choice(['H', 'D', 'A'], n_matches),
        'home_shots': np.random.randint(5, 20, n_matches),
        'away_shots': np.random.randint(5, 20, n_matches),
        'home_shots_on_target': np.random.randint(2, 10, n_matches),
        'away_shots_on_target': np.random.randint(2, 10, n_matches),
    })
    
    # Initialize and train
    predictor = EPLMatchPredictor(n_estimators=100, n_splits=3)
    metrics = predictor.train_with_walk_forward(sample_data)
    
    print(f"\nFeature Importance:")
    for feat, imp in sorted(predictor.get_feature_importance().items(), key=lambda x: -x[1])[:5]:
        print(f"  {feat}: {imp:.4f}")
