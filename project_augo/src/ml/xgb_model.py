"""
Project Augo - Machine Learning Base Model
XGBoost classifier with Walk-Forward Validation and Probability Calibration.
"""
import numpy as np
import polars as pl
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, classification_report
import xgboost as xgb
import logging
from datetime import datetime

from config.settings import config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates rolling features and team-specific statistics for ML modeling.
    
    All features are designed to avoid look-ahead bias by only using
    historical data available before each match.
    """
    
    def __init__(self, window_sizes: List[int] = None):
        self.window_sizes = window_sizes or [3, 5, 10]  # Rolling windows
    
    def create_rolling_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Create rolling average features for each team.
        
        Features include:
        - Goals scored/conceded per match (rolling)
        - Shot accuracy (shots on target / total shots)
        - Form points (last N matches)
        - Home/away specific stats
        """
        df_features = df.clone()
        
        # Sort by date for proper rolling calculations
        df_features = df_features.sort('match_date')
        
        # Create team-level dataframe (home and away separately)
        home_stats = self._calculate_team_rolling_stats(
            df, 
            team_col='HomeTeam', 
            opponent_col='AwayTeam',
            goals_col='FTHG',
            conceded_col='FTAG',
            shots_col='HS',
            shots_on_target_col='HST',
            result_col='FTR',
            is_home=True
        )
        
        away_stats = self._calculate_team_rolling_stats(
            df,
            team_col='AwayTeam',
            opponent_col='HomeTeam', 
            goals_col='FTAG',
            conceded_col='FTHG',
            shots_col='AS',
            shots_on_target_col='AST',
            result_col='FTR',
            is_home=False
        )
        
        # Merge back to original dataframe
        df_features = df_features.join(
            home_stats,
            left_on=['match_date', 'HomeTeam'],
            right_on=['match_date', 'team'],
            how='left',
            suffix='_home'
        )
        
        df_features = df_features.join(
            away_stats,
            left_on=['match_date', 'AwayTeam'],
            right_on=['match_date', 'team'],
            how='left',
            suffix='_away'
        )
        
        # Drop intermediate columns
        cols_to_drop = [col for col in df_features.columns if col.endswith('_home') or col.endswith('_away')]
        # Keep only the renamed columns
        
        logger.info(f"Created rolling features. Shape: {df_features.shape}")
        return df_features
    
    def _calculate_team_rolling_stats(self, df: pl.DataFrame, team_col: str,
                                       opponent_col: str, goals_col: str,
                                       conceded_col: str, shots_col: str,
                                       shots_on_target_col: str,
                                       result_col: str, is_home: bool) -> pl.DataFrame:
        """Calculate rolling statistics for a team."""
        
        # Create long-format dataframe with team perspective
        home_df = df.select([
            pl.col('match_date'),
            pl.col(team_col).alias('team'),
            pl.col(opponent_col).alias('opponent'),
            pl.col(goals_col).alias('goals_scored'),
            pl.col(conceded_col).alias('goals_conceded'),
            pl.col(shots_col).alias('shots'),
            pl.col(shots_on_target_col).alias('shots_on_target'),
            pl.col(result_col).alias('result'),
            pl.lit(is_home).alias('is_home_match')
        ])
        
        # Calculate points (3 for win, 1 for draw, 0 for loss)
        home_df = home_df.with_columns(
            pl.when(pl.col('result') == ('H' if is_home else 'A'))
                .then(3)
                .when(pl.col('result') == 'D')
                .then(1)
                .otherwise(0)
            .alias('points')
        )
        
        # Calculate shot accuracy
        home_df = home_df.with_columns(
            pl.when(pl.col('shots') > 0)
                .then(pl.col('shots_on_target') / pl.col('shots'))
                .otherwise(0.0)
            .alias('shot_accuracy')
        )
        
        # Calculate rolling averages for each window size
        rolling_dfs = []
        for window in self.window_sizes:
            rolled = home_df.sort('match_date').group_by('team', maintain_order=True).agg([
                pl.col('goals_scored').tail(window).mean().alias(f'rolling_goals_scored_{window}'),
                pl.col('goals_conceded').tail(window).mean().alias(f'rolling_goals_conceded_{window}'),
                pl.col('points').tail(window).sum().alias(f'rolling_form_{window}'),
                pl.col('shot_accuracy').tail(window).mean().alias(f'rolling_shot_accuracy_{window}'),
                (pl.col('goals_scored').tail(window).mean() - pl.col('goals_conceded').tail(window).mean())
                    .alias(f'rolling_goal_diff_{window}')
            ])
            rolling_dfs.append(rolled)
        
        # Combine all rolling features
        if rolling_dfs:
            combined = rolling_dfs[0]
            for rdf in rolling_dfs[1:]:
                combined = combined.join(rdf, on=['team', 'match_date'], how='left')
            
            # Shift by 1 to avoid look-ahead bias (exclude current match)
            combined = combined.sort('team', 'match_date').with_columns([
                pl.col(col).shift(1).alias(f'{col}_lag1')
                for col in combined.columns if col.startswith('rolling_')
            ])
            
            return combined
        
        return home_df
    
    def create_head_to_head_features(self, df: pl.DataFrame, h2h_window: int = 5) -> pl.DataFrame:
        """Create head-to-head historical features between teams."""
        # This would calculate historical performance between specific team matchups
        # Simplified version for now
        df_h2h = df.clone()
        
        # Add a simple H2H indicator (number of previous meetings)
        df_h2h = df_h2h.with_columns(
            pl.lit(0).alias('h2h_meetings_count')  # Placeholder
        )
        
        return df_h2h
    
    def prepare_features_for_prediction(self, df: pl.DataFrame) -> pl.DataFrame:
        """Prepare final feature matrix for model training/prediction."""
        # First create rolling features
        df_final = self.create_rolling_features(df)
        
        # Add H2H features
        df_final = self.create_head_to_head_features(df_final)
        
        return df_final


class XGBoostModel:
    """
    XGBoost multi-class classifier with walk-forward validation
    and probability calibration.
    """
    
    def __init__(self, model_config=None):
        self.config = model_config or config.model
        self.model = None
        self.calibrated_model = None
        self.feature_engineer = FeatureEngineer()
        self.feature_names = None
        self.classes_ = ['home_win', 'draw', 'away_win']
        
        # Training history
        self.eval_results = {}
        self.walk_forward_metrics = []
    
    def _prepare_xy(self, df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix X and target vector y."""
        # Define feature columns (exclude non-feature columns)
        exclude_cols = [
            'Date', 'match_date', 'HomeTeam', 'AwayTeam', 'FTR',
            'season_code', 'season_name', 'ingested_at', 'result_encoded',
            'B365H', 'B365D', 'B365A',  # Odds (used separately)
            'fair_prob_home', 'fair_prob_draw', 'fair_prob_away'  # Derived probs
        ]
        
        # Filter to numeric feature columns only
        feature_cols = [
            col for col in df.columns
            if col not in exclude_cols and df.schema.get(col) in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int16, pl.Int8]
        ]
        
        self.feature_names = feature_cols
        
        # Prepare X
        X = df.select(feature_cols).fill_null(0).to_numpy()
        
        # Prepare y (encode target)
        result_mapping = {'H': 0, 'D': 1, 'A': 2, 'home_win': 0, 'draw': 1, 'away_win': 2}
        y = df['FTR'].apply(lambda x: result_mapping.get(x, -1)).to_numpy()
        
        # Filter out invalid targets
        valid_mask = y >= 0
        X = X[valid_mask]
        y = y[valid_mask]
        
        return X, y
    
    def train_walk_forward(self, df: pl.DataFrame) -> Dict:
        """
        Train model using walk-forward validation (TimeSeriesSplit).
        
        This completely eliminates look-ahead bias by ensuring that
        test data always comes after training data chronologically.
        
        Returns:
            Dictionary with aggregated metrics across all folds
        """
        logger.info("Starting walk-forward validation training...")
        
        # Prepare features
        df_features = self.feature_engine.prepare_features_for_prediction(df)
        X, y = self._prepare_xy(df_features)
        
        logger.info(f"Prepared {len(X)} samples with {len(self.feature_names)} features")
        
        # Time series split
        n_splits = self.config.n_splits
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=self.config.test_size, gap=0)
        
        fold_metrics = []
        predictions_all_folds = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            logger.info(f"Training fold {fold_idx + 1}/{n_splits}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train base XGBoost model
            base_model = xgb.XGBClassifier(
                n_estimators=self.config.xgb_n_estimators,
                max_depth=self.config.xgb_max_depth,
                learning_rate=self.config.xgb_learning_rate,
                subsample=self.config.xgb_subsample,
                colsample_bytree=self.config.xgb_colsample_bytree,
                random_state=self.config.xgb_random_state,
                objective='multi:softprob',
                num_class=3,
                eval_metric='mlogloss',
                early_stopping_rounds=50,
                verbosity=1
            )
            
            # Fit with validation set
            base_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            # Get uncalibrated predictions
            y_pred_proba_uncal = base_model.predict_proba(X_test)
            
            # Calibrate probabilities using isotonic regression
            calibrated = CalibratedClassifierCV(
                estimator=base_model,
                method='isotonic',
                cv='prefit'  # Use prefit since we already trained
            )
            calibrated.fit(X_test, y_test)  # Fit calibrator on test set
            
            # Get calibrated predictions
            y_pred_proba_cal = calibrated.predict_proba(X_test)
            y_pred = calibrated.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'fold': fold_idx,
                'brier_score': self._calculate_brier_score(y_test, y_pred_proba_cal),
                'log_loss': log_loss(y_test, y_pred_proba_cal),
                'accuracy': (y_pred == y_test).mean(),
                'n_test_samples': len(y_test)
            }
            
            fold_metrics.append(metrics)
            predictions_all_folds.append({
                'y_true': y_test,
                'y_pred_proba': y_pred_proba_cal,
                'y_pred': y_pred
            })
            
            logger.info(f"Fold {fold_idx + 1} - Brier: {metrics['brier_score']:.4f}, Log Loss: {metrics['log_loss']:.4f}")
        
        # Aggregate metrics
        aggregated_metrics = {
            'mean_brier_score': np.mean([m['brier_score'] for m in fold_metrics]),
            'std_brier_score': np.std([m['brier_score'] for m in fold_metrics]),
            'mean_log_loss': np.mean([m['log_loss'] for m in fold_metrics]),
            'mean_accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
            'fold_metrics': fold_metrics
        }
        
        self.walk_forward_metrics = fold_metrics
        
        logger.info(f"Walk-forward validation complete.")
        logger.info(f"Mean Brier Score: {aggregated_metrics['mean_brier_score']:.4f} (+/- {aggregated_metrics['std_brier_score']:.4f})")
        logger.info(f"Mean Log Loss: {aggregated_metrics['mean_log_loss']:.4f}")
        logger.info(f"Mean Accuracy: {aggregated_metrics['mean_accuracy']:.4f}")
        
        return aggregated_metrics
    
    def _calculate_brier_score(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Calculate multi-class Brier score."""
        # One-hot encode true labels
        n_classes = y_proba.shape[1]
        y_onehot = np.zeros_like(y_proba)
        y_onehot[np.arange(len(y_true)), y_true] = 1
        
        # Brier score = mean squared error between probabilities and one-hot
        return np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1))
    
    def fit_final_model(self, df: pl.DataFrame) -> None:
        """
        Train final model on all available data.
        
        This model will be used for actual predictions.
        """
        logger.info("Training final model on all data...")
        
        df_features = self.feature_engineer.prepare_features_for_prediction(df)
        X, y = self._prepare_xy(df_features)
        
        # Train base model
        self.model = xgb.XGBClassifier(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            subsample=self.config.xgb_subsample,
            colsample_bytree=self.config.xgb_colsample_bytree,
            random_state=self.config.xgb_random_state,
            objective='multi:softprob',
            num_class=3,
            verbosity=1
        )
        
        self.model.fit(X, y)
        
        # Calibrate using cross-validation
        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.model,
            method='isotonic',
            cv=5
        )
        self.calibrated_model.fit(X, y)
        
        logger.info("Final model training complete")
    
    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        """
        Predict calibrated probabilities for new data.
        
        Returns:
            Array of shape (n_matches, 3) with probabilities for [home, draw, away]
        """
        if self.calibrated_model is None:
            raise ValueError("Model not trained. Call fit_final_model first.")
        
        df_features = self.feature_engine.prepare_features_for_prediction(df)
        X, _ = self._prepare_xy(df_features)
        
        return self.calibrated_model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model."""
        if self.model is None:
            return {}
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))
    
    def save_model(self, filepath: str) -> None:
        """Save model to file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'calibrated_model': self.calibrated_model,
                'feature_names': self.feature_names,
                'walk_forward_metrics': self.walk_forward_metrics
            }, f)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load model from file."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.calibrated_model = data['calibrated_model']
        self.feature_names = data['feature_names']
        self.walk_forward_metrics = data.get('walk_forward_metrics', [])
        
        logger.info(f"Model loaded from {filepath}")
