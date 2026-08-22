"""
Project Augo - Main Entry Point
Orchestrates all components of the EPL betting framework.
"""
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import config
from src.ingestion.quantitative import QuantitativeIngestionEngine
from src.ingestion.qualitative import QualitativeIngestionEngine
from src.database.schema import DatabaseManager
from src.ml.xgb_model import XGBoostModel
from src.ml.fusion import HybridFusionEngine, BankrollEngine, PredictionPipeline


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProjectAugoApp:
    """
    Main application class that orchestrates all Project Augo components.
    """
    
    def __init__(self):
        logger.info("Initializing Project Augo...")
        
        # Initialize engines
        self.quant_engine = QuantitativeIngestionEngine()
        self.qual_engine = QualitativeIngestionEngine()
        self.db_manager = DatabaseManager()
        self.ml_model = XGBoostModel()
        self.fusion_engine = HybridFusionEngine()
        self.bankroll_engine = BankrollEngine()
        self.prediction_pipeline = PredictionPipeline(
            fusion_engine=self.fusion_engine,
            bankroll_engine=self.bankroll_engine
        )
        
        logger.info("All engines initialized successfully")
    
    def setup_database(self):
        """Initialize database schema."""
        logger.info("Setting up database...")
        self.db_manager.initialize_schema()
        logger.info("Database setup complete")
    
    def ingest_quantitative_data(self, seasons: list = None):
        """Fetch and store quantitative match data."""
        if seasons is None:
            seasons = ['2425', '2324', '2223']  # Default: last 3 seasons
        
        logger.info(f"Fetching quantitative data for seasons: {seasons}")
        
        # Fetch data
        df = self.quant_engine.fetch_multiple_seasons(seasons)
        
        # Clean and standardize
        df_clean = self.quant_engine.clean_and_standardize(df)
        
        # Select features
        df_final = self.quant_engine.select_features(df_clean)
        
        logger.info(f"Fetched {len(df_final)} matches total")
        
        # Save to file for later use
        output_path = project_root / "data" / "matches_processed.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.write_csv(output_path)
        
        logger.info(f"Saved processed data to {output_path}")
        
        return df_final
    
    def ingest_qualitative_data(self, max_articles: int = 60):
        """Fetch RSS articles and process with LLM."""
        logger.info("Fetching RSS articles...")
        
        # Fetch articles
        articles = self.qual_engine.fetch_rss_feeds(max_entries_per_feed=max_articles // 3)
        logger.info(f"Fetched {len(articles)} articles")
        
        if not articles:
            logger.warning("No articles fetched. Check RSS feed availability.")
            return []
        
        # Process with LLM
        logger.info("Processing articles with Ollama LLM...")
        signals = self.qual_engine.process_articles(articles)
        
        logger.info(f"Successfully processed {len(signals)} articles")
        
        # Save signals
        output_path = project_root / "data" / "llm_signals.json"
        self.qual_engine.save_signals_to_json(signals, output_path)
        
        return signals
    
    def train_model(self, df: dict):
        """Train the ML model with walk-forward validation."""
        logger.info("Training XGBoost model with walk-forward validation...")
        
        # Run walk-forward validation
        metrics = self.ml_model.train_walk_forward(df)
        
        logger.info(f"Walk-forward validation complete")
        logger.info(f"Mean Brier Score: {metrics['mean_brier_score']:.4f}")
        logger.info(f"Mean Log Loss: {metrics['mean_log_loss']:.4f}")
        logger.info(f"Mean Accuracy: {metrics['mean_accuracy']:.4f}")
        
        # Train final model on all data
        logger.info("Training final model on all data...")
        self.ml_model.fit_final_model(df)
        
        # Save model
        model_path = project_root / "models" / "xgb_model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.ml_model.save_model(model_path)
        
        logger.info(f"Model saved to {model_path}")
        
        return metrics
    
    def generate_predictions(self, df: dict, llm_signals: list = None):
        """Generate predictions for upcoming matches."""
        logger.info("Generating predictions...")
        
        # Get XGBoost probabilities
        xgb_probs = self.ml_model.predict_proba(df)
        
        # Convert LLM signals to team-based dict
        team_signals = {}
        if llm_signals:
            for signal in llm_signals:
                for team in signal.teams_mentioned:
                    if team not in team_signals:
                        team_signals[team] = {
                            'key_absences_impact': signal.key_absences_impact,
                            'fatigue_rotation_risk': signal.fatigue_rotation_risk,
                            'morale_sentiment_score': signal.morale_sentiment_score,
                            'confidence_score': signal.confidence_score
                        }
        
        # Prepare matches data
        matches = df.to_dicts()
        
        # Generate full predictions
        predictions = self.prediction_pipeline.generate_predictions(
            matches=matches,
            xgb_probs=xgb_probs,
            llm_signals=team_signals
        )
        
        # Summary
        bets_recommended = sum(1 for p in predictions if p.has_positive_edge)
        logger.info(f"Generated {len(predictions)} predictions, {bets_recommended} bets recommended")
        
        return predictions
    
    def run_full_pipeline(self):
        """Execute the complete pipeline from data ingestion to predictions."""
        logger.info("=" * 60)
        logger.info("PROJECT AUGO - FULL PIPELINE EXECUTION")
        logger.info("=" * 60)
        
        # Step 1: Setup database
        self.setup_database()
        
        # Step 2: Ingest quantitative data
        df = self.ingest_quantitative_data()
        
        # Step 3: Ingest qualitative data
        llm_signals = self.ingest_qualitative_data()
        
        # Step 4: Train model
        self.train_model(df)
        
        # Step 5: Generate predictions
        predictions = self.generate_predictions(df, llm_signals)
        
        # Step 6: Export to Telegram format
        telegram_msg = self.prediction_pipeline.export_to_telegram_format(predictions)
        print("\n" + telegram_msg)
        
        # Step 7: Performance summary
        perf_summary = self.bankroll_engine.get_performance_summary()
        logger.info("=" * 60)
        logger.info("BANKROLL PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        for key, value in perf_summary.items():
            logger.info(f"{key}: {value}")
        
        logger.info("Pipeline execution complete!")
        
        return predictions


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Project Augo - EPL Betting Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --pipeline          # Run full pipeline
  python main.py --train             # Train model only
  python main.py --predict           # Generate predictions only
  python main.py --gui               # Launch GUI
        """
    )
    
    parser.add_argument('--pipeline', action='store_true',
                        help='Run full end-to-end pipeline')
    parser.add_argument('--train', action='store_true',
                        help='Train the ML model')
    parser.add_argument('--predict', action='store_true',
                        help='Generate predictions')
    parser.add_argument('--gui', action='store_true',
                        help='Launch the GUI application')
    parser.add_argument('--seasons', nargs='+', default=None,
                        help='Season codes to fetch (e.g., 2425 2324)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    app = ProjectAugoApp()
    
    if args.pipeline:
        app.run_full_pipeline()
    elif args.gui:
        from src.gui.app import run_gui
        run_gui(
            quant_engine=app.quant_engine,
            qual_engine=app.qual_engine,
            ml_model=app.ml_model,
            prediction_pipeline=app.prediction_pipeline
        )
    elif args.train:
        df = app.ingest_quantitative_data(args.seasons)
        app.train_model(df)
    elif args.predict:
        # Load existing data and generate predictions
        from src.ingestion.quantitative import QuantitativeIngestionEngine
        quant_engine = QuantitativeIngestionEngine()
        df = quant_engine.load_from_csv(project_root / "data" / "matches_processed.csv")
        app.generate_predictions(df)
    else:
        # Default: show help
        parser.print_help()


if __name__ == "__main__":
    main()
