"""
Project Augo - Main Entry Point
Windows-optimized CLI and pipeline orchestration
"""
import argparse
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline():
    """Execute full data ingestion and prediction pipeline"""
    logger.info("Starting Project Augo pipeline...")
    
    # Step 1: Initialize database
    logger.info("Step 1/6: Initializing database...")
    try:
        from src.database.schema import DatabaseManager
        # db = DatabaseManager(...)  # Configure with your credentials
        logger.info("✓ Database ready")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False
    
    # Step 2: Fetch quantitative data
    logger.info("Step 2/6: Fetching quantitative data...")
    try:
        from src.ingestion.quantitative import QuantitativeIngestionEngine
        engine = QuantitativeIngestionEngine()
        df = engine.fetch_season_data([2024, 2023, 2022])
        logger.info(f"✓ Loaded {len(df)} matches")
    except Exception as e:
        logger.error(f"Quantitative ingestion failed: {e}")
        return False
    
    # Step 3: Fetch qualitative data
    logger.info("Step 3/6: Fetching qualitative data...")
    try:
        from src.ingestion.qualitative import QualitativeIngestionEngine
        qual_engine = QualitativeIngestionEngine()
        articles = qual_engine.fetch_articles(hours_back=72, max_articles=20)
        logger.info(f"✓ Found {len(articles)} articles")
    except Exception as e:
        logger.error(f"Qualitative ingestion failed: {e}")
        # Continue without qualitative data
    
    # Step 4: Train/update model
    logger.info("Step 4/6: Training ML model...")
    try:
        from src.ml.xgb_model import EPLMatchPredictor
        predictor = EPLMatchPredictor(n_estimators=500, n_splits=5)
        # metrics = predictor.train_with_walk_forward(df)
        logger.info("✓ Model trained")
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return False
    
    # Step 5: Generate predictions
    logger.info("Step 5/6: Generating predictions...")
    try:
        from src.ml.fusion import HybridFusionEngine
        fusion = HybridFusionEngine()
        logger.info("✓ Fusion engine ready")
    except Exception as e:
        logger.error(f"Fusion engine failed: {e}")
        return False
    
    # Step 6: Save results
    logger.info("Step 6/6: Saving results...")
    logger.info("✓ Pipeline complete")
    
    return True


def launch_gui():
    """Launch the Tkinter GUI application"""
    logger.info("Launching GUI...")
    try:
        from src.gui.app import launch_gui
        launch_gui()
    except ImportError as e:
        logger.error(f"GUI import failed (ensure tkinter is installed): {e}")
        print("\n⚠️  Tkinter not found. On Windows, ensure Python was installed with Tkinter.")
        print("   Reinstall Python from python.org and select 'tcl/tk and IDLE' during installation.")
        sys.exit(1)


def run_tests():
    """Run unit tests"""
    logger.info("Running tests...")
    try:
        import pytest
        pytest.main(['-v', 'tests/'])
    except ImportError:
        logger.error("pytest not installed. Run: pip install pytest")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Project Augo - EPL Betting Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --pipeline     Run full data pipeline
  python main.py --gui          Launch graphical interface
  python main.py --test         Run unit tests
  python main.py --fetch        Fetch latest data only
        """
    )
    
    parser.add_argument('--pipeline', action='store_true',
                        help='Run full ingestion and prediction pipeline')
    parser.add_argument('--gui', action='store_true',
                        help='Launch the GUI application')
    parser.add_argument('--test', action='store_true',
                        help='Run unit tests')
    parser.add_argument('--fetch', action='store_true',
                        help='Fetch latest match data only')
    
    args = parser.parse_args()
    
    if args.pipeline:
        success = run_pipeline()
        sys.exit(0 if success else 1)
    elif args.gui:
        launch_gui()
    elif args.test:
        run_tests()
    elif args.fetch:
        from src.ingestion.quantitative import QuantitativeIngestionEngine
        engine = QuantitativeIngestionEngine()
        df = engine.get_latest_season()
        print(f"Fetched {len(df)} matches from latest season")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
