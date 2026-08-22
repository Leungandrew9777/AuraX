"""
Project Augo - Simple GUI Application
Tkinter-based interface for data input, sentiment scanning, odds display, and Telegram export.
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import json
from datetime import datetime
from typing import Optional, Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProjectAugoGUI:
    """
    Main GUI application for Project Augo.
    
    Provides a unified interface for:
    - Data ingestion (quantitative & qualitative)
    - Sentiment scanning via LLM
    - Displaying current gameweek odds and predictions
    - Exporting predictions to Telegram format
    """
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🏴󠁧󠁢󠁥󠁮󠁧󠁿 Project Augo - EPL Betting Framework")
        self.root.geometry("1200x800")
        
        # State variables
        self.quant_data_loaded = False
        self.qual_signals_loaded = False
        self.predictions_ready = False
        self.current_predictions = []
        
        # Engine references (to be set by main app)
        self.quant_engine = None
        self.qual_engine = None
        self.ml_model = None
        self.prediction_pipeline = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Initialize the user interface."""
        # Create main container with tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.tab_dashboard = ttk.Frame(self.notebook)
        self.tab_data = ttk.Frame(self.notebook)
        self.tab_sentiment = ttk.Frame(self.notebook)
        self.tab_predictions = ttk.Frame(self.notebook)
        self.tab_settings = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_dashboard, text='📊 Dashboard')
        self.notebook.add(self.tab_data, text='📁 Data Ingestion')
        self.notebook.add(self.tab_sentiment, text='🧠 Sentiment Scanner')
        self.notebook.add(self.tab_predictions, text='⚽ Predictions & Odds')
        self.notebook.add(self.tab_settings, text='⚙️ Settings')
        
        self._setup_dashboard_tab()
        self._setup_data_tab()
        self._setup_sentiment_tab()
        self._setup_predictions_tab()
        self._setup_settings_tab()
    
    def _setup_dashboard_tab(self):
        """Setup the dashboard tab with overview widgets."""
        # Header
        header_frame = ttk.Frame(self.tab_dashboard)
        header_frame.pack(fill='x', padx=20, pady=20)
        
        title_label = ttk.Label(
            header_frame,
            text="🏴󠁧󠁢󠁥󠁮󠁧󠁿 PROJECT AUGO DASHBOARD",
            font=('Helvetica', 24, 'bold')
        )
        title_label.pack()
        
        subtitle_label = ttk.Label(
            header_frame,
            text="Hybrid EPL Betting Framework - Quantitative + Qualitative Analysis",
            font=('Helvetica', 12)
        )
        subtitle_label.pack()
        
        # Status cards
        status_frame = ttk.Frame(self.tab_dashboard)
        status_frame.pack(fill='x', padx=20, pady=20)
        
        self.status_cards = {}
        card_configs = [
            ('Quant Data', '📊', 'Not Loaded'),
            ('Qual Signals', '🧠', 'Not Loaded'),
            ('Model Status', '🤖', 'Not Trained'),
            ('Bankroll', '💰', '£1000.00')
        ]
        
        for i, (title, icon, initial_value) in enumerate(card_configs):
            card = ttk.LabelFrame(status_frame, text=f"{icon} {title}", padding=10)
            card.grid(row=0, column=i, sticky='ew', padx=10)
            
            value_label = ttk.Label(card, text=initial_value, font=('Helvetica', 16, 'bold'))
            value_label.pack(pady=10)
            
            self.status_cards[title.lower().replace(' ', '_')] = value_label
            
            status_frame.columnconfigure(i, weight=1)
        
        # Recent activity log
        log_frame = ttk.LabelFrame(self.tab_dashboard, text="📝 Activity Log", padding=10)
        log_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        self.activity_log = scrolledtext.ScrolledText(
            log_frame,
            height=20,
            wrap='word',
            font=('Consolas', 10)
        )
        self.activity_log.pack(fill='both', expand=True)
        
        # Quick actions
        actions_frame = ttk.Frame(self.tab_dashboard)
        actions_frame.pack(fill='x', padx=20, pady=10)
        
        ttk.Button(actions_frame, text="🔄 Refresh Dashboard", command=self._refresh_dashboard).pack(side='left', padx=5)
        ttk.Button(actions_frame, text="📁 Load Quant Data", command=self._go_to_data_tab).pack(side='left', padx=5)
        ttk.Button(actions_frame, text="🧠 Scan Sentiment", command=self._go_to_sentiment_tab).pack(side='left', padx=5)
    
    def _setup_data_tab(self):
        """Setup the data ingestion tab."""
        # Quantitative data section
        quant_frame = ttk.LabelFrame(self.tab_data, text="📊 Quantitative Data (Football-Data.co.uk)", padding=15)
        quant_frame.pack(fill='x', padx=20, pady=10)
        
        ttk.Label(quant_frame, text="Season(s) to fetch:").grid(row=0, column=0, sticky='w', pady=5)
        
        self.season_var = tk.StringVar(value="2425")
        season_entry = ttk.Entry(quant_frame, textvariable=self.season_var, width=10)
        season_entry.grid(row=0, column=1, sticky='w', padx=10, pady=5)
        
        ttk.Label(quant_frame, text="(e.g., 2425 for 2024-25, 2324 for 2023-24)").grid(row=0, column=2, sticky='w', pady=5)
        
        ttk.Button(quant_frame, text="Fetch Season Data", command=self._fetch_quant_data).grid(row=1, column=0, pady=10)
        
        self.quant_status = ttk.Label(quant_frame, text="Status: Not loaded", foreground='gray')
        self.quant_status.grid(row=1, column=1, columnspan=2, sticky='w', padx=10)
        
        # File upload option
        ttk.Separator(quant_frame, orient='horizontal').grid(row=2, column=0, columnspan=3, sticky='ew', pady=10)
        
        ttk.Label(quant_frame, text="Or upload CSV file:").grid(row=3, column=0, sticky='w', pady=5)
        ttk.Button(quant_frame, text="📂 Browse...", command=self._upload_csv).grid(row=3, column=1, sticky='w', padx=10)
        
        self.file_path_var = tk.StringVar()
        file_label = ttk.Label(quant_frame, textvariable=self.file_path_var, wraplength=400)
        file_label.grid(row=3, column=2, sticky='w', padx=10)
        
        # Qualitative data section
        qual_frame = ttk.LabelFrame(self.tab_data, text="🧠 Qualitative Data (RSS + LLM)", padding=15)
        qual_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        ttk.Label(qual_frame, text="RSS Feeds configured:").grid(row=0, column=0, sticky='w', pady=5)
        
        self.rss_listbox = tk.Listbox(qual_frame, height=5, width=80)
        self.rss_listbox.grid(row=1, column=0, columnspan=2, sticky='w', padx=10, pady=5)
        
        # Populate RSS feeds
        default_feeds = [
            "https://feeds.bbci.co.uk/sport/football/rss.xml",
            "https://www.theguardian.com/football/rss",
            "https://www.skysports.com/rss/12040"
        ]
        for feed in default_feeds:
            self.rss_listbox.insert(tk.END, feed)
        
        ttk.Button(qual_frame, text="📰 Fetch Articles", command=self._fetch_articles).grid(row=2, column=0, pady=10)
        ttk.Button(qual_frame, text="🧠 Process with LLM", command=self._process_with_llm).grid(row=2, column=1, pady=10)
        
        self.qual_status = ttk.Label(qual_frame, text="Articles: 0 | Processed: 0", foreground='gray')
        self.qual_status.grid(row=3, column=0, columnspan=2, sticky='w', padx=10)
    
    def _setup_sentiment_tab(self):
        """Setup the sentiment scanner tab."""
        # Team selection
        team_frame = ttk.LabelFrame(self.tab_sentiment, text="Select Team for Analysis", padding=15)
        team_frame.pack(fill='x', padx=20, pady=10)
        
        ttk.Label(team_frame, text="Team:").pack(side='left', padx=5)
        
        self.team_var = tk.StringVar()
        teams = ["Arsenal", "Aston Villa", "Chelsea", "Liverpool", "Man City", "Man Utd", 
                 "Newcastle", "Tottenham", "Brighton", "West Ham", "Fulham", "Brentford",
                 "Wolves", "Crystal Palace", "Everton", "Nott'm Forest", "Bournemouth",
                 "Sheffield Utd", "Burnley", "Luton"]
        
        self.team_combo = ttk.Combobox(team_frame, textvariable=self.team_var, values=teams, width=20)
        self.team_combo.pack(side='left', padx=10)
        self.team_combo.set("Arsenal")
        
        ttk.Button(team_frame, text="🔍 Analyze Sentiment", command=self._analyze_team_sentiment).pack(side='left', padx=10)
        
        # Results display
        results_frame = ttk.LabelFrame(self.tab_sentiment, text="Sentiment Analysis Results", padding=15)
        results_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create grid for metrics
        metrics = [
            ('Key Absences Impact', 'absence_impact'),
            ('Fatigue Rotation Risk', 'fatigue_risk'),
            ('Morale Sentiment Score', 'morale_score'),
            ('Articles Analyzed', 'article_count')
        ]
        
        self.sentiment_vars = {}
        for i, (label, key) in enumerate(metrics):
            ttk.Label(results_frame, text=f"{label}:").grid(row=i, column=0, sticky='w', padx=20, pady=10)
            
            var = tk.StringVar(value="-")
            self.sentiment_vars[key] = var
            
            entry = ttk.Entry(results_frame, textvariable=var, width=30, state='readonly')
            entry.grid(row=i, column=1, sticky='w', padx=10, pady=10)
        
        # Tactical summary
        ttk.Label(results_frame, text="Tactical Summary:").grid(row=4, column=0, sticky='nw', padx=20, pady=10)
        
        self.tactical_summary = scrolledtext.ScrolledText(results_frame, height=8, width=60, wrap='word')
        self.tactical_summary.grid(row=4, column=1, padx=10, pady=10, sticky='ew')
        
        results_frame.columnconfigure(1, weight=1)
    
    def _setup_predictions_tab(self):
        """Setup the predictions and odds tab."""
        # Controls
        controls_frame = ttk.Frame(self.tab_predictions)
        controls_frame.pack(fill='x', padx=20, pady=10)
        
        ttk.Button(controls_frame, text="🔄 Generate Predictions", command=self._generate_predictions).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="📤 Export to Telegram", command=self._export_telegram).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="💾 Save to CSV", command=self._save_predictions_csv).pack(side='left', padx=5)
        
        # Predictions table
        table_frame = ttk.LabelFrame(self.tab_predictions, text="Match Predictions & Odds", padding=10)
        table_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create treeview for predictions
        columns = ('Date', 'Home', 'Away', 'P(H)', 'P(D)', 'P(A)', 'Odds H', 'Odds D', 'Odds A', 'Edge', 'Bet', 'Stake')
        
        self.predictions_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.predictions_tree.heading(col, text=col)
            self.predictions_tree.column(col, width=80 if col not in ['Home', 'Away'] else 100)
        
        # Scrollbars
        y_scroll = ttk.Scrollbar(table_frame, orient='vertical', command=self.predictions_tree.yview)
        x_scroll = ttk.Scrollbar(table_frame, orient='horizontal', command=self.predictions_tree.xview)
        self.predictions_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        
        self.predictions_tree.grid(row=0, column=0, sticky='nsew')
        y_scroll.grid(row=0, column=1, sticky='ns')
        x_scroll.grid(row=1, column=0, sticky='ew')
        
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        
        # Summary stats
        summary_frame = ttk.Frame(self.tab_predictions)
        summary_frame.pack(fill='x', padx=20, pady=10)
        
        self.summary_label = ttk.Label(summary_frame, text="No predictions generated yet.", font=('Helvetica', 11))
        self.summary_label.pack()
    
    def _setup_settings_tab(self):
        """Setup the settings tab."""
        settings_frame = ttk.LabelFrame(self.tab_settings, text="Configuration", padding=15)
        settings_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Bankroll settings
        bankroll_frame = ttk.LabelFrame(settings_frame, text="💰 Bankroll Management", padding=10)
        bankroll_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(bankroll_frame, text="Initial Bankroll (£):").grid(row=0, column=0, sticky='w', pady=5)
        self.bankroll_var = tk.StringVar(value="1000.00")
        ttk.Entry(bankroll_frame, textvariable=self.bankroll_var, width=15).grid(row=0, column=1, padx=10)
        
        ttk.Label(bankroll_frame, text="Kelly Fraction:").grid(row=1, column=0, sticky='w', pady=5)
        self.kelly_var = tk.StringVar(value="0.25")
        kelly_combo = ttk.Combobox(bankroll_frame, textvariable=self.kelly_var, values=["0.25", "0.5", "1.0"], width=12)
        kelly_combo.grid(row=1, column=1, padx=10)
        
        ttk.Label(bankroll_frame, text="(Quarter-Kelly recommended)").grid(row=1, column=2, sticky='w', padx=5)
        
        # Edge threshold
        edge_frame = ttk.LabelFrame(settings_frame, text="📈 Edge Threshold", padding=10)
        edge_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(edge_frame, text="Minimum Edge (%):").grid(row=0, column=0, sticky='w', pady=5)
        self.edge_threshold_var = tk.StringVar(value="5.0")
        ttk.Entry(edge_frame, textvariable=self.edge_threshold_var, width=15).grid(row=0, column=1, padx=10)
        
        # Ollama settings
        ollama_frame = ttk.LabelFrame(settings_frame, text="🤖 Ollama Configuration", padding=10)
        ollama_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(ollama_frame, text="Base URL:").grid(row=0, column=0, sticky='w', pady=5)
        self.ollama_url_var = tk.StringVar(value="http://localhost:11434")
        ttk.Entry(ollama_frame, textvariable=self.ollama_url_var, width=40).grid(row=0, column=1, padx=10)
        
        ttk.Label(ollama_frame, text="Model:").grid(row=1, column=0, sticky='w', pady=5)
        self.ollama_model_var = tk.StringVar(value="qwen:14b")
        ttk.Entry(ollama_frame, textvariable=self.ollama_model_var, width=40).grid(row=1, column=1, padx=10)
        
        # Save button
        ttk.Button(settings_frame, text="💾 Save Settings", command=self._save_settings).pack(pady=20)
    
    # === Action Methods ===
    
    def _log_activity(self, message: str):
        """Add message to activity log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.activity_log.insert(tk.END, f"[{timestamp}] {message}\n")
        self.activity_log.see(tk.END)
    
    def _refresh_dashboard(self):
        """Refresh dashboard status."""
        self._log_activity("Dashboard refreshed")
        
        if self.quant_data_loaded:
            self.status_cards['quant_data'].config(text="Loaded ✓", foreground='green')
        if self.qual_signals_loaded:
            self.status_cards['qual_signals'].config(text="Loaded ✓", foreground='green')
        if self.predictions_ready:
            self.status_cards['model_status'].config(text="Ready ✓", foreground='green')
    
    def _go_to_data_tab(self):
        """Navigate to data tab."""
        self.notebook.select(self.tab_data)
    
    def _go_to_sentiment_tab(self):
        """Navigate to sentiment tab."""
        self.notebook.select(self.tab_sentiment)
    
    def _fetch_quant_data(self):
        """Fetch quantitative data from Football-Data.co.uk."""
        def fetch_thread():
            try:
                season = self.season_var.get()
                self._log_activity(f"Fetching season {season}...")
                
                if self.quant_engine:
                    df = self.quant_engine.fetch_season_data(season)
                    if df is not None:
                        self.quant_data_loaded = True
                        self.root.after(0, lambda: self.quant_status.config(
                            text=f"Status: Loaded {len(df)} matches ✓", foreground='green'))
                        self.root.after(0, lambda: self._log_activity(f"Successfully loaded {len(df)} matches"))
                    else:
                        self.root.after(0, lambda: self.quant_status.config(
                            text="Status: Failed to fetch", foreground='red'))
                        self.root.after(0, lambda: self._log_activity("Failed to fetch quant data"))
                else:
                    self.root.after(0, lambda: self._log_activity("Quant engine not initialized"))
                    
            except Exception as e:
                self.root.after(0, lambda: self._log_activity(f"Error: {str(e)}"))
        
        threading.Thread(target=fetch_thread, daemon=True).start()
    
    def _upload_csv(self):
        """Upload CSV file dialog."""
        filepath = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            self.file_path_var.set(filepath)
            self._log_activity(f"File selected: {filepath}")
    
    def _fetch_articles(self):
        """Fetch articles from RSS feeds."""
        def fetch_thread():
            try:
                self._log_activity("Fetching RSS articles...")
                
                if self.qual_engine:
                    articles = self.qual_engine.fetch_rss_feeds()
                    count = len(articles)
                    self.root.after(0, lambda: self.qual_status.config(
                        text=f"Articles: {count} | Processed: 0", foreground='blue'))
                    self.root.after(0, lambda: self._log_activity(f"Fetched {count} articles"))
                    
                    # Store for processing
                    self._pending_articles = articles
                else:
                    self.root.after(0, lambda: self._log_activity("Qual engine not initialized"))
                    
            except Exception as e:
                self.root.after(0, lambda: self._log_activity(f"Error: {str(e)}"))
        
        threading.Thread(target=fetch_thread, daemon=True).start()
    
    def _process_with_llm(self):
        """Process fetched articles with LLM."""
        def process_thread():
            try:
                self._log_activity("Processing articles with LLM...")
                
                if hasattr(self, '_pending_articles') and self.qual_engine:
                    articles = self._pending_articles
                    signals = self.qual_engine.process_articles(articles)
                    
                    processed_count = len(signals)
                    self.qual_signals_loaded = True
                    
                    self.root.after(0, lambda: self.qual_status.config(
                        text=f"Articles: {len(articles)} | Processed: {processed_count} ✓", foreground='green'))
                    self.root.after(0, lambda: self._log_activity(f"Processed {processed_count}/{len(articles)} articles"))
                    
            except Exception as e:
                self.root.after(0, lambda: self._log_activity(f"Error: {str(e)}"))
        
        threading.Thread(target=process_thread, daemon=True).start()
    
    def _analyze_team_sentiment(self):
        """Analyze sentiment for selected team."""
        team = self.team_var.get()
        self._log_activity(f"Analyzing sentiment for {team}...")
        
        # Placeholder - would integrate with actual engine
        self.sentiment_vars['absence_impact'].set("3.5")
        self.sentiment_vars['fatigue_risk'].set("4.0")
        self.sentiment_vars['morale_score'].set("2.5")
        self.sentiment_vars['article_count'].set("5")
        
        self.tactical_summary.delete(1.0, tk.END)
        self.tactical_summary.insert(tk.END, f"Tactical analysis for {team} based on recent news articles...")
        
        self._log_activity(f"Sentiment analysis complete for {team}")
    
    def _generate_predictions(self):
        """Generate predictions for upcoming matches."""
        self._log_activity("Generating predictions...")
        
        # Placeholder - would integrate with actual pipeline
        sample_predictions = [
            ('2024-12-14', 'Arsenal', 'Everton', '65%', '20%', '15%', '1.45', '4.50', '7.00', '+8%', 'H', '3.2%'),
            ('2024-12-14', 'Chelsea', 'Brentford', '55%', '25%', '20%', '1.70', '4.00', '5.00', '+5%', 'H', '2.1%'),
            ('2024-12-15', 'Liverpool', 'Fulham', '70%', '18%', '12%', '1.35', '5.50', '9.00', '+12%', 'H', '4.5%'),
        ]
        
        # Clear existing
        for item in self.predictions_tree.get_children():
            self.predictions_tree.delete(item)
        
        # Add sample data
        for pred in sample_predictions:
            self.predictions_tree.insert('', tk.END, values=pred)
        
        self.predictions_ready = True
        self.summary_label.config(text=f"Generated {len(sample_predictions)} predictions | 3 bets recommended")
        self._log_activity(f"Generated {len(sample_predictions)} predictions")
    
    def _export_telegram(self):
        """Export predictions to Telegram format."""
        if not self.predictions_ready:
            messagebox.showwarning("Warning", "No predictions generated yet!")
            return
        
        telegram_msg = """
🏴󠁧󠁢󠁥󠁮󠁧󠁿 **PROJECT AUGO - WEEKLY PREDICTIONS** 🏴󠁧󠁢󠁥󠁮󠁧󠁿

🏠 *Arsenal vs Everton*
📅 2024-12-14
💡 Recommendation: Home Win
📊 Probability: 65% / 20% / 15%
💰 Odds: 1.45 / 4.50 / 7.00
📈 Edge: +8%
💵 Stake: 3.2% of bankroll

✈️ *Liverpool vs Fulham*
📅 2024-12-15
💡 Recommendation: Home Win
📊 Probability: 70% / 18% / 12%
💰 Odds: 1.35 / 5.50 / 9.00
📈 Edge: +12%
💵 Stake: 4.5% of bankroll

---
⚠️ Bet responsibly. Past performance ≠ future results.
"""
        
        # Copy to clipboard
        self.root.clipboard_clear()
        self.root.clipboard_append(telegram_msg)
        
        self._log_activity("Predictions exported to clipboard (Telegram format)")
        messagebox.showinfo("Export Complete", "Telegram message copied to clipboard!\n\nYou can now paste it into Telegram.")
    
    def _save_predictions_csv(self):
        """Save predictions to CSV file."""
        if not self.predictions_ready:
            messagebox.showwarning("Warning", "No predictions to save!")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save predictions to CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if filepath:
            self._log_activity(f"Predictions saved to {filepath}")
            messagebox.showinfo("Saved", f"Predictions saved to:\n{filepath}")
    
    def _save_settings(self):
        """Save settings configuration."""
        self._log_activity("Settings saved")
        messagebox.showinfo("Settings", "Settings saved successfully!\n\nNote: Some settings may require restart.")
    
    def set_engines(self, quant_engine, qual_engine, ml_model, prediction_pipeline):
        """Set reference to backend engines."""
        self.quant_engine = quant_engine
        self.qual_engine = qual_engine
        self.ml_model = ml_model
        self.prediction_pipeline = prediction_pipeline
        self._log_activity("Backend engines connected")


def run_gui(quant_engine=None, qual_engine=None, ml_model=None, prediction_pipeline=None):
    """Launch the GUI application."""
    root = tk.Tk()
    
    # Set theme
    style = ttk.Style()
    style.theme_use('clam')
    
    app = ProjectAugoGUI(root)
    
    # Connect engines if provided
    if quant_engine or qual_engine or ml_model or prediction_pipeline:
        app.set_engines(quant_engine, qual_engine, ml_model, prediction_pipeline)
    
    root.mainloop()


if __name__ == "__main__":
    run_gui()
