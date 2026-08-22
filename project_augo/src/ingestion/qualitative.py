"""
Project Augo - Qualitative Data Ingestion Engine
RSS Feed aggregation and LLM-based sentiment extraction via Ollama.
"""
import feedparser
import requests
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

from config.settings import config
from schemas.qualitative import QualitativeSignal, LLMPromptTemplate


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualitativeIngestionEngine:
    """
    Engine for fetching news from RSS feeds and extracting
    quantitative metrics using a local Ollama LLM instance.
    """
    
    def __init__(self, ollama_base_url: str = None, model_name: str = None):
        self.ollama_base_url = ollama_base_url or config.ollama.base_url
        self.model_name = model_name or config.ollama.model_name
        self.timeout = config.ollama.timeout
        self.rss_feeds = config.data_sources.rss_feeds
        
        # Simple in-memory cache to avoid re-processing articles
        self._processed_articles: Dict[str, QualitativeSignal] = {}
    
    def fetch_rss_feeds(self, max_entries_per_feed: int = 20) -> List[Dict]:
        """
        Fetch articles from all configured RSS feeds.
        
        Args:
            max_entries_per_feed: Maximum number of articles to fetch per feed
            
        Returns:
            List of article dictionaries with title, content, source, etc.
        """
        all_articles = []
        
        for feed_url in self.rss_feeds:
            logger.info(f"Fetching RSS feed: {feed_url}")
            try:
                feed = feedparser.parse(feed_url)
                
                if feed.bozo:
                    logger.warning(f"Malformed RSS feed: {feed_url}")
                    continue
                
                entries = feed.entries[:max_entries_per_feed]
                
                for entry in entries:
                    article = {
                        'title': entry.get('title', 'No Title'),
                        'content': self._extract_content(entry),
                        'published_at': self._parse_published_date(entry),
                        'source': feed.feed.get('title', 'Unknown Source'),
                        'link': entry.get('link', ''),
                        'feed_url': feed_url
                    }
                    
                    # Only include if we have content
                    if article['content']:
                        all_articles.append(article)
                        
            except Exception as e:
                logger.error(f"Error fetching RSS feed {feed_url}: {e}")
        
        logger.info(f"Fetched {len(all_articles)} total articles from {len(self.rss_feeds)} feeds")
        return all_articles
    
    def _extract_content(self, entry: Dict) -> str:
        """Extract clean text content from RSS entry."""
        # Try different content fields
        content_fields = ['content', 'summary', 'description']
        
        for field in content_fields:
            if field in entry:
                value = entry[field]
                # Handle list of dicts (common in RSS)
                if isinstance(value, list) and len(value) > 0:
                    value = value[0].get('value', '')
                
                # Strip HTML tags (simple approach)
                if isinstance(value, str):
                    # Remove HTML tags
                    import re
                    clean_text = re.sub(r'<[^>]+>', '', value)
                    return clean_text.strip()
        
        return ""
    
    def _parse_published_date(self, entry: Dict) -> datetime:
        """Parse published date from RSS entry."""
        date_fields = ['published_parsed', 'updated_parsed', 'created_parsed']
        
        for field in date_fields:
            if field in entry and entry[field]:
                try:
                    return datetime(*entry[field][:6])
                except (TypeError, ValueError):
                    continue
        
        return datetime.now()
    
    def _generate_article_id(self, title: str, source: str, published_at: datetime) -> str:
        """Generate unique ID for an article."""
        content = f"{title}{source}{published_at.isoformat()}"
        return f"{source.lower().replace(' ', '_')}_{hashlib.md5(content.encode()).hexdigest()[:12]}"
    
    def query_ollama(self, article: Dict) -> Optional[QualitativeSignal]:
        """
        Send article to local Ollama instance for analysis.
        
        Args:
            article: Article dictionary with title, content, etc.
            
        Returns:
            QualitativeSignal object if successful, None otherwise
        """
        # Build prompt
        prompt = LLMPromptTemplate.build_prompt(
            title=article['title'],
            content=article['content'],
            published_at=article['published_at'].isoformat(),
            source=article['source']
        )
        
        # Prepare request to Ollama API
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": LLMPromptTemplate.SYSTEM_PROMPT,
            "stream": False,
            "format": "json",  # Request JSON output
            "options": {
                "temperature": 0.1,  # Low temperature for consistent extraction
                "top_p": 0.9
            }
        }
        
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            llm_output = result.get('response', '')
            
            # Parse JSON from LLM output
            try:
                # Clean up potential markdown formatting
                llm_output = llm_output.strip()
                if llm_output.startswith('```json'):
                    llm_output = llm_output[7:]
                if llm_output.endswith('```'):
                    llm_output = llm_output[:-3]
                
                parsed_json = json.loads(llm_output.strip())
                
                # Add metadata
                article_id = self._generate_article_id(
                    article['title'],
                    article['source'],
                    article['published_at']
                )
                
                parsed_json['article_id'] = article_id
                parsed_json['source'] = article['source']
                parsed_json['published_at'] = article['published_at'].isoformat()
                parsed_json['raw_json_payload'] = parsed_json.copy()
                
                # Validate against Pydantic schema
                signal = QualitativeSignal(**parsed_json)
                return signal
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM JSON output: {e}")
                logger.debug(f"Raw output: {llm_output}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error querying Ollama: {e}")
            return None
    
    def process_articles(self, articles: List[Dict]) -> List[QualitativeSignal]:
        """
        Process multiple articles through the LLM.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            List of successfully processed QualitativeSignal objects
        """
        signals = []
        
        for i, article in enumerate(articles):
            logger.info(f"Processing article {i+1}/{len(articles)}: {article['title'][:50]}...")
            
            # Check cache
            cache_key = f"{article['title']}_{article['source']}"
            if cache_key in self._processed_articles:
                logger.debug(f"Using cached result for: {cache_key}")
                signals.append(self._processed_articles[cache_key])
                continue
            
            # Query LLM
            signal = self.query_ollama(article)
            
            if signal is not None:
                signals.append(signal)
                self._processed_articles[cache_key] = signal
                logger.info(f"Successfully extracted signal for: {article['title'][:30]}...")
            else:
                logger.warning(f"Failed to extract signal for: {article['title'][:30]}...")
        
        logger.info(f"Successfully processed {len(signals)}/{len(articles)} articles")
        return signals
    
    def extract_team_signals(self, signals: List[QualitativeSignal], 
                            team_name: str) -> Dict[str, float]:
        """
        Aggregate qualitative signals for a specific team.
        
        Args:
            signals: List of QualitativeSignal objects
            team_name: Team to filter by
            
        Returns:
            Dictionary with aggregated metrics for the team
        """
        team_signals = [s for s in signals if team_name in s.teams_mentioned]
        
        if not team_signals:
            return {
                'key_absences_impact': 0.0,
                'fatigue_rotation_risk': 0.0,
                'morale_sentiment_score': 0.0,
                'article_count': 0
            }
        
        # Weight by confidence and recency
        weights = [s.confidence_score for s in team_signals]
        total_weight = sum(weights)
        
        if total_weight == 0:
            total_weight = len(team_signals)  # Equal weights if no confidence scores
        
        aggregated = {
            'key_absences_impact': sum(s.key_absences_impact * s.confidence_score for s in team_signals) / total_weight,
            'fatigue_rotation_risk': sum(s.fatigue_rotation_risk * s.confidence_score for s in team_signals) / total_weight,
            'morale_sentiment_score': sum(s.morale_sentiment_score * s.confidence_score for s in team_signals) / total_weight,
            'article_count': len(team_signals)
        }
        
        return {k: round(v, 2) for k, v in aggregated.items()}
    
    def get_all_teams_from_signals(self, signals: List[QualitativeSignal]) -> List[str]:
        """Extract all unique team names mentioned in signals."""
        teams = set()
        for signal in signals:
            teams.update(signal.teams_mentioned)
        return sorted(list(teams))
    
    def save_signals_to_json(self, signals: List[QualitativeSignal], filepath: str) -> None:
        """Save signals to JSON file for debugging/audit."""
        data = [s.model_dump() for s in signals]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved {len(signals)} signals to {filepath}")
    
    def load_signals_from_json(self, filepath: str) -> List[QualitativeSignal]:
        """Load signals from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        signals = [QualitativeSignal(**item) for item in data]
        logger.info(f"Loaded {len(signals)} signals from {filepath}")
        return signals
