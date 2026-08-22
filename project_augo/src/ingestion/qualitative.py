"""
Qualitative Data Ingestion Engine
RSS feed aggregation + Local LLM parsing via Ollama
Windows-optimized with Pydantic validation
"""
import requests
import feedparser
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import logging
import json

from schemas.qualitative import QualitativeSignal, LLMBatchResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualitativeIngestionEngine:
    """
    Aggregates football news from RSS feeds and parses with local Ollama LLM.
    
    Extracts structured metrics:
    - key_absences_impact (0-10)
    - fatigue_rotation_risk (0-10)  
    - morale_sentiment_score (-5 to +5)
    - tactical_summary
    """
    
    # Default RSS feeds (can be overridden)
    DEFAULT_FEEDS = [
        "https://feeds.bbci.co.uk/sport/football/rss.xml",
        "https://www.theguardian.com/football/rss",
        "https://www.skysports.com/rss/12040",
    ]
    
    # LLM prompt template for consistent extraction
    LLM_PROMPT_TEMPLATE = """
You are an expert football analyst. Analyze the following news article and extract structured data.

ARTICLE TITLE: {title}
ARTICLE CONTENT: {content}

Extract the following metrics as JSON:
1. key_absences_impact: Float 0.0-10.0 (0=no impact, 10=critical injuries/suspensions)
2. fatigue_rotation_risk: Float 0.0-10.0 (0=fresh team, 10=exhausted from fixture congestion)
3. morale_sentiment_score: Float -5.0 to +5.0 (-5=very negative, +5=very positive)
4. tactical_summary: String max 500 chars summarizing tactical insights
5. teams_mentioned: List of EPL team names mentioned
6. confidence_score: Float 0.0-1.0 (your confidence in this extraction)

Return ONLY valid JSON matching this schema. No markdown, no explanations.
"""

    def __init__(
        self, 
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "qwen:14b",
        ollama_timeout: int = 120
    ):
        """
        Initialize the qualitative ingestion engine.
        
        Args:
            ollama_base_url: URL of local Ollama instance
            ollama_model: Model name to use (e.g., "qwen:14b", "deepseek-r1:14b")
            ollama_timeout: Request timeout in seconds
        """
        self.ollama_base_url = ollama_base_url.rstrip('/')
        self.ollama_model = ollama_model
        self.ollama_timeout = ollama_timeout
        
    def fetch_articles(
        self, 
        rss_urls: Optional[List[str]] = None,
        hours_back: int = 72,
        max_articles: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent articles from RSS feeds.
        
        Args:
            rss_urls: List of RSS feed URLs (uses defaults if None)
            hours_back: Only fetch articles from last N hours
            max_articles: Maximum total articles to return
            
        Returns:
            List of article dictionaries with title, content, source, date
        """
        if rss_urls is None:
            rss_urls = self.DEFAULT_FEEDS
        
        all_articles = []
        cutoff_date = datetime.now() - timedelta(hours=hours_back)
        
        for url in rss_urls:
            try:
                logger.info(f"Fetching RSS feed: {url}")
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                
                feed = feedparser.parse(response.content)
                
                for entry in feed.entries[:20]:  # Limit per feed
                    # Parse publication date
                    published = self._parse_date(entry.get('published', ''))
                    
                    if published and published >= cutoff_date:
                        article = {
                            'title': entry.get('title', 'No title'),
                            'content': entry.get('summary', entry.get('description', '')),
                            'source': feed.feed.get('title', 'Unknown'),
                            'published_date': published,
                            'link': entry.get('link', '')
                        }
                        
                        # Clean content (remove HTML tags)
                        article['content'] = self._clean_html(article['content'])
                        
                        all_articles.append(article)
                        
                        if len(all_articles) >= max_articles:
                            break
                            
            except Exception as e:
                logger.warning(f"Failed to fetch feed {url}: {e}")
        
        # Sort by date descending
        all_articles.sort(key=lambda x: x['published_date'], reverse=True)
        
        logger.info(f"✓ Fetched {len(all_articles)} articles from {len(rss_urls)} feeds")
        return all_articles[:max_articles]
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various RSS date formats"""
        formats = [
            "%a, %d %b %Y %H:%M:%S %Z",
            "%a, %d %b %Y %H:%M:%S %z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
            "%d %b %Y %H:%M:%S",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        
        # Fallback: return current time if parsing fails
        logger.warning(f"Could not parse date: {date_str}")
        return datetime.now()
    
    def _clean_html(self, text: str) -> str:
        """Remove HTML tags and normalize whitespace"""
        import re
        # Remove HTML tags
        clean = re.sub(r'<[^>]+>', '', text)
        # Remove extra whitespace
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean
    
    def parse_article_with_llm(
        self, 
        article: Dict[str, Any],
        validate_response: bool = True
    ) -> Optional[QualitativeSignal]:
        """
        Send article to local Ollama LLM for structured extraction.
        
        Args:
            article: Article dictionary with title, content, etc.
            validate_response: If True, validate output against Pydantic schema
            
        Returns:
            QualitativeSignal object or None if parsing fails
        """
        # Build prompt
        prompt = self.LLM_PROMPT_TEMPLATE.format(
            title=article['title'],
            content=article['content'][:2000]  # Truncate very long articles
        )
        
        try:
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for consistent extraction
                        "num_predict": 500,
                    }
                },
                timeout=self.ollama_timeout
            )
            response.raise_for_status()
            
            llm_output = response.json().get('response', '')
            
            # Extract JSON from LLM response
            json_str = self._extract_json_from_response(llm_output)
            
            if not json_str:
                logger.warning(f"No valid JSON in LLM response for: {article['title']}")
                return None
            
            # Parse JSON
            parsed_data = json.loads(json_str)
            
            # Add metadata
            parsed_data['article_title'] = article['title']
            parsed_data['source'] = article['source']
            parsed_data['published_date'] = article['published_date'].isoformat()
            
            # Validate against Pydantic schema
            if validate_response:
                signal = QualitativeSignal(**parsed_data)
            else:
                signal = QualitativeSignal.model_validate(parsed_data)
            
            logger.info(f"✓ Parsed article: {article['title'][:50]}...")
            return signal
            
        except requests.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing article: {e}")
            return None
    
    def _extract_json_from_response(self, text: str) -> Optional[str]:
        """Extract JSON substring from LLM response (handles markdown wrapping)"""
        import re
        
        # Try to find JSON between braces
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        # Try to find JSON array
        array_match = re.search(r'\[[^\[\]]*\]', text, re.DOTALL)
        if array_match:
            return array_match.group(0)
        
        # Try removing markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Try again after cleaning
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        return None
    
    def process_batch(
        self,
        articles: List[Dict[str, Any]],
        skip_failed: bool = True
    ) -> LLMBatchResponse:
        """
        Process multiple articles through LLM in batch.
        
        Args:
            articles: List of article dictionaries
            skip_failed: If True, continue on failures; if False, raise on first error
            
        Returns:
            LLMBatchResponse with all signals and statistics
        """
        signals = []
        failed_count = 0
        
        for i, article in enumerate(articles):
            logger.info(f"Processing article {i+1}/{len(articles)}")
            
            signal = self.parse_article_with_llm(article)
            
            if signal:
                signals.append(signal)
            else:
                failed_count += 1
                if not skip_failed:
                    raise RuntimeError(f"Failed to parse article: {article['title']}")
        
        return LLMBatchResponse(
            signals=signals,
            total_articles=len(articles),
            successful_extractions=len(signals),
            failed_extractions=failed_count
        )
    
    def get_team_signals(
        self,
        team_name: str,
        days_back: int = 7
    ) -> List[QualitativeSignal]:
        """
        Get all qualitative signals for a specific team.
        
        Args:
            team_name: EPL team name (e.g., "Arsenal", "Manchester United")
            days_back: Number of days to look back
            
        Returns:
            List of QualitativeSignal objects mentioning the team
        """
        articles = self.fetch_articles(hours_back=days_back * 24)
        
        # Filter articles mentioning the team
        team_articles = [
            a for a in articles 
            if team_name.lower() in a['title'].lower() or team_name.lower() in a['content'].lower()
        ]
        
        logger.info(f"Found {len(team_articles)} articles mentioning {team_name}")
        
        batch_response = self.process_batch(team_articles)
        return batch_response.signals


# Example usage
if __name__ == "__main__":
    engine = QualitativeIngestionEngine()
    
    # Fetch recent articles
    articles = engine.fetch_articles(hours_back=24, max_articles=5)
    
    print(f"\nFetched {len(articles)} articles:")
    for article in articles:
        print(f"  - {article['title'][:60]}...")
    
    # Parse one article with LLM
    if articles:
        signal = engine.parse_article_with_llm(articles[0])
        if signal:
            print(f"\nLLM Extraction Result:")
            print(f"  Key Absences Impact: {signal.key_absences_impact}")
            print(f"  Fatigue Risk: {signal.fatigue_rotation_risk}")
            print(f"  Morale Score: {signal.morale_sentiment_score}")
            print(f"  Tactical Summary: {signal.tactical_summary}")
