"""
Pydantic schemas for LLM output validation
Ensures strict JSON structure from Ollama responses
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime


class QualitativeSignal(BaseModel):
    """
    Schema for LLM-parsed qualitative metrics from news/articles.
    All fields are validated against strict bounds.
    """
    article_title: str = Field(..., description="Title of the news article")
    source: str = Field(..., description="News source (BBC, Guardian, Sky Sports)")
    published_date: datetime = Field(..., description="Article publication date")
    
    # Core quantitative metrics extracted by LLM
    key_absences_impact: float = Field(
        ..., 
        ge=0.0, 
        le=10.0, 
        description="Impact of key player absences (0=no impact, 10=critical)"
    )
    
    fatigue_rotation_risk: float = Field(
        ..., 
        ge=0.0, 
        le=10.0, 
        description="Risk level from fixture congestion/rotation (0=fresh, 10=exhausted)"
    )
    
    morale_sentiment_score: float = Field(
        ..., 
        ge=-5.0, 
        le=5.0, 
        description="Team morale sentiment (-5=very negative, +5=very positive)"
    )
    
    tactical_summary: str = Field(
        ..., 
        max_length=500,
        description="Brief summary of tactical insights from the article"
    )
    
    teams_mentioned: List[str] = Field(
        default_factory=list,
        description="List of EPL teams mentioned in the article"
    )
    
    confidence_score: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="LLM confidence in its own extraction (0-1)"
    )

    @field_validator('key_absences_impact', 'fatigue_rotation_risk', 'morale_sentiment_score')
    @classmethod
    def validate_scores(cls, v: float) -> float:
        """Ensure scores are within valid ranges and properly rounded"""
        return round(v, 2)

    class Config:
        json_schema_extra = {
            "example": {
                "article_title": "Arsenal suffer key injury blow ahead of Manchester City clash",
                "source": "BBC Sport",
                "published_date": "2024-03-15T10:30:00",
                "key_absences_impact": 7.5,
                "fatigue_rotation_risk": 4.0,
                "morale_sentiment_score": -2.5,
                "tactical_summary": "Arsenal's main striker ruled out for 3 weeks. Manager hints at tactical shift to 4-4-2.",
                "teams_mentioned": ["Arsenal", "Manchester City"],
                "confidence_score": 0.92
            }
        }


class LLMBatchResponse(BaseModel):
    """Container for multiple qualitative signals from batch processing"""
    signals: List[QualitativeSignal]
    processed_at: datetime = Field(default_factory=datetime.now)
    total_articles: int
    successful_extractions: int
    failed_extractions: int


class MatchContext(BaseModel):
    """Context window for LLM analysis of a specific match"""
    home_team: str
    away_team: str
    match_date: datetime
    recent_form_home: str  # e.g., "W-D-W-L-W"
    recent_form_away: str
    articles_for_context: List[str]  # Raw article texts

    class Config:
        json_schema_extra = {
            "example": {
                "home_team": "Liverpool",
                "away_team": "Chelsea",
                "match_date": "2024-03-20T17:30:00",
                "recent_form_home": "W-W-D-W-W",
                "recent_form_away": "L-D-W-L-D",
                "articles_for_context": ["Article 1 text...", "Article 2 text..."]
            }
        }
