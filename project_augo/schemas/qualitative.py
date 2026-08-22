"""
Project Augo - Pydantic Schemas for LLM Output Validation
Strict JSON schema for qualitative signal extraction.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime


class QualitativeSignal(BaseModel):
    """
    Schema for LLM-parsed qualitative metrics from news/articles.
    
    All numerical fields are validated to ensure they fall within
    the specified ranges for consistent downstream processing.
    """
    article_id: str = Field(..., description="Unique identifier for the source article")
    source: str = Field(..., description="News source (e.g., BBC Sport, Guardian)")
    published_at: datetime = Field(..., description="Article publication timestamp")
    teams_mentioned: List[str] = Field(default_factory=list, description="EPL teams mentioned in article")
    
    # Core quantitative metrics extracted by LLM
    key_absences_impact: float = Field(
        ..., 
        ge=0.0, 
        le=10.0,
        description="Impact score of player absences (0=no impact, 10=critical)"
    )
    
    fatigue_rotation_risk: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Risk score for fatigue/rotation (0=fresh, 10=exhausted)"
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
    
    confidence_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="LLM confidence in its own extraction (0-1)"
    )
    
    raw_json_payload: Optional[dict] = Field(
        default=None,
        description="Original raw JSON response from LLM for audit trail"
    )
    
    @field_validator('key_absences_impact', 'fatigue_rotation_risk')
    @classmethod
    def validate_impact_scores(cls, v: float) -> float:
        if not (0.0 <= v <= 10.0):
            raise ValueError("Impact scores must be between 0.0 and 10.0")
        return round(v, 2)
    
    @field_validator('morale_sentiment_score')
    @classmethod
    def validate_sentiment(cls, v: float) -> float:
        if not (-5.0 <= v <= 5.0):
            raise ValueError("Sentiment score must be between -5.0 and +5.0")
        return round(v, 2)
    
    class Config:
        json_schema_extra = {
            "example": {
                "article_id": "bbc_20241201_001",
                "source": "BBC Sport",
                "published_at": "2024-12-01T10:30:00Z",
                "teams_mentioned": ["Arsenal", "Chelsea"],
                "key_absences_impact": 7.5,
                "fatigue_rotation_risk": 4.0,
                "morale_sentiment_score": 2.5,
                "tactical_summary": "Arsenal missing key midfielder; Chelsea showing strong form after recent tactical adjustment to 3-4-3.",
                "confidence_score": 0.85
            }
        }


class LLMPromptTemplate:
    """
    Prompt template for consistent LLM extraction.
    This ensures the Ollama model returns properly structured JSON.
    """
    
    SYSTEM_PROMPT = """You are a specialized sports analytics AI. Your task is to extract 
quantitative metrics from football news articles and return ONLY valid JSON matching the 
specified schema. Do not include any explanatory text outside the JSON object.

Extract these fields:
- key_absences_impact: 0.0-10.0 (10 = multiple key players out)
- fatigue_rotation_risk: 0.0-10.0 (10 = exhausted squad, congested fixtures)
- morale_sentiment_score: -5.0 to +5.0 (negative to positive team morale)
- tactical_summary: Brief tactical insight (max 500 chars)
- confidence_score: 0.0-1.0 (your confidence in this analysis)

Be objective and data-driven. Consider:
- Injury reports and suspension news
- Recent fixture congestion
- Manager press conference tone
- Historical performance patterns mentioned"""

    USER_PROMPT_TEMPLATE = """Analyze this football news article and extract quantitative metrics:

ARTICLE TITLE: {title}
ARTICLE CONTENT: {content}
PUBLISHED: {published_at}
SOURCE: {source}

Return ONLY valid JSON matching the QualitativeSignal schema."""

    @classmethod
    def build_prompt(cls, title: str, content: str, published_at: str, source: str) -> str:
        return cls.USER_PROMPT_TEMPLATE.format(
            title=title,
            content=content[:2000],  # Truncate for context window
            published_at=published_at,
            source=source
        )
