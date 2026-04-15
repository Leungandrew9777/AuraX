# odds_fetcher.py
"""
Fetch upcoming EPL fixture odds from API-Football (v3).

Requires a free API key from https://dashboard.api-football.com/
Set it as env var  APIFOOTBALL_KEY  or pass it directly.

Produces a dict of  (HomeTeam, AwayTeam) -> {poly_prob_H, poly_prob_D, poly_prob_A}
(column names kept as poly_prob_* for downstream compatibility with app.py divergence tab).
"""

import os
import requests
import pandas as pd
from typing import Dict, Tuple

BASE_URL = "https://v3.football.api-sports.io"
EPL_LEAGUE_ID = 39
CURRENT_SEASON = 2025  # API-Football uses the *start year* of the season


TEAM_NAME_MAP = {
    "Manchester United":        "Man United",
    "Manchester City":          "Man City",
    "Tottenham Hotspur":        "Tottenham",
    "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion":   "Brighton",
    "Wolverhampton Wanderers":  "Wolves",
    "West Ham United":          "West Ham",
    "Nottingham Forest":        "Nott'ham Forest",
    "Newcastle United":         "Newcastle",
    "Leicester City":           "Leicester",
    "Crystal Palace":           "Crystal Palace",
    "AFC Bournemouth":          "Bournemouth",
}


class APIFootballOdds:
    """Thin wrapper around the API-Football v3 /fixtures + /odds endpoints."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("APIFOOTBALL_KEY", "6616c4b8cfcdaa51bc3129b1896faacb")
        if not self.api_key:
            raise ValueError(
                "No API key. Set APIFOOTBALL_KEY env var or pass api_key= "
                "(free tier at https://dashboard.api-football.com/)"
            )
        self.headers = {"x-apisports-key": self.api_key}

    def _get(self, endpoint: str, params: dict) -> list[dict]:
        url = f"{BASE_URL}/{endpoint}"
        resp = requests.get(url, headers=self.headers, params=params, timeout=20)
        resp.raise_for_status()
        body = resp.json()
        errors = body.get("errors")
        if errors:
            raise RuntimeError(f"API-Football error: {errors}")
        return body.get("response", [])

    @staticmethod
    def _normalize(name: str) -> str:
        return TEAM_NAME_MAP.get(name, name)

    # ------------------------------------------------------------------
    def get_upcoming_fixtures(self, next_n: int = 20) -> list[dict]:
        """Return the next *next_n* scheduled EPL fixtures."""
        rows = self._get("fixtures", {
            "league": EPL_LEAGUE_ID,
            "season": CURRENT_SEASON,
            "next": next_n,
        })
        fixtures = []
        for r in rows:
            fix = r.get("fixture", {})
            teams = r.get("teams", {})
            fixtures.append({
                "fixture_id": fix.get("id"),
                "date":       fix.get("date"),
                "home":       self._normalize(teams.get("home", {}).get("name", "")),
                "away":       self._normalize(teams.get("away", {}).get("name", "")),
            })
        return fixtures

    # ------------------------------------------------------------------
    def get_match_winner_odds(self, fixture_id: int) -> dict[str, float] | None:
        """
        Fetch pre-match 1X2 odds for a single fixture.

        Returns  {"poly_prob_H": .., "poly_prob_D": .., "poly_prob_A": ..}
        (implied probabilities normalised to sum=1) or None if unavailable.

        Uses bet id 1 = Match Winner.
        """
        rows = self._get("odds", {
            "fixture": fixture_id,
            "bet":     1,           # Match Winner
        })
        if not rows:
            return None

        bookmakers = rows[0].get("bookmakers", [])
        if not bookmakers:
            return None

        # Average implied probs across all available bookmakers
        h_probs, d_probs, a_probs = [], [], []
        for bk in bookmakers:
            for bet in bk.get("bets", []):
                if bet.get("id") != 1 and bet.get("name") != "Match Winner":
                    continue
                odds_map: dict[str, float] = {}
                for v in bet.get("values", []):
                    try:
                        odds_map[v["value"]] = float(v["odd"])
                    except (KeyError, TypeError, ValueError):
                        pass
                if "Home" in odds_map and "Draw" in odds_map and "Away" in odds_map:
                    h_probs.append(1.0 / odds_map["Home"])
                    d_probs.append(1.0 / odds_map["Draw"])
                    a_probs.append(1.0 / odds_map["Away"])

        if not h_probs:
            return None

        avg_h = sum(h_probs) / len(h_probs)
        avg_d = sum(d_probs) / len(d_probs)
        avg_a = sum(a_probs) / len(a_probs)
        total = avg_h + avg_d + avg_a
        return {
            "poly_prob_H": round(avg_h / total, 4),
            "poly_prob_D": round(avg_d / total, 4),
            "poly_prob_A": round(avg_a / total, 4),
        }

    # ------------------------------------------------------------------
    def get_upcoming_epl_odds(
        self, next_n: int = 20
    ) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        Main entry point.  Returns:
            {(HomeTeam, AwayTeam): {"poly_prob_H": .., "poly_prob_D": .., "poly_prob_A": ..}}
        for the next *next_n* EPL fixtures that have odds available.
        """
        fixtures = self.get_upcoming_fixtures(next_n)
        result: Dict[Tuple[str, str], Dict[str, float]] = {}
        for f in fixtures:
            probs = self.get_match_winner_odds(f["fixture_id"])
            if probs:
                result[(f["home"], f["away"])] = probs
        print(f"OK: fetched odds for {len(result)}/{len(fixtures)} upcoming EPL fixtures")
        return result


# ====================== TEST / USAGE ======================
if __name__ == "__main__":
    fetcher = APIFootballOdds()
    odds = fetcher.get_upcoming_epl_odds(next_n=10)
    for match, probs in odds.items():
        print(f"  {match[0]:>22} vs {match[1]:<22}  H={probs['poly_prob_H']:.2f}  D={probs['poly_prob_D']:.2f}  A={probs['poly_prob_A']:.2f}")
