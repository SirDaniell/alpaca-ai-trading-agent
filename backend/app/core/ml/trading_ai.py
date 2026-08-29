"""
Trending Features & Advanced Recommendations
AI-powered trading insights, market sentiment, and predictive analytics
"""

import asyncio
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import json

from app.database.connection import DbConfig, connect_with_retry
from app.services.mt5_service import MT5Service
from app.core.analysis.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class RecommendationType(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class MarketSentiment(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class TradingRecommendation:
    """AI-powered trading recommendation"""
    symbol: str
    recommendation: RecommendationType
    confidence: float  # 0-1
    reasoning: List[str]
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    risk_level: str
    expected_return: float
    stop_loss: float
    take_profit: float
    timeframe: str
    timestamp: datetime


@dataclass
class MarketInsight:
    """Market analysis insight"""
    insight_type: str
    title: str
    description: str
    symbols_affected: List[str]
    impact_level: str  # high, medium, low
    confidence: float
    timestamp: datetime


class AIRecommendationEngine:
    """
    AI-powered trading recommendation engine

    Features:
    - Machine learning-based predictions
    - Technical analysis scoring
    - Market sentiment analysis
    - Risk-adjusted recommendations
    - Portfolio optimization suggestions
    """

    def __init__(self, db_config: DbConfig, mt5_service: MT5Service):
        self.db_config = db_config
        self.mt5_service = mt5_service
        self.ti = TechnicalIndicators()

    async def generate_recommendations(
        self,
        symbols: List[str],
        user_id: str,
        risk_tolerance: str = "medium"
    ) -> List[TradingRecommendation]:
        """
        Generate AI-powered trading recommendations

        Combines technical, fundamental, and sentiment analysis
        """
        recommendations = []

        for symbol in symbols:
            try:
                # Get comprehensive symbol analysis
                analysis = await self._analyze_symbol(symbol, risk_tolerance)

                if analysis:
                    recommendations.append(analysis)

            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {e}")

        # Sort by confidence and expected return
        recommendations.sort(key=lambda x: (x.confidence, x.expected_return), reverse=True)

        return recommendations[:20]  # Top 20 recommendations

    async def _analyze_symbol(self, symbol: str, risk_tolerance: str) -> Optional[TradingRecommendation]:
        """Comprehensive analysis of a single symbol"""

        # Get market data
        ohlc_data = await self.mt5_service.fetch_ohlc_data_v2(
            symbol=symbol,
            timeframe="H1",
            count=200  # Last 200 hours
        )

        if not ohlc_data or 'error' in ohlc_data:
            return None

        df = pd.DataFrame(ohlc_data)
        if df.empty:
            return None

        # Calculate technical indicators
        technical_score = await self._calculate_technical_score(df)

        # Get fundamental data (placeholder)
        fundamental_score = await self._calculate_fundamental_score(symbol)

        # Analyze market sentiment
        sentiment_score = await self._analyze_sentiment(symbol, df)

        # Combine scores with AI weighting
        combined_score = self._combine_scores(
            technical_score, fundamental_score, sentiment_score, risk_tolerance
        )

        # Generate recommendation
        recommendation, confidence = self._generate_recommendation(combined_score)

        if recommendation == RecommendationType.HOLD:
            return None  # Skip hold recommendations

        # Calculate risk parameters
        risk_level, stop_loss, take_profit = self._calculate_risk_parameters(df, recommendation)

        # Expected return calculation
        expected_return = self._calculate_expected_return(df, recommendation, confidence)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            technical_score, fundamental_score, sentiment_score, recommendation
        )

        return TradingRecommendation(
            symbol=symbol,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning,
            technical_score=technical_score,
            fundamental_score=fundamental_score,
            sentiment_score=sentiment_score,
            risk_level=risk_level,
            expected_return=expected_return,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timeframe="H1",
            timestamp=datetime.now()
        )

    async def _calculate_technical_score(self, df: pd.DataFrame) -> float:
        """Calculate technical analysis score (0-100)"""
        try:
            # Calculate key indicators
            close_prices = df['close'].values

            # Trend indicators
            sma_20 = pd.Series(close_prices).rolling(20).mean().iloc[-1]
            sma_50 = pd.Series(close_prices).rolling(50).mean().iloc[-1]
            current_price = close_prices[-1]

            trend_score = 50  # Neutral
            if current_price > sma_20 > sma_50:
                trend_score = 80  # Strong uptrend
            elif current_price < sma_20 < sma_50:
                trend_score = 20  # Strong downtrend

            # Momentum indicators (simplified RSI calculation)
            def calculate_rsi(prices, period=14):
                gains = []
                losses = []
                for i in range(1, len(prices)):
                    change = prices[i] - prices[i-1]
                    if change > 0:
                        gains.append(change)
                        losses.append(0)
                    else:
                        gains.append(0)
                        losses.append(abs(change))

                avg_gain = np.mean(gains[-period:]) if gains else 0
                avg_loss = np.mean(losses[-period:]) if losses else 0

                if avg_loss == 0:
                    return 100
                rs = avg_gain / avg_loss
                return 100 - (100 / (1 + rs))

            rsi = calculate_rsi(close_prices)

            momentum_score = 50
            if rsi > 70:
                momentum_score = 80  # Overbought
            elif rsi < 30:
                momentum_score = 20  # Oversold
            else:
                momentum_score = 50

            # Volume analysis (if available)
            volume_score = 50
            if 'tick_volume' in df.columns:
                volumes = df['tick_volume'].values
                avg_volume = np.mean(volumes[-20:])
                current_volume = volumes[-1]
                if current_volume > avg_volume * 1.5:
                    volume_score = 75  # High volume
                elif current_volume < avg_volume * 0.5:
                    volume_score = 25  # Low volume

            # Combine scores
            technical_score = (trend_score * 0.4 + momentum_score * 0.4 + volume_score * 0.2)

            return min(100, max(0, technical_score))

        except Exception as e:
            logger.error(f"Technical score calculation failed: {e}")
            return 50  # Neutral

    async def _calculate_fundamental_score(self, symbol: str) -> float:
        """Calculate fundamental analysis score (placeholder)"""
        # In production, this would integrate with financial data APIs
        # For now, return neutral score
        return 50.0

    async def _analyze_sentiment(self, symbol: str, df: pd.DataFrame) -> float:
        """Analyze market sentiment for the symbol"""
        try:
            # Price action sentiment
            close_prices = df['close'].values
            recent_prices = close_prices[-20:]  # Last 20 periods

            # Calculate price momentum
            momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

            # Volatility sentiment (high volatility = uncertain sentiment)
            returns = np.diff(np.log(recent_prices))
            volatility = np.std(returns)

            # Combine factors
            if momentum > 0.02:  # Strong upward momentum
                base_sentiment = 75
            elif momentum < -0.02:  # Strong downward momentum
                base_sentiment = 25
            else:
                base_sentiment = 50

            # Adjust for volatility (high volatility reduces confidence)
            volatility_adjustment = min(20, volatility * 1000)  # Cap adjustment
            sentiment_score = base_sentiment - volatility_adjustment

            return max(0, min(100, sentiment_score))

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return 50  # Neutral

    def _combine_scores(
        self,
        technical: float,
        fundamental: float,
        sentiment: float,
        risk_tolerance: str
    ) -> float:
        """Combine multiple analysis scores with risk weighting"""

        # Weight factors based on risk tolerance
        if risk_tolerance == "low":
            weights = {'technical': 0.5, 'fundamental': 0.3, 'sentiment': 0.2}
        elif risk_tolerance == "high":
            weights = {'technical': 0.7, 'fundamental': 0.1, 'sentiment': 0.2}
        else:  # medium
            weights = {'technical': 0.5, 'fundamental': 0.2, 'sentiment': 0.3}

        combined = (
            technical * weights['technical'] +
            fundamental * weights['fundamental'] +
            sentiment * weights['sentiment']
        )

        return combined

    def _generate_recommendation(self, combined_score: float) -> Tuple[RecommendationType, float]:
        """Generate trading recommendation from combined score"""

        if combined_score >= 75:
            return RecommendationType.STRONG_BUY, min(0.9, combined_score / 100)
        elif combined_score >= 60:
            return RecommendationType.BUY, min(0.8, combined_score / 100)
        elif combined_score >= 40:
            return RecommendationType.HOLD, 0.5
        elif combined_score >= 25:
            return RecommendationType.SELL, min(0.7, (100 - combined_score) / 100)
        else:
            return RecommendationType.STRONG_SELL, min(0.8, (100 - combined_score) / 100)

    def _calculate_risk_parameters(
        self,
        df: pd.DataFrame,
        recommendation: RecommendationType
    ) -> Tuple[str, float, float]:
        """Calculate risk management parameters"""

        close_prices = df['close'].values
        current_price = close_prices[-1]

        # Calculate ATR for stop loss (simplified)
        def calculate_atr(prices, period=14):
            high_low = np.abs(np.diff(prices))
            atr = np.mean(high_low[-period:])
            return atr

        atr = calculate_atr(close_prices)

        # Risk level assessment
        volatility = np.std(np.diff(np.log(close_prices[-20:]))) * 100
        if volatility > 3:
            risk_level = "high"
            stop_distance = atr * 2  # Wider stops for volatile assets
        elif volatility > 1.5:
            risk_level = "medium"
            stop_distance = atr * 1.5
        else:
            risk_level = "low"
            stop_distance = atr

        # Set stop loss and take profit based on recommendation
        if recommendation in [RecommendationType.STRONG_BUY, RecommendationType.BUY]:
            stop_loss = current_price - stop_distance
            take_profit = current_price + (stop_distance * 2)  # 2:1 reward ratio
        elif recommendation in [RecommendationType.STRONG_SELL, RecommendationType.SELL]:
            stop_loss = current_price + stop_distance
            take_profit = current_price - (stop_distance * 2)
        else:
            stop_loss = current_price - stop_distance
            take_profit = current_price + stop_distance

        return risk_level, stop_loss, take_profit

    def _calculate_expected_return(
        self,
        df: pd.DataFrame,
        recommendation: RecommendationType,
        confidence: float
    ) -> float:
        """Calculate expected return based on historical performance"""

        # Simplified: base expected return on recent performance
        close_prices = df['close'].values
        recent_return = (close_prices[-1] - close_prices[-20]) / close_prices[-20]

        # Adjust based on recommendation and confidence
        if recommendation in [RecommendationType.STRONG_BUY, RecommendationType.BUY]:
            base_return = max(0.02, recent_return)  # At least 2%
        elif recommendation in [RecommendationType.STRONG_SELL, RecommendationType.SELL]:
            base_return = min(-0.02, recent_return)  # At least -2%
        else:
            base_return = recent_return

        # Scale by confidence
        expected_return = base_return * confidence

        return expected_return * 100  # Convert to percentage

    def _generate_reasoning(
        self,
        technical: float,
        fundamental: float,
        sentiment: float,
        recommendation: RecommendationType
    ) -> List[str]:
        """Generate human-readable reasoning for the recommendation"""

        reasoning = []

        # Technical analysis reasoning
        if technical > 70:
            reasoning.append("Strong technical indicators showing bullish momentum")
        elif technical < 30:
            reasoning.append("Technical indicators suggest bearish pressure")
        else:
            reasoning.append("Technical indicators are neutral")

        # Sentiment reasoning
        if sentiment > 70:
            reasoning.append("Market sentiment is strongly bullish")
        elif sentiment < 30:
            reasoning.append("Market sentiment is bearish")
        else:
            reasoning.append("Market sentiment is mixed")

        # Fundamental reasoning (placeholder)
        if fundamental > 60:
            reasoning.append("Fundamental factors are supportive")
        elif fundamental < 40:
            reasoning.append("Fundamental factors are concerning")

        # Add recommendation-specific reasoning
        if recommendation == RecommendationType.STRONG_BUY:
            reasoning.append("All factors align for a strong buying opportunity")
        elif recommendation == RecommendationType.BUY:
            reasoning.append("Positive signals suggest a buying opportunity")
        elif recommendation == RecommendationType.SELL:
            reasoning.append("Negative signals suggest reducing exposure")
        elif recommendation == RecommendationType.STRONG_SELL:
            reasoning.append("Strong negative signals recommend exiting positions")

        return reasoning


class MarketSentimentAnalyzer:
    """
    Advanced market sentiment analysis using multiple data sources
    """

    def __init__(self, db_config: DbConfig):
        self.db_config = db_config

    async def analyze_market_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze overall market sentiment"""

        # Price-based sentiment
        price_sentiment = await self._analyze_price_sentiment(symbols)

        # Volume-based sentiment
        volume_sentiment = await self._analyze_volume_sentiment(symbols)

        # Order flow sentiment (if available)
        order_flow_sentiment = await self._analyze_order_flow_sentiment(symbols)

        # Combine sentiments
        overall_sentiment = self._combine_sentiments([
            price_sentiment,
            volume_sentiment,
            order_flow_sentiment
        ])

        return {
            'overall_sentiment': overall_sentiment,
            'price_sentiment': price_sentiment,
            'volume_sentiment': volume_sentiment,
            'order_flow_sentiment': order_flow_sentiment,
            'timestamp': datetime.now()
        }

    async def _analyze_price_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze sentiment based on price movements"""
        # Implementation would analyze price patterns across symbols
        return {'score': 50, 'description': 'Neutral price action'}

    async def _analyze_volume_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze sentiment based on volume patterns"""
        # Implementation would analyze volume trends
        return {'score': 55, 'description': 'Moderate volume activity'}

    async def _analyze_order_flow_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze sentiment based on order flow data"""
        # Implementation would analyze bid/ask imbalances
        return {'score': 45, 'description': 'Mixed order flow'}

    def _combine_sentiments(self, sentiments: List[Dict]) -> Dict[str, Any]:
        """Combine multiple sentiment indicators"""
        avg_score = np.mean([s['score'] for s in sentiments])

        if avg_score > 60:
            overall = MarketSentiment.BULLISH
        elif avg_score < 40:
            overall = MarketSentiment.BEARISH
        else:
            overall = MarketSentiment.NEUTRAL

        return {
            'sentiment': overall,
            'score': avg_score,
            'confidence': min(0.9, abs(avg_score - 50) / 50)  # Higher confidence further from neutral
        }


class PredictiveAnalytics:
    """
    Machine learning-based predictive analytics for trading
    """

    def __init__(self, db_config: DbConfig):
        self.db_config = db_config

    async def predict_price_movement(
        self,
        symbol: str,
        timeframe: str = "H1",
        prediction_horizon: int = 24
    ) -> Dict[str, Any]:
        """
        Predict price movement using machine learning models

        Returns prediction with confidence intervals
        """
        try:
            # Load historical data
            historical_data = await self._load_historical_data(symbol, timeframe, 1000)

            if not historical_data:
                return {'error': 'Insufficient historical data'}

            # Feature engineering
            features = self._engineer_features(historical_data)

            # Load trained model (placeholder - would load from MLflow)
            prediction = self._make_prediction(features)

            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(prediction)

            return {
                'symbol': symbol,
                'prediction': prediction,
                'confidence_intervals': confidence_intervals,
                'prediction_horizon': prediction_horizon,
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Price prediction failed for {symbol}: {e}")
            return {'error': str(e)}

    async def _load_historical_data(self, symbol: str, timeframe: str, count: int) -> Optional[pd.DataFrame]:
        """Load historical data for prediction"""
        # Implementation would fetch from database or MT5
        return None

    def _engineer_features(self, data: pd.DataFrame) -> np.ndarray:
        """Engineer features for ML model"""
        # Implementation would create technical indicators and other features
        return np.array([])

    def _make_prediction(self, features: np.ndarray) -> Dict[str, Any]:
        """Make price prediction using trained model"""
        # Placeholder - would use actual ML model
        return {
            'direction': 'up',
            'magnitude': 0.015,  # 1.5% expected move
            'probability': 0.65,
            'confidence': 0.7
        }

    def _calculate_confidence_intervals(self, prediction: Dict) -> Dict[str, float]:
        """Calculate prediction confidence intervals"""
        return {
            'lower_95': prediction['magnitude'] * 0.5,
            'upper_95': prediction['magnitude'] * 1.5,
            'lower_80': prediction['magnitude'] * 0.7,
            'upper_80': prediction['magnitude'] * 1.3
        }


# API Endpoints for Advanced Features
async def get_ai_recommendations(
    symbols: List[str],
    risk_tolerance: str = "medium",
    current_user: Dict = None,
    db_config: DbConfig = None,
    mt5_service: MT5Service = None
) -> Dict[str, Any]:
    """API endpoint for AI-powered recommendations"""

    engine = AIRecommendationEngine(db_config, mt5_service)
    recommendations = await engine.generate_recommendations(
        symbols, current_user['user_id'], risk_tolerance
    )

    return {
        'recommendations': [rec.__dict__ for rec in recommendations],
        'count': len(recommendations),
        'generated_at': datetime.now()
    }


async def get_market_sentiment(
    symbols: List[str],
    db_config: DbConfig = None
) -> Dict[str, Any]:
    """API endpoint for market sentiment analysis"""

    analyzer = MarketSentimentAnalyzer(db_config)
    sentiment = await analyzer.analyze_market_sentiment(symbols)

    return sentiment


async def get_price_prediction(
    symbol: str,
    timeframe: str = "H1",
    prediction_horizon: int = 24,
    db_config: DbConfig = None
) -> Dict[str, Any]:
    """API endpoint for price predictions"""

    analytics = PredictiveAnalytics(db_config)
    prediction = await analytics.predict_price_movement(
        symbol, timeframe, prediction_horizon
    )

    return prediction