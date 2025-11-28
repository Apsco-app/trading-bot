# ultimate_forex_trading_platform.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import MetaTrader5 as mt5
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import threading
from api_server import start_flask, incoming_signals
from dataclasses import dataclass
from typing import Dict, List
from enum import Enum

# --- TECHNICAL ANALYSIS LIBRARY IMPORTS ---
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION & SETUP ====================

class TradingMode(Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"
    OPTIMIZATION = "optimization"

@dataclass
class MarketData:
    symbol: str
    bid: float
    ask: float
    spread: float
    volume: float
    timestamp: datetime

# ==================== PERFORMANCE TRACKING ====================

class PerformanceTracker:
    def __init__(self):
        self.trades = pd.DataFrame()

    def calculate_metrics(self, trades: List) -> Dict:
        """Calculate comprehensive performance metrics"""
        if len(trades) < 2:
            return {}

        df = pd.DataFrame(trades)
        if 'profit' not in df.columns:
            return {}

        # Basic metrics
        total_trades = len(df)
        winning_trades = len(df[df['profit'] > 0])
        total_profit = df['profit'].sum()
        
        # Drawdown
        equity_curve = df['profit'].cumsum().values
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

        return {
            'total_trades': total_trades,
            'win_rate': (winning_trades / total_trades) * 100 if total_trades > 0 else 0,
            'total_profit': total_profit,
            'max_drawdown': max_drawdown * 100
        }

class AdvancedForexTradingPlatform:
    def __init__(self):
        self.connected = False
        self.ml_models = {}
        self.scalers = {}
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        defaults = {
            'account_info': {'balance': 10000, 'equity': 10000, 'profit': 0, 'open_trades': 0},
            'signals': [],
            'news_signals': [],  # NEW: Store news signals
            'strategy_params': self.get_default_strategy_params()
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def get_default_strategy_params(self):
        return {'ema_fast': 12, 'ema_slow': 26, 'rsi_period': 14}

    # ==================== MT5 CONNECTION ====================

    def connect_mt5(self) -> bool:
        """Connect to MetaTrader 5"""
        default_path = None 
        win_path = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"

        if not mt5.initialize(path=default_path):
            if not mt5.initialize(path=win_path):
                st.error("❌ MT5 initialization failed. Ensure MT5 is installed and running.")
                return False
        
        account_info = mt5.account_info()
        if account_info is None:
            st.warning("⚠️ MT5 initialized, but no account is logged in. Using Demo Mode (Data only).")
        else:
            st.toast(f"✅ Connected to MT5 Account: {account_info.login}")

        self.connected = True
        return True

    def fetch_market_data(self, symbol: str, bars: int = 100):
        """Fetch market data from MT5 or generate sample if offline"""
        try:
            if self.connected:
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, bars)
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    return df
            return self.generate_sample_data(symbol, bars)
        except Exception:
            return self.generate_sample_data(symbol, bars)

    def generate_sample_data(self, symbol: str, bars: int = 100):
        dates = pd.date_range(end=datetime.now(), periods=bars, freq='h')
        base_price = 1.10
        prices = base_price + np.cumsum(np.random.randn(bars) * 0.001)
        
        df = pd.DataFrame({
            'open': prices * 0.999, 'high': prices * 1.001,
            'low': prices * 0.998, 'close': prices,
            'volume': np.random.randint(100, 1000, bars)
        }, index=dates)
        return df

    # ==================== TECHNICAL ANALYSIS ====================

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators using 'ta' library"""
        try:
            df = df.fillna(method='ffill').fillna(method='bfill')

            df['ema_12'] = EMAIndicator(close=df['close'], window=12).ema_indicator()
            df['ema_26'] = EMAIndicator(close=df['close'], window=26).ema_indicator()
            df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
            
            macd = MACD(close=df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_hist'] = macd.macd_diff()

            bb = BollingerBands(close=df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()

            atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['atr'] = atr.average_true_range()
            
            return df.fillna(0)
            
        except Exception as e:
            st.warning(f"Indicator warning: {str(e)}")
            return df

    # ==================== MACHINE LEARNING ====================

    def prepare_ml_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['returns'] = df['close'].pct_change()
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        return df.dropna()

    def train_ml_model(self, symbol: str, df: pd.DataFrame):
        """Simple Random Forest Training"""
        try:
            features = ['ema_12', 'ema_26', 'rsi', 'macd']
            available_features = [f for f in features if f in df.columns]
            
            if len(df) < 50 or not available_features:
                return False

            X = df[available_features].values
            y = df['target'].values
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = RandomForestClassifier(n_estimators=50, max_depth=5)
            model.fit(X_scaled, y)
            
            self.ml_models[symbol] = model
            self.scalers[symbol] = scaler
            return True
        except Exception as e:
            st.error(f"ML Error: {e}")
            return False

    def ml_predict(self, symbol: str, df: pd.DataFrame) -> Dict:
        if symbol not in self.ml_models:
            return {"signal": "none", "confidence": 0}
            
        try:
            latest = df.iloc[-1:]
            features = ['ema_12', 'ema_26', 'rsi', 'macd']
            available = [f for f in features if f in df.columns]
            
            X = latest[available].values
            X_scaled = self.scalers[symbol].transform(X)
            
            prob = self.ml_models[symbol].predict_proba(X_scaled)[0]
            prediction = self.ml_models[symbol].predict(X_scaled)[0]
            
            signal = "buy" if prediction == 1 else "sell"
            return {"signal": signal, "confidence": max(prob)}
        except:
            return {"signal": "none", "confidence": 0}

    # ==================== TRADING LOGIC ====================

    def technical_analysis_signal(self, df: pd.DataFrame) -> Dict:
        """Technical analysis signal generation"""
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            signal = "none"
            confidence = 0.0

            bullish_cross = (prev['ema_12'] <= prev['ema_26']) and (latest['ema_12'] > latest['ema_26'])
            bearish_cross = (prev['ema_12'] >= prev['ema_26']) and (latest['ema_12'] < latest['ema_26'])

            if bullish_cross:
                signal = "buy"
                confidence = 0.6
                if latest['rsi'] < 70: confidence += 0.2
            elif bearish_cross:
                signal = "sell"
                confidence = 0.6
                if latest['rsi'] > 30: confidence += 0.2

            return {"signal": signal, "confidence": confidence}
        except Exception:
            return {"signal": "none", "confidence": 0}

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Generate trading signal with news integration"""
        ml_res = self.ml_predict(symbol, df)
        ta_res = self.technical_analysis_signal(df)
        
        # Base signal from ML and TA
        base_signal = "none"
        base_confidence = 0
        
        if ml_res['signal'] == ta_res['signal'] and ml_res['signal'] != "none":
            base_signal = ml_res['signal']
            base_confidence = (ml_res['confidence'] + ta_res['confidence']) / 2
        elif ta_res['confidence'] > 0.7:
            base_signal = ta_res['signal']
            base_confidence = ta_res['confidence']
        
        # Integrate with news-based recommendation
        recommendation = self.get_trading_recommendation(symbol, base_signal)
        
        return {
            'signal': recommendation['final_signal'],
            'confidence': min(base_confidence + recommendation['confidence_boost'], 1.0),
            'position_multiplier': recommendation['position_size_multiplier'],
            'reasoning': recommendation['reasoning']
        }

    # ==================== NEWS SIGNAL PROCESSING (NEW) ====================

    def process_news_signal(self, sig: Dict):
        """Process incoming news signal and integrate with trading logic"""
        try:
            # Store signal in session state
            st.session_state.news_signals.append({
                'timestamp': datetime.now(),
                'symbols': sig.get('symbols', []),
                'sentiment': sig.get('sentiment', 'neutral'),
                'impact': sig.get('impact', 'unknown'),
                'confidence': sig.get('confidence', 0),
                'headline': sig.get('headline', 'No headline'),
                'action_taken': None  # Will store what action was taken
            })
            
            # Limit stored signals to last 10
            if len(st.session_state.news_signals) > 10:
                st.session_state.news_signals = st.session_state.news_signals[-10:]
            
            # Integrated Trading Logic
            sentiment = sig.get('sentiment', 'neutral').lower()
            impact = sig.get('impact', 'unknown').lower()
            confidence = sig.get('confidence', 0)
            symbols = sig.get('symbols', [])
            
            action_taken = "No action"
            
            # High confidence bullish signals
            if sentiment == 'bullish' and confidence >= 0.75:
                if impact == 'high':
                    action_taken = "🟢 STRONG BUY signal flagged - Consider increasing position size by 50%"
                    self._flag_trading_opportunity(symbols, 'buy', 'high', confidence)
                elif impact == 'medium':
                    action_taken = "🟢 BUY signal flagged - Standard position size"
                    self._flag_trading_opportunity(symbols, 'buy', 'medium', confidence)
            
            # High confidence bearish signals
            elif sentiment == 'bearish' and confidence >= 0.75:
                if impact == 'high':
                    action_taken = "🔴 STRONG SELL signal flagged - Consider reducing exposure by 50%"
                    self._flag_trading_opportunity(symbols, 'sell', 'high', confidence)
                elif impact == 'medium':
                    action_taken = "🔴 SELL signal flagged - Standard position size"
                    self._flag_trading_opportunity(symbols, 'sell', 'medium', confidence)
            
            # Moderate confidence signals - use as confirmation
            elif confidence >= 0.60 and confidence < 0.75:
                if sentiment == 'bullish':
                    action_taken = "🟡 Bullish bias - Use as confirmation for technical signals"
                    self._add_bias_filter(symbols, 'bullish', confidence)
                elif sentiment == 'bearish':
                    action_taken = "🟡 Bearish bias - Use as confirmation for technical signals"
                    self._add_bias_filter(symbols, 'bearish', confidence)
            
            # High impact news - pause trading temporarily
            elif impact == 'high' and sentiment == 'neutral':
                action_taken = "⚠️ High volatility expected - Reducing position sizes by 30%"
                self._adjust_risk_parameters(symbols, reduce_risk=True)
            
            # Store the action taken
            st.session_state.news_signals[-1]['action_taken'] = action_taken
            
            return action_taken
            
        except Exception as e:
            st.error(f"Error processing news signal: {e}")
            return "Error processing signal"

    def _flag_trading_opportunity(self, symbols: List[str], direction: str, strength: str, confidence: float):
        """Flag a trading opportunity based on news"""
        for symbol in symbols:
            if 'trading_opportunities' not in st.session_state:
                st.session_state.trading_opportunities = []
            
            opportunity = {
                'symbol': symbol,
                'direction': direction,
                'strength': strength,
                'confidence': confidence,
                'source': 'news',
                'timestamp': datetime.now(),
                'expired': False
            }
            st.session_state.trading_opportunities.append(opportunity)
            
            # Expire old opportunities (older than 4 hours)
            current_time = datetime.now()
            for opp in st.session_state.trading_opportunities:
                if (current_time - opp['timestamp']).total_seconds() > 14400:
                    opp['expired'] = True

    def _add_bias_filter(self, symbols: List[str], bias: str, confidence: float):
        """Add market bias filter to trading decisions"""
        if 'market_bias' not in st.session_state:
            st.session_state.market_bias = {}
        
        for symbol in symbols:
            st.session_state.market_bias[symbol] = {
                'bias': bias,
                'confidence': confidence,
                'timestamp': datetime.now()
            }

    def _adjust_risk_parameters(self, symbols: List[str], reduce_risk: bool = True):
        """Adjust risk parameters based on news impact"""
        if 'risk_adjustments' not in st.session_state:
            st.session_state.risk_adjustments = {}
        
        for symbol in symbols:
            if reduce_risk:
                st.session_state.risk_adjustments[symbol] = {
                    'position_size_multiplier': 0.7,  # Reduce to 70%
                    'stop_loss_multiplier': 1.5,      # Wider stops
                    'reason': 'high_impact_news',
                    'timestamp': datetime.now()
                }
            else:
                st.session_state.risk_adjustments[symbol] = {
                    'position_size_multiplier': 1.0,
                    'stop_loss_multiplier': 1.0,
                    'reason': 'normal',
                    'timestamp': datetime.now()
                }

    def get_trading_recommendation(self, symbol: str, technical_signal: str) -> Dict:
        """Combine technical signals with news-based insights"""
        recommendation = {
            'final_signal': technical_signal,
            'position_size_multiplier': 1.0,
            'confidence_boost': 0,
            'reasoning': []
        }
        
        # Check for active trading opportunities from news
        if 'trading_opportunities' in st.session_state:
            active_opps = [opp for opp in st.session_state.trading_opportunities 
                          if opp['symbol'] == symbol and not opp['expired']]
            
            for opp in active_opps:
                if opp['direction'] == technical_signal:
                    # News confirms technical signal
                    if opp['strength'] == 'high':
                        recommendation['position_size_multiplier'] *= 1.5
                        recommendation['confidence_boost'] += 0.2
                        recommendation['reasoning'].append(f"Strong {opp['direction']} news confirmation")
                    else:
                        recommendation['position_size_multiplier'] *= 1.2
                        recommendation['confidence_boost'] += 0.1
                        recommendation['reasoning'].append(f"News confirmation")
                elif opp['direction'] != technical_signal and opp['strength'] == 'high':
                    # Strong news conflicts with technical
                    recommendation['final_signal'] = 'hold'
                    recommendation['reasoning'].append(f"News conflicts with technical - HOLD recommended")
        
        # Check market bias
        if 'market_bias' in st.session_state and symbol in st.session_state.market_bias:
            bias_info = st.session_state.market_bias[symbol]
            if bias_info['bias'] == 'bullish' and technical_signal == 'buy':
                recommendation['confidence_boost'] += 0.05
                recommendation['reasoning'].append("Bullish market bias supports trade")
            elif bias_info['bias'] == 'bearish' and technical_signal == 'sell':
                recommendation['confidence_boost'] += 0.05
                recommendation['reasoning'].append("Bearish market bias supports trade")
        
        # Apply risk adjustments
        if 'risk_adjustments' in st.session_state and symbol in st.session_state.risk_adjustments:
            risk_adj = st.session_state.risk_adjustments[symbol]
            recommendation['position_size_multiplier'] *= risk_adj['position_size_multiplier']
            if risk_adj['reason'] == 'high_impact_news':
                recommendation['reasoning'].append("Position size reduced due to high-impact news")
        
        return recommendation

# ==================== STREAMLIT UI ====================

def main():
    st.set_page_config(page_title="Ultimate Forex Bot", layout="wide")
    st.title("🤖 Ultimate Forex Trading Platform")
    
    # Initialize Bot
    bot = AdvancedForexTradingPlatform()
    
    # Sidebar
    st.sidebar.header("Configuration")
    symbol = st.sidebar.text_input("Symbol", "EURUSD")
    if st.sidebar.button("Connect MT5"):
        bot.connect_mt5()

    # Main Dashboard - Account Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Balance", f"${st.session_state.account_info['balance']:.2f}")
    with col2:
        st.metric("Equity", f"${st.session_state.account_info['equity']:.2f}")
    with col3:
        st.metric("Status", "🟢 Connected" if bot.connected else "🔴 Disconnected")

    # ==================== NEWS SIGNALS SECTION (NEW) ====================
    st.markdown("---")
    st.subheader("📩 Incoming AI News-Based Signals")
    
    # Process signals received from n8n
    signals_list = []
    while not incoming_signals.empty():
        signals_list.append(incoming_signals.get())
    
    if signals_list:
        for sig in signals_list:
            # Process each signal and get action taken
            action_taken = bot.process_news_signal(sig)
            
            # Display the signal
            sentiment_emoji = "📈" if sig.get('sentiment') == 'bullish' else "📉" if sig.get('sentiment') == 'bearish' else "➡️"
            impact_color = "🔴" if sig.get('impact') == 'high' else "🟡" if sig.get('impact') == 'medium' else "🟢"
            
            st.success(f"""
            🔔 **News Signal Received** {sentiment_emoji}
            - **Pairs**: {sig.get('symbols', 'N/A')}
            - **Sentiment**: {sig.get('sentiment', 'neutral').upper()}
            - **Impact**: {impact_color} {sig.get('impact', 'unknown').upper()}
            - **Confidence**: {sig.get('confidence', 0):.1%}
            - **Headline**: _{sig.get('headline', 'No headline')}_
            - **Action**: {action_taken}
            """)
    
    # Display recent news signals history
    if st.session_state.news_signals:
        with st.expander("📋 Recent News Signals History (Last 5)"):
            for i, sig in enumerate(reversed(st.session_state.news_signals[-5:]), 1):
                action = sig.get('action_taken', 'No action recorded')
                st.markdown(f"""
                **{i}. [{sig['timestamp'].strftime('%H:%M:%S')}]** {sig['sentiment'].upper()} - {sig['symbols']}  
                📰 {sig['headline'][:60]}...  
                ⚡ {action}
                """)
    
    # Display active trading opportunities
    if 'trading_opportunities' in st.session_state:
        active_opps = [opp for opp in st.session_state.trading_opportunities if not opp['expired']]
        if active_opps:
            with st.expander("🎯 Active News-Based Trading Opportunities"):
                for opp in active_opps[-5:]:
                    direction_emoji = "🟢" if opp['direction'] == 'buy' else "🔴"
                    time_ago = (datetime.now() - opp['timestamp']).total_seconds() / 60
                    st.info(f"""
                    {direction_emoji} **{opp['symbol']}** - {opp['direction'].upper()}  
                    Strength: {opp['strength'].upper()} | Confidence: {opp['confidence']:.1%}  
                    Posted: {time_ago:.0f} minutes ago
                    """)
    
    if not signals_list and not st.session_state.news_signals:
        st.info("No news signals received yet. Waiting for n8n webhook data...")

    st.markdown("---")

    # ==================== MARKET ANALYSIS & CHARTING ====================
    st.subheader(f"Market Analysis: {symbol}")
    df = bot.fetch_market_data(symbol)
    df = bot.calculate_indicators(df)
    
    # Chart
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['open'], high=df['high'],
                low=df['low'], close=df['close'], name='Price')])
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_12'], name='EMA 12', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_26'], name='EMA 26', line=dict(color='blue')))
    fig.update_layout(height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # ML Training & Signals
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Train ML Model"):
            df_ml = bot.prepare_ml_data(df)
            success = bot.train_ml_model(symbol, df_ml)
            if success:
                st.success(f"Model trained for {symbol}")
            else:
                st.warning("Not enough data to train model")
                
    with col_b:
        if st.button("Generate Signal"):
            signal_result = bot.generate_signal(symbol, df)
            
            # Display comprehensive signal information
            st.info(f"**Signal**: {signal_result['signal'].upper()}")
            st.metric("Confidence", f"{signal_result['confidence']:.1%}")
            st.metric("Position Size", f"{signal_result['position_multiplier']:.0%} of normal")
            
            if signal_result['reasoning']:
                st.write("**Reasoning:**")
                for reason in signal_result['reasoning']:
                    st.write(f"- {reason}")
    
    with col_c:
        if st.button("Clear Old Signals"):
            # Clear expired opportunities and old news
            if 'trading_opportunities' in st.session_state:
                st.session_state.trading_opportunities = [
                    opp for opp in st.session_state.trading_opportunities 
                    if not opp['expired']
                ]
            st.session_state.news_signals = []
            st.success("Cleared old signals")

if __name__ == "__main__":
    main()
