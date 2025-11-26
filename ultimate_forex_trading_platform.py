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
from dataclasses import dataclass
from typing import Dict, List
from enum import Enum

# --- TECHNICAL ANALYSIS LIBRARY IMPORTS (The fix for red lines) ---
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
            'strategy_params': self.get_default_strategy_params()
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def get_default_strategy_params(self):
        return {'ema_fast': 12, 'ema_slow': 26, 'rsi_period': 14}

    # ==================== MT5 CONNECTION (UPDATED) ====================

    def connect_mt5(self) -> bool:
        """Connect to MetaTrader 5, checking for running terminal and account login."""
        
        # NOTE: If this default path is wrong for you, change it to your actual path!
        # The path should point to the terminal64.exe file.
        # Example: r"C:\Program Files\YourBrokerName MT5\terminal64.exe"
        default_path = None 
        
        # Common Windows 64-bit default path
        win_path = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"

        if not mt5.initialize(path=default_path):
            # Try with the common Windows path if default failed
            if not mt5.initialize(path=win_path):
                st.error("❌ MT5 initialization failed. Ensure MT5 is installed and running.")
                return False
        
        # Check if an account is actually logged in after initialization
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
                # Using H1 timeframe (Hourly) as a reasonable default
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
            # Ensure we fill any missing data first to avoid errors
            df = df.fillna(method='ffill').fillna(method='bfill')

            # 1. Moving Averages
            df['ema_12'] = EMAIndicator(close=df['close'], window=12).ema_indicator()
            df['ema_26'] = EMAIndicator(close=df['close'], window=26).ema_indicator()
            
            # 2. RSI
            df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
            
            # 3. MACD
            macd = MACD(close=df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_hist'] = macd.macd_diff()

            # 4. Bollinger Bands
            bb = BollingerBands(close=df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()

            # 5. ATR
            atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['atr'] = atr.average_true_range()
            
            # Clean up NaNs generated by indicators (e.g. first 26 rows)
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
            # Ensure features exist
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
        """Completed logic for technical analysis"""
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            signal = "none"
            confidence = 0.0

            # EMA Crossover Logic
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
            # Fallback for insufficient data
            return {"signal": "none", "confidence": 0}

    def generate_signal(self, symbol: str, df: pd.DataFrame):
        ml_res = self.ml_predict(symbol, df)
        ta_res = self.technical_analysis_signal(df)
        
        # Simple voting mechanism
        final_signal = "none"
        if ml_res['signal'] == ta_res['signal'] and ml_res['signal'] != "none":
            final_signal = ml_res['signal']
        elif ta_res['confidence'] > 0.7:
            final_signal = ta_res['signal']
            
        return final_signal

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

    # Main Dashboard
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Balance", f"${st.session_state.account_info['balance']:.2f}")
    with col2:
        st.metric("Equity", f"${st.session_state.account_info['equity']:.2f}")
    with col3:
        st.metric("Status", "🟢 Connected" if bot.connected else "🔴 Disconnected")

    # Data & Charting
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
    col_a, col_b = st.columns(2)
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
            sig = bot.generate_signal(symbol, df)
            st.info(f"AI Prediction: {sig.upper()}")

if __name__ == "__main__":
    main()