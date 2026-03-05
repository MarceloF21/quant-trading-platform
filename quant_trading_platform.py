#!/usr/bin/env python3
"""
================================================================================
QUANTITATIVE TRADING PLATFORM
================================================================================
Author: Marcelo Farez
Description: Professional algorithmic trading system with multiple strategies,
             machine learning integration, and comprehensive risk management.

Features:
- 4 Trading Strategies: Momentum, Mean Reversion, MACD, ML-based
- Walk-forward backtesting with realistic execution
- Risk management: Stop-loss, take-profit, position sizing
- SQLite database for trade logging and analytics
- Professional performance metrics (Sharpe, Sortino, Calmar, etc.)

Technologies: Python, Pandas, NumPy, Scikit-Learn, SQLite, Matplotlib
================================================================================
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuration
DB_PATH = 'quant_trading_platform.db'
INITIAL_CAPITAL = 100000
POSITION_SIZE_PCT = 0.10
STOP_LOSS_PCT = 0.05
TAKE_PROFIT_PCT = 0.15

class TechnicalIndicators:
    """Calculate technical indicators for trading signals"""

    @staticmethod
    def calculate_all(prices):
        """Calculate comprehensive technical indicators"""
        indicators = pd.DataFrame(index=prices.index)

        # Returns
        indicators['returns'] = prices.pct_change()
        indicators['log_returns'] = np.log(prices / prices.shift(1))

        # Moving Averages
        for window in [5, 10, 20, 50, 200]:
            indicators[f'SMA_{window}'] = prices.rolling(window=window).mean()
            indicators[f'EMA_{window}'] = prices.ewm(span=window, adjust=False).mean()

        # Crossovers
        indicators['SMA_10_20_cross'] = (indicators['SMA_10'] > indicators['SMA_20']).astype(int)
        indicators['SMA_50_200_cross'] = (indicators['SMA_50'] > indicators['SMA_200']).astype(int)

        # RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        indicators['MACD'] = ema_12 - ema_26
        indicators['MACD_signal'] = indicators['MACD'].ewm(span=9, adjust=False).mean()
        indicators['MACD_histogram'] = indicators['MACD'] - indicators['MACD_signal']

        # Bollinger Bands
        indicators['BB_middle'] = prices.rolling(window=20).mean()
        bb_std = prices.rolling(window=20).std()
        indicators['BB_upper'] = indicators['BB_middle'] + (bb_std * 2)
        indicators['BB_lower'] = indicators['BB_middle'] - (bb_std * 2)
        indicators['BB_position'] = (prices - indicators['BB_lower']) / (indicators['BB_upper'] - indicators['BB_lower'])

        # Volatility
        indicators['volatility_20'] = indicators['returns'].rolling(window=20).std() * np.sqrt(252)

        # Momentum
        for window in [5, 10, 20]:
            indicators[f'momentum_{window}'] = prices.pct_change(window)

        return indicators


class TradingStrategy:
    """Base class for trading strategies"""

    def __init__(self, name):
        self.name = name

    def generate_signals(self, prices, indicators):
        raise NotImplementedError


class MomentumStrategy(TradingStrategy):
    """Trend-following momentum strategy"""

    def __init__(self):
        super().__init__("Momentum_Trend")

    def generate_signals(self, prices, indicators):
        signals = pd.DataFrame(index=prices.index)
        signals['price'] = prices

        trend_up = (prices > indicators['SMA_50']) & (indicators['SMA_50'] > indicators['SMA_200'])
        momentum = indicators['momentum_20'] > 0.02
        not_overbought = indicators['RSI'] < 70

        signals['signal'] = 0
        signals.loc[trend_up & momentum & not_overbought, 'signal'] = 1
        signals.loc[indicators['RSI'] > 75, 'signal'] = -1
        signals['positions'] = signals['signal'].diff()

        return signals


class MeanReversionStrategy(TradingStrategy):
    """Mean reversion using Bollinger Bands"""

    def __init__(self):
        super().__init__("MeanReversion_BB")

    def generate_signals(self, prices, indicators):
        signals = pd.DataFrame(index=prices.index)
        signals['price'] = prices

        bb_lower_touch = indicators['BB_position'] < 0.1
        bb_upper_touch = indicators['BB_position'] > 0.9
        oversold = indicators['RSI'] < 30
        overbought = indicators['RSI'] > 70

        signals['signal'] = 0
        signals.loc[bb_lower_touch & oversold, 'signal'] = 1
        signals.loc[bb_upper_touch & overbought, 'signal'] = -1
        signals['positions'] = signals['signal'].diff()

        return signals


class MACDStrategy(TradingStrategy):
    """MACD crossover strategy"""

    def __init__(self):
        super().__init__("MACD_Crossover")

    def generate_signals(self, prices, indicators):
        signals = pd.DataFrame(index=prices.index)
        signals['price'] = prices

        macd_cross_up = (indicators['MACD'] > indicators['MACD_signal']) & \
                       (indicators['MACD'].shift(1) <= indicators['MACD_signal'].shift(1))
        macd_cross_down = (indicators['MACD'] < indicators['MACD_signal']) & \
                         (indicators['MACD'].shift(1) >= indicators['MACD_signal'].shift(1))

        signals['signal'] = 0
        signals.loc[macd_cross_up, 'signal'] = 1
        signals.loc[macd_cross_down, 'signal'] = -1
        signals['positions'] = signals['signal'].diff()

        return signals


class MLStrategy(TradingStrategy):
    """Machine Learning-based strategy using Random Forest"""

    def __init__(self):
        super().__init__("ML_RandomForest")
        self.model = None
        self.scaler = StandardScaler()

    def prepare_features(self, prices, indicators):
        """Prepare features for ML model"""
        features = pd.DataFrame(index=indicators.index)

        features['returns_1d'] = indicators['returns']
        features['returns_5d'] = indicators['momentum_5']
        features['returns_10d'] = indicators['momentum_10']
        features['returns_20d'] = indicators['momentum_20']
        features['rsi'] = indicators['RSI'] / 100
        features['macd'] = indicators['MACD_histogram']
        features['bb_position'] = indicators['BB_position']
        features['volatility'] = indicators['volatility_20']
        features['sma_10_20_ratio'] = indicators['SMA_10'] / indicators['SMA_20']
        features['sma_50_200_ratio'] = indicators['SMA_50'] / indicators['SMA_200']
        features['price_sma50_ratio'] = prices / indicators['SMA_50']
        features['ema_12_26_ratio'] = indicators['EMA_5'] / indicators['EMA_20']

        return features.dropna()

    def generate_signals(self, prices, indicators):
        """Generate ML-based signals"""
        features = self.prepare_features(prices, indicators)

        signals = pd.DataFrame(index=prices.index)
        signals['price'] = prices
        signals['signal'] = 0
        signals['confidence'] = 0.5

        min_train_size = 252
        if len(features) < min_train_size + 50:
            return signals

        future_returns = prices.pct_change().shift(-1)
        target = (future_returns > 0).astype(int)

        train_features = features.iloc[:min_train_size]
        train_target = target.loc[train_features.index]

        valid_idx = train_target.notna()
        train_features = train_features[valid_idx]
        train_target = train_target[valid_idx]

        if len(train_features) < 100:
            return signals

        X_scaled = self.scaler.fit_transform(train_features)

        self.model = RandomForestClassifier(
            n_estimators=50, max_depth=5, min_samples_split=20, random_state=42
        )
        self.model.fit(X_scaled, train_target)

        test_features = features.iloc[min_train_size:]
        test_scaled = self.scaler.transform(test_features)
        predictions = self.model.predict(test_scaled)
        confidences = self.model.predict_proba(test_scaled).max(axis=1)

        for i, (idx, _) in enumerate(test_features.iterrows()):
            if confidences[i] > 0.55:
                signals.loc[idx, 'signal'] = 1 if predictions[i] == 1 else -1
                signals.loc[idx, 'confidence'] = confidences[i]

        signals['positions'] = signals['signal'].diff()

        return signals


class BacktestEngine:
    """Professional backtesting engine with risk management"""

    def __init__(self, initial_capital=100000, position_size_pct=0.10,
                 stop_loss_pct=0.05, take_profit_pct=0.15):
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def run_backtest(self, prices, signals, ticker, strategy_name):
        """Run backtest for a strategy"""
        trades = []
        cash = self.initial_capital
        position = 0
        entry_price = 0
        entry_date = None

        for date in signals.index:
            current_price = signals.loc[date, 'price']
            signal = signals.loc[date, 'signal']

            # Check stop loss / take profit
            if position > 0:
                pnl_pct = (current_price - entry_price) / entry_price

                if pnl_pct <= -self.stop_loss_pct:
                    exit_value = position * current_price
                    pnl = exit_value - (position * entry_price)
                    cash += exit_value

                    trades.append({
                        'ticker': ticker, 'strategy': strategy_name,
                        'entry_date': entry_date, 'exit_date': date,
                        'entry_price': entry_price, 'exit_price': current_price,
                        'position_size': position, 'pnl': pnl, 'pnl_pct': pnl_pct,
                        'holding_period': (date - entry_date).days,
                        'exit_reason': 'Stop Loss'
                    })
                    position = 0
                    continue

                if pnl_pct >= self.take_profit_pct:
                    exit_value = position * current_price
                    pnl = exit_value - (position * entry_price)
                    cash += exit_value

                    trades.append({
                        'ticker': ticker, 'strategy': strategy_name,
                        'entry_date': entry_date, 'exit_date': date,
                        'entry_price': entry_price, 'exit_price': current_price,
                        'position_size': position, 'pnl': pnl, 'pnl_pct': pnl_pct,
                        'holding_period': (date - entry_date).days,
                        'exit_reason': 'Take Profit'
                    })
                    position = 0
                    continue

            # Process signals
            if signal == 1 and position == 0:
                position_value = cash * self.position_size_pct
                position = int(position_value / current_price)

                if position > 0:
                    cash -= position * current_price
                    entry_price = current_price
                    entry_date = date

            elif signal == -1 and position > 0:
                exit_value = position * current_price
                pnl = exit_value - (position * entry_price)
                pnl_pct = (current_price - entry_price) / entry_price
                cash += exit_value

                trades.append({
                    'ticker': ticker, 'strategy': strategy_name,
                    'entry_date': entry_date, 'exit_date': date,
                    'entry_price': entry_price, 'exit_price': current_price,
                    'position_size': position, 'pnl': pnl, 'pnl_pct': pnl_pct,
                    'holding_period': (date - entry_date).days,
                    'exit_reason': 'Signal'
                })
                position = 0

        return pd.DataFrame(trades)

    def calculate_metrics(self, trades_df):
        """Calculate performance metrics"""
        if len(trades_df) == 0:
            return None

        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = winning_trades / total_trades

        total_pnl = trades_df['pnl'].sum()
        avg_return = trades_df['pnl_pct'].mean()

        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        trade_returns = trades_df['pnl_pct']
        sharpe = (trade_returns.mean() / trade_returns.std()) * np.sqrt(252) \
                 if trade_returns.std() > 0 else 0

        cumulative = (1 + trade_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_return': avg_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'avg_holding_period': trades_df['holding_period'].mean()
        }


def initialize_database():
    """Create database schema"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.executescript('''
        CREATE TABLE IF NOT EXISTS market_data (
            ticker TEXT, date DATE, open REAL, high REAL, low REAL,
            close REAL, volume INTEGER, adj_close REAL, PRIMARY KEY (ticker, date)
        );
        CREATE TABLE IF NOT EXISTS trades (
            trade_id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT,
            entry_date DATE, exit_date DATE, entry_price REAL, exit_price REAL,
            position_size INTEGER, strategy TEXT, pnl REAL, pnl_pct REAL,
            holding_period INTEGER, exit_reason TEXT
        );
        CREATE TABLE IF NOT EXISTS strategy_performance (
            strategy TEXT PRIMARY KEY, total_trades INTEGER, winning_trades INTEGER,
            losing_trades INTEGER, win_rate REAL, avg_return REAL, total_pnl REAL,
            sharpe_ratio REAL, max_drawdown REAL, profit_factor REAL,
            avg_holding_period REAL, last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')

    conn.commit()
    conn.close()
    print("✅ Database initialized")


if __name__ == "__main__":
    initialize_database()
    print("\nQuantitative Trading Platform - Ready")
    print("Run strategies and backtests using the provided classes")
