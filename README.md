# Quantitative Trading Platform

**Author:** Marcelo Farez  
**Technologies:** Python, Pandas, NumPy, Scikit-Learn, SQLite, Matplotlib  
**Domain:** Algorithmic Trading, Machine Learning, Risk Management

### What it does
Professional backtesting platform with **4 trading strategies** + machine learning.  
Includes realistic costs, risk controls, and full performance analytics.

### Key Results
- **Best Strategy (Mean Reversion BB):** Sharpe 7.348 | Win Rate 61.8% | Max DD -21.4%  
- **Total Trades:** 1,676  
- **Total P&L:** $332,711 (on $100K starting capital)  
- **Database:** 18,256 market records  
- **ML Strategy (Random Forest):** 66.2% accuracy

### Files in this repo
- `quant_trading_platform.py`     → Main trading engine & strategies  
- `quant_trading_platform.db`     → SQLite database with prices & trades  
- `trading_performance_dashboard.png` → 8-panel performance visuals  
- `trading_executive_summary.png` → One-page summary  
- `README.md`                     → This file  
- `requirements.txt`              → Dependencies

### Skills shown
- 4 quant strategies (Momentum, Mean Reversion, MACD, ML)  
- Machine Learning (Random Forest + walk-forward validation)  
- Risk controls (5% stop-loss, 15% take-profit, 10% position size)  
- Realistic backtesting (transaction costs modeled)  
- Professional metrics (Sharpe, Sortino, Calmar, Profit Factor)

Contact: marcelodavid1404@gmail.com  
GitHub: github.com/MarceloF21/quant-trading-platform
