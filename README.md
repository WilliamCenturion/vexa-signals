
# VEXA Signals

### Cryptocurrency Market Intelligence & Trading Signal Scanner

VEXA Signals is a Python-based cryptocurrency market scanner designed to identify potential trading opportunities using real-time market data from the Hyperliquid API.

The project combines technical indicators, market volume analysis and trend confirmation to generate LONG and SHORT signals through a custom dashboard.

## Features

- Hyperliquid API integration
- Automated cryptocurrency market scanning
- EMA 9, 21, 50 and 200
- RSI (14) analysis
- ADX (14) trend strength
- ATR-based Stop Loss and Take Profit
- Relative volume analysis
- LONG / SHORT / WAIT classification
- Confluence scoring system
- Custom Black & Yellow dashboard
- Automatic market monitoring

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| Pandas | Market data processing |
| NumPy | Numerical calculations |
| Requests | API integration |
| Google Colab | Cloud development environment |
| Hyperliquid API | Cryptocurrency market data |

## Technical Strategy

The scanner evaluates multiple technical conditions:

- EMA alignment for trend identification.
- RSI for momentum confirmation.
- ADX for trend strength.
- Relative volume for market participation.
- ATR for volatility-based risk levels.

Signals are classified as LONG, SHORT or WAIT based on the combined criteria.

## Dashboard

The project includes a custom dashboard displaying:

- Active trading signals
- Market rankings
- Confluence scores
- Entry reference prices
- Stop Loss levels
- Take Profit targets

## Project Status

**Version 1.0 — Experimental / Under Validation**

VEXA Signals is currently being tested and improved.

The project provides analytical signals only. It does not execute trades automatically and does not guarantee trading performance.

## Disclaimer

This project is provided for educational and research purposes. Cryptocurrency trading involves substantial financial risk. Always validate signals and manage risk independently.

## Author

**William Centurion**

GitHub: https://github.com/WilliamCenturion
