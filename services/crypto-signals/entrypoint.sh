#!/bin/bash
# Start fallback scanner in background
python /freqtrade/fallback_scanner.py &
# Start WebSocket monitor in background
python /freqtrade/ws_monitor.py &
# Start Freqtrade (foreground)
freqtrade trade --config /freqtrade/config.json --strategy NovaFXCryptoStrategy
