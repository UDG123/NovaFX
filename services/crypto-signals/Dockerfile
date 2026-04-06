FROM freqtradeorg/freqtrade:stable
COPY shared/ /freqtrade/shared/
COPY services/crypto-signals/config.json /freqtrade/config.json
COPY services/crypto-signals/strategies/ /freqtrade/user_data/strategies/
RUN pip install --no-cache-dir ccxt cryptocompare httpx circuitbreaker tenacity cachetools websockets
COPY services/crypto-signals/fallback_scanner.py /freqtrade/
COPY services/crypto-signals/ws_monitor.py /freqtrade/
COPY services/crypto-signals/entrypoint.sh /freqtrade/
RUN chmod +x /freqtrade/entrypoint.sh
CMD ["/freqtrade/entrypoint.sh"]
