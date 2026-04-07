FROM freqtradeorg/freqtrade:stable
COPY shared/ /freqtrade/shared/
COPY services/crypto-signals/config.json /freqtrade/config.json
COPY services/crypto-signals/strategies/ /freqtrade/user_data/strategies/
RUN pip install --no-cache-dir ccxt cryptocompare httpx circuitbreaker tenacity cachetools websockets redis[hiredis] pydantic-settings
COPY services/crypto-signals/fallback_scanner.py /freqtrade/
COPY services/crypto-signals/ws_monitor.py /freqtrade/
COPY --chmod=755 services/crypto-signals/entrypoint.sh /freqtrade/
ENTRYPOINT []
CMD ["bash", "/freqtrade/entrypoint.sh"]
