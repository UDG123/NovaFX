FROM freqtradeorg/freqtrade:stable
COPY config.json /freqtrade/config.json
COPY strategies/ /freqtrade/user_data/strategies/
# Install fallback dependencies
RUN pip install --no-cache-dir ccxt cryptocompare httpx circuitbreaker tenacity cachetools websockets
COPY fallback_scanner.py /freqtrade/
COPY ws_monitor.py /freqtrade/
COPY ../../shared /freqtrade/shared/
# Entrypoint runs Freqtrade + fallback monitor
COPY entrypoint.sh /freqtrade/
RUN chmod +x /freqtrade/entrypoint.sh
CMD ["/freqtrade/entrypoint.sh"]
