#!/bin/bash
# Start background health responder immediately
python3 -c "
import http.server, os, threading
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'ok')
    def log_message(self, *a): pass
port = int(os.environ.get('PORT', 8080))
srv = http.server.HTTPServer(('0.0.0.0', port), H)
t = threading.Thread(target=srv.serve_forever, daemon=True)
t.start()
import time; time.sleep(999999)
" &
HEALTH_PID=$!

sleep 5  # brief pause before starting freqtrade

# Start fallback scanner in background
python /freqtrade/fallback_scanner.py &
# Start WebSocket monitor in background
python /freqtrade/ws_monitor.py &
# Start Freqtrade (foreground)
freqtrade trade --config /freqtrade/config.json --strategy NovaFXCryptoStrategy
