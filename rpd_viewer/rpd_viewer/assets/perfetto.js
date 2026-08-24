window.dash_clientside = window.dash_clientside || {};
window.dash_clientside.perfetto = {
    open_trace: function(n) {
        if (!n) return '';
        var ORIGIN = 'https://ui.perfetto.dev';
        var log = document.getElementById('trace-log');
        if (log) log.textContent = 'Fetching trace JSON...\n';

        fetch('/tracedata')
            .then(function(resp) { return resp.blob(); })
            .then(function(blob) {
                if (log) log.textContent += 'Buffering (' + (blob.size / 1e6).toFixed(1) + ' MB)...\n';
                return blob.arrayBuffer();
            })
            .then(function(arrayBuffer) {
                if (log) log.textContent += 'Opening ui.perfetto.dev...\n';
                var win = window.open(ORIGIN);
                if (!win) {
                    if (log) log.textContent += 'Popup blocked! Please allow popups for this site.\n';
                    return;
                }
                var timer = setInterval(function() { win.postMessage('PING', ORIGIN); }, 50);
                window.addEventListener('message', function handler(evt) {
                    if (evt.data !== 'PONG') return;
                    clearInterval(timer);
                    window.removeEventListener('message', handler);
                    win.postMessage({
                        perfetto: {
                            buffer: arrayBuffer,
                            title: 'RPD Trace',
                        }
                    }, ORIGIN);
                    if (log) log.textContent += 'Trace sent to Perfetto.\n';
                });
            })
            .catch(function(e) {
                if (log) log.textContent += 'Error: ' + e.message + '\n';
            });
        return 'Fetching...';
    }
};
