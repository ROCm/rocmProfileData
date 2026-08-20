
(function () {
    "use strict";

    var STORAGE_KEY = "rpd_chat_history";

    function ChatUI() {
        this.messages = this._load();
        this.eventSource = null;
        this.active = false;
        this.logLines = [];
        this._bindEvents();
        this._render();
    }

    ChatUI.prototype._load = function () {
        try {
            var raw = sessionStorage.getItem(STORAGE_KEY);
            return raw ? JSON.parse(raw) : [];
        } catch (e) {
            return [];
        }
    };

    ChatUI.prototype._save = function () {
        try {
            sessionStorage.setItem(STORAGE_KEY, JSON.stringify(this.messages));
        } catch (e) { /* ignore */ }
    };

    ChatUI.prototype._el = function (id) {
        return document.getElementById(id);
    };

    ChatUI.prototype._render = function () {
        var container = this._el("chat-messages");
        if (!container) return;

        if (this.messages.length === 0 && !this.streamingContent) {
            container.innerHTML = '<div style="color:#666;text-align:center;padding:40px 0">Start a conversation about this trace file.</div>';
            return;
        }

        var html = "";
        for (var i = 0; i < this.messages.length; i++) {
            var msg = this.messages[i];
            if (msg.role === "user") {
                html += '<div class="chat-bubble user">' + this._esc(this._textContent(msg.content)) + "</div>";
            } else {
                html += '<div class="chat-bubble assistant"><div class="chat-md">' + this._esc(msg.content) + "</div></div>";
            }
        }
        if (this.streamingContent !== undefined && this.streamingContent !== null) {
            html += '<div class="chat-bubble assistant" id="chat-streaming-bubble"><div class="chat-md">'
                + this._esc(this.streamingContent) + "</div></div>";
        }
        container.innerHTML = html;
        container.scrollTop = container.scrollHeight;

        // Render markdown in assistant bubbles
        this._renderMarkdown();
    };

    ChatUI.prototype._renderStreamingBubble = function () {
        var container = this._el("chat-messages");
        if (!container) return;
        var bubble = this._el("chat-streaming-bubble");
        if (!bubble) {
            this._render();
            return;
        }
        var md = bubble.querySelector(".chat-md");
        if (md) md.textContent = this.streamingContent;
        container.scrollTop = container.scrollHeight;
    };

    ChatUI.prototype._renderMarkdown = function () {
        var els = this._el("chat-messages").getElementsByClassName("chat-md");
        for (var i = 0; i < els.length; i++) {
            // Simple markdown: tables, code blocks, bold, lists
            var text = els[i].innerHTML;
            // Code blocks
            text = text.replace(/```(\w*)\n([\s\S]*?)```/g, function (_, lang, code) {
                return '<div class="copy-block">'
                    + '<button class="copy-btn" data-copy-target>Copy</button>'
                    + '<pre style="background:#f0f0f0;padding:8px;border-radius:4px;overflow-x:auto;font-size:12px"><code>' + code.replace(/\n$/, "") + "</code></pre>"
                    + "</div>";
            });
            // Inline code
            text = text.replace(/`([^`]+)`/g, '<code style="background:#f0f0f0;padding:1px 4px;border-radius:3px;font-size:12px">$1</code>');
            // Bold
            text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
            // Italic
            text = text.replace(/\*([^*]+)\*/g, "<em>$1</em>");
            // Tables
            text = this._renderTable(text);
            // Line breaks
            text = text.replace(/\n/g, "<br>");
            els[i].innerHTML = text;
        }
    };

    ChatUI.prototype._renderTable = function (html) {
        var tableRe = /((?:<br>[|].*<br>)*)/g;
        return html.replace(tableRe, function (match) {
            var lines = match.replace(/<br>/g, "\n").split("\n").filter(function (l) {
                return l.trim() !== "" && !/^[-:\s]+$/.test(l.trim());
            });
            if (lines.length < 2) return match;
            var t = '<table style="border-collapse:collapse;margin:8px 0;font-size:13px">';
            for (var i = 0; i < lines.length; i++) {
                var cells = lines[i].split("|").filter(function (c) {
                    return c.trim() !== "";
                });
                t += "<tr>";
                var tag = i === 0 ? "th" : "td";
                for (var j = 0; j < cells.length; j++) {
                    var style = i === 0
                        ? "font-weight:bold;border:1px solid #ddd;padding:6px 10px;text-align:left;background:#fafafa"
                        : "border:1px solid #ddd;padding:6px 10px";
                    t += "<" + tag + ' style="' + style + '">' + cells[j].trim() + "</" + tag + ">";
                }
                t += "</tr>";
            }
            t += "</table>";
            return t;
        });
    };

    ChatUI.prototype._esc = function (s) {
        var div = document.createElement("div");
        div.appendChild(document.createTextNode(s));
        return div.innerHTML;
    };

    ChatUI.prototype._textContent = function (s) {
        return s;
    };

    ChatUI.prototype._scroll = function () {
        var c = this._el("chat-messages");
        if (c) c.scrollTop = c.scrollHeight;
    };

    ChatUI.prototype._setInputEnabled = function (enabled) {
        var inp = this._el("chat-input");
        var btn = this._el("chat-send-btn");
        if (inp) inp.disabled = !enabled;
        if (btn) btn.disabled = !enabled;
    };

    ChatUI.prototype._setSpinner = function (show) {
        var sp = this._el("chat-spinner");
        if (sp) sp.style.display = show ? "block" : "none";
    };

    ChatUI.prototype._setCancelBtn = function (show) {
        var btn = this._el("chat-cancel-btn");
        if (btn) btn.style.display = show ? "inline-block" : "none";
    };

    ChatUI.prototype._appendLog = function (text) {
        this.logLines.push(text);
        var panel = this._el("chat-progress");
        if (panel) {
            panel.textContent = this.logLines.join("\n");
            panel.scrollTop = panel.scrollHeight;
        }
    };

    ChatUI.prototype._toggleLog = function () {
        var panel = this._el("chat-progress");
        if (panel) {
            if (panel.style.display === "none" || panel.style.display === "") {
                panel.style.display = "block";
            } else {
                panel.style.display = "none";
            }
        }
    };

    ChatUI.prototype._bindEvents = function () {
        var self = this;

        var sendBtn = this._el("chat-send-btn");
        if (sendBtn) {
            sendBtn.addEventListener("click", function () {
                self._send();
            });
        }

        var clearBtn = this._el("chat-clear-btn");
        if (clearBtn) {
            clearBtn.addEventListener("click", function () {
                self._clear();
            });
        }

        var cancelBtn = this._el("chat-cancel-btn");
        if (cancelBtn) {
            cancelBtn.addEventListener("click", function () {
                self._cancel();
            });
        }

        var logBtn = this._el("chat-log-btn");
        if (logBtn) {
            logBtn.addEventListener("click", function () {
                self._toggleLog();
            });
        }

        var inp = this._el("chat-input");
        if (inp) {
            inp.addEventListener("keydown", function (e) {
                if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    self._send();
                }
            });
        }

        document.addEventListener("click", function (e) {
            var btn = e.target.closest && e.target.closest(".copy-btn");
            if (!btn) return;
            var block = btn.closest(".copy-block");
            var code = block && block.querySelector("code");
            if (!code) return;
            navigator.clipboard.writeText(code.textContent).then(function () {
                var original = btn.textContent;
                btn.textContent = "Copied!";
                setTimeout(function () {
                    btn.textContent = original;
                }, 1500);
            });
        });
    };

    ChatUI.prototype._send = function () {
        var inp = this._el("chat-input");
        if (!inp || inp.disabled) return;
        var text = inp.value.trim();
        if (!text) return;

        inp.value = "";
        this.streamingContent = null;
        this.messages.push({ role: "user", content: text });
        this._save();
        this._render();
        this._setInputEnabled(false);
        this._setSpinner(true);
        this._setCancelBtn(true);
        this.logLines = [];

        this._appendLog("Sending...");
        this._toggleLog();

        var self = this;
        var xhr = new XMLHttpRequest();
        xhr.open("POST", "/api/chat/send", true);
        xhr.setRequestHeader("Content-Type", "application/json");
        xhr.timeout = 180000;
        this._sseCursor = 0;

        xhr.onprogress = function () {
            if (xhr.status && xhr.status !== 200) return;
            self._parseSSEIncremental(xhr.responseText);
        };

        xhr.onload = function () {
            if (xhr.status !== 200) {
                self._appendLog("HTTP error: " + xhr.status + " " + xhr.statusText);
                if (xhr.responseText) {
                    self._appendLog("Response: " + xhr.responseText.substring(0, 500));
                }
                self._done(false);
                return;
            }
            self._parseSSEIncremental(xhr.responseText);
        };

        xhr.onerror = function () {
            self._appendLog("Request error — check server is running.");
            self._done(false);
        };

        xhr.ontimeout = function () {
            self._appendLog("Request timeout (180s).");
            self._done(false);
        };

        xhr.send(JSON.stringify({ text: text, history: self.messages.slice(0, -1) }));
    };

    // Incrementally parse newly-arrived SSE bytes as they stream in over XHR.
    ChatUI.prototype._parseSSEIncremental = function (fullText) {
        var chunk = fullText.slice(this._sseCursor);
        if (!chunk) return;

        var lastBoundary = chunk.lastIndexOf("\n\n");
        if (lastBoundary === -1) return;

        var complete = chunk.slice(0, lastBoundary + 2);
        this._sseCursor += complete.length;

        var lines = complete.split("\n");
        for (var i = 0; i < lines.length; i++) {
            if (lines[i].indexOf("data: ") === 0) {
                var data = lines[i].slice(6);
                try {
                    var ev = JSON.parse(data);
                    this._handleEvent(ev);
                } catch (e) { /* skip keep-alive/comment lines */ }
            }
        }
    };

    ChatUI.prototype._handleEvent = function (ev) {
        switch (ev.type) {
            case "start":
                this._appendLog("Session: " + ev.sessionId);
                break;
            case "progress":
            case "info":
                this._appendLog(ev.text);
                break;
            case "sql":
                this._appendLog("[SQL] " + ev.text);
                break;
            case "token":
                if (!this.streamingContent) this._setSpinner(false);
                this.streamingContent = (this.streamingContent || "") + ev.content;
                this._renderStreamingBubble();
                break;
            case "answer":
                this.streamingContent = null;
                this.messages.push({ role: "assistant", content: ev.content });
                this._save();
                this._render();
                break;
            case "done":
                this._done(true);
                break;
            case "error":
                this.streamingContent = null;
                this._appendLog("Error: " + ev.message);
                this.messages.push({ role: "assistant", content: "**Error:** " + ev.message });
                this._save();
                this._render();
                this._done(true);
                break;
            case "stream_end":
                this._done(true);
                break;
        }
    };

    ChatUI.prototype._done = function (ok) {
        this._setSpinner(false);
        this._setCancelBtn(false);
        this._setInputEnabled(true);
        if (ok) this._appendLog("---");
    };

    ChatUI.prototype._cancel = function () {
        this._appendLog("Cancelling...");
        this._setSpinner(false);
        this._setCancelBtn(false);
        this._setInputEnabled(true);
        this.streamingContent = null;
        window.dispatchEvent(new CustomEvent("rpd-toast", { detail: "Request cancelled" }));
        if (this.currentSession) {
            var self = this;
            var xhr = new XMLHttpRequest();
            xhr.open("POST", "/api/chat/cancel/" + this.currentSession, true);
            xhr.send();
        }
    };

    ChatUI.prototype._clear = function () {
        this.messages = [];
        this.streamingContent = null;
        this._save();
        this._render();
        this.logLines = [];
        var panel = this._el("chat-progress");
        if (panel) panel.textContent = "";
        var inp = this._el("chat-input");
        if (inp) inp.value = "";
    };

    // Store session id for cancel
    var origHandle = ChatUI.prototype._handleEvent;
    ChatUI.prototype._handleEvent = function (ev) {
        if (ev.type === "start") this.currentSession = ev.sessionId;
        origHandle.call(this, ev);
    };

    function tryInit() {
        if (document.getElementById("chat-input")) {
            window._chatUI = new ChatUI();
            return true;
        }
        return false;
    }

    // Initialize on DOMContentLoaded, or wait for chat elements via MutationObserver
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", function () {
            if (!tryInit()) {
                var observer = new MutationObserver(function (mutations, obs) {
                    if (tryInit()) {
                        obs.disconnect();
                    }
                });
                observer.observe(document.body, { childList: true, subtree: true });
            }
        });
    } else {
        tryInit();
    }
})();
