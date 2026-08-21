(function () {
    "use strict";

    function describe(evt) {
        var detail = evt.detail || {};
        var xhr = detail.xhr;
        var path = (detail.pathInfo && detail.pathInfo.requestPath) || "";
        if (!xhr) return null;

        var rowMatch = null;
        try {
            var text = xhr.responseText || "";
            var rows = (text.match(/<tr/g) || []).length;
            if (rows > 0) rowMatch = rows;
        } catch (e) { /* ignore */ }

        var label = path.replace("/api/page/", "").replace(/-/g, " ");
        if (rowMatch) {
            return label + ": " + rowMatch.toLocaleString() + " rows loaded";
        }
        var chars = (xhr.responseText || "").length;
        if (chars > 0) {
            return label + " loaded (" + chars.toLocaleString() + " chars)";
        }
        return null;
    }

    document.body.addEventListener("htmx:afterSwap", function (evt) {
        var msg = describe(evt);
        if (!msg) return;
        window.dispatchEvent(new CustomEvent("rpd-toast", { detail: msg }));
    });
})();
