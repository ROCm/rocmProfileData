(function () {
    "use strict";

    // Dash renders page content into the DOM via React *after* the initial
    // page load, so htmx's automatic DOMContentLoaded scan never sees
    // elements with `data-hx-*` attributes that Dash injects later (e.g.
    // when switching pages in the single-page app, or on first render of
    // a page whose layout is built server-side but mounted client-side).
    //
    // This bridge watches for DOM mutations and asks htmx to process any
    // newly-added element nodes, so `data-hx-get`/`data-hx-trigger` etc.
    // behave as if they were present at initial page load.
    function processNode(node) {
        if (!window.htmx || node.nodeType !== 1) return;
        window.htmx.process(node);
    }

    function start() {
        if (!document.body) {
            document.addEventListener("DOMContentLoaded", start);
            return;
        }

        // Process anything already in the DOM (covers the very first render).
        window.htmx && window.htmx.process(document.body);

        var observer = new MutationObserver(function (mutations) {
            for (var i = 0; i < mutations.length; i++) {
                var added = mutations[i].addedNodes;
                for (var j = 0; j < added.length; j++) {
                    processNode(added[j]);
                }
            }
        });

        observer.observe(document.body, { childList: true, subtree: true });
    }

    start();
})();
