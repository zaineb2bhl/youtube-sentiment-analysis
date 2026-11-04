// Content script : extrait les commentaires de la page YouTube.
(() => {
  if (window.__ytSentimentLoaded) return;
  window.__ytSentimentLoaded = true;

  const COMMENT_SELECTOR =
    "ytd-comment-thread-renderer #content-text, ytd-comment-view-model #content-text, ytd-comment-renderer #content-text";

  function extractComments(includeReplies) {
    const seen = new Set();
    const out = [];
    document.querySelectorAll(COMMENT_SELECTOR).forEach((el) => {
      const text = (el.innerText || "").trim();
      if (!text) return;
      const isReply = !!el.closest("#replies, ytd-comment-replies-renderer");
      if (isReply && !includeReplies) return;
      const container = el.closest("ytd-comment-view-model, ytd-comment-renderer");
      const author = container?.querySelector("#author-text")?.innerText?.trim() || "";
      const likes = container?.querySelector("#vote-count-middle")?.innerText?.trim() || "0";
      const key = author + "|" + text;
      if (seen.has(key)) return;
      seen.add(key);
      out.push({ text, author, likes, isReply });
    });
    return out;
  }

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // Les commentaires YouTube sont chargés dynamiquement au scroll (lazy loading) :
  // on fait défiler la page jusqu'à atteindre le nombre voulu ou ne plus rien charger.
  async function loadComments(max, includeReplies) {
    const startY = window.scrollY;
    let last = -1;
    let stagnant = 0;
    const section = document.querySelector("ytd-comments#comments");
    if (section) section.scrollIntoView();
    await sleep(800);
    while (stagnant < 3) {
      const n = extractComments(includeReplies).length;
      if (n >= max) break;
      stagnant = n === last ? stagnant + 1 : 0;
      last = n;
      window.scrollTo(0, document.documentElement.scrollHeight);
      await sleep(1200);
    }
    window.scrollTo(0, startY);
    return extractComments(includeReplies).slice(0, max);
  }

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg.action === "ping") {
      sendResponse({ ok: true });
      return;
    }
    if (msg.action === "getComments") {
      const title = document.querySelector("h1.ytd-watch-metadata, h1.title")?.innerText?.trim() || document.title;
      const run = msg.autoScroll
        ? loadComments(msg.max, msg.includeReplies)
        : Promise.resolve(extractComments(msg.includeReplies).slice(0, msg.max));
      run.then((comments) => sendResponse({ ok: true, title, comments }))
         .catch((e) => sendResponse({ ok: false, error: String(e) }));
      return true; // réponse asynchrone
    }
  });
})();
