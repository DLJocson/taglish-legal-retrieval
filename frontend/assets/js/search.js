/**
 * Search page: single-model search, compare, filtered search, analytics pane, modal.
 */
const { $, escapeHtml, truncate, showToast, apiGet, apiPost, downloadBlob, scoreColor, updateTopKSlider } =
  LegalTech;

let models = [];
let filters = { languages: ["All"], document_types: ["All"], date_from: null, date_to: null };
let passageCache = new Map();
let previousTab = "search";
let currentTab = "search";
let lastExportResults = [];

const STOP_WORDS = new Set([
  "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
  "by", "from", "is", "are", "was", "were", "be", "been", "have", "has", "do", "does",
  "did", "will", "would", "could", "should", "may", "might", "must", "can", "this",
  "that", "these", "those", "what", "which", "who", "when", "where", "why", "how",
]);

function extractKeywords(query) {
  const words = query.toLowerCase().match(/\b\w+\b/g) || [];
  return words.filter((w) => w.length > 2 && !STOP_WORDS.has(w));
}

function highlightTextWithKeywords(text, keywords) {
  if (!keywords?.length) return text;
  const uniqueKeywords = [...new Set(keywords)];
  const pattern = new RegExp(
    `\\b(${uniqueKeywords.map((k) => k.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})\\b`,
    "gi"
  );
  return text.replace(pattern, '<mark class="query-highlight">$1</mark>');
}

function confClass(confidence) {
  const c = (confidence || "").toUpperCase();
  if (c === "HIGH") return "tag-conf-high";
  if (c === "MEDIUM") return "tag-conf-medium";
  return "tag-conf-low";
}

function cachePassage(result) {
  passageCache.set(result.passage_id, result);
}

function renderResultCard(result, query, compact = false) {
  cachePassage(result);
  const sc = scoreColor(result.score);
  const pct = Math.round(result.score * 100);
  const maxLen = compact ? 240 : 340;
  const keywords = query ? extractKeywords(query) : [];
  const excerpt = highlightTextWithKeywords(
    escapeHtml(truncate(result.passage_text, maxLen)),
    keywords
  );
  const title = escapeHtml(
    truncate(
      result.display_title ||
        result.short_title ||
        result.document_title ||
        result.document_type,
      120
    )
  );
  const conf = result.confidence
    ? `<span class="tag-conf ${confClass(result.confidence)}">${escapeHtml(result.confidence)}</span>`
    : "";
  const datePill =
    result.date_filed && result.date_filed !== "None"
      ? `<span class="tag-rect">${escapeHtml(result.date_filed)}</span>`
      : "";

  return `
    <article class="result-card" style="border-left-color:${sc.bar}">
      <div class="card-head">
        <div class="rank-badge">${result.rank}</div>
        <div class="card-body">
          <div class="card-title">${title}</div>
          <div class="card-meta">
            <span class="tag-id">${escapeHtml(result.passage_id)}</span>
            <span class="tag-score" style="background:${sc.bg};color:${sc.tx}">${Number(result.score).toFixed(4)}</span>
            ${conf}
            <span class="tag-pill">${escapeHtml(result.language)}</span>
            <span class="tag-rect">${escapeHtml(result.document_type)}</span>
            ${datePill}
            <div class="score-bar-group">
              <span class="score-pct">${pct}%</span>
              <div class="score-track">
                <div class="score-fill" style="width:${pct}%;background:${sc.bar}"></div>
              </div>
            </div>
          </div>
          <p class="card-text">${excerpt}</p>
          <button type="button" class="read-btn" data-passage-id="${escapeHtml(result.passage_id)}">
            <i class="ti ti-arrow-right" aria-hidden="true"></i> Read full passage
          </button>
        </div>
      </div>
    </article>`;
}

function renderResultsList(containerId, results, query, compact = false) {
  const container = $(containerId);
  if (!container) return;

  if (!results?.length) {
    container.innerHTML = `
      <div class="empty-state" style="padding:2rem">
        <i class="ti ti-mood-empty" aria-hidden="true"></i>
        <div class="empty-state-title">No results found</div>
        <div class="empty-state-sub">Try a different query or relax your filters.</div>
      </div>`;
    return;
  }

  container.innerHTML = results.map((r) => renderResultCard(r, query, compact)).join("");
  container.querySelectorAll(".read-btn").forEach((btn) => {
    btn.addEventListener("click", () => openPassageModal(btn.dataset.passageId));
  });
}

function setLoading(show) {
  const global = $("global-loading");
  if (currentTab !== "search" && global) {
    global.classList.toggle("open", show);
    global.setAttribute("aria-hidden", show ? "false" : "true");
    return;
  }
  const el = $("search-loading");
  if (el) el.classList.toggle("visible", show);
}

function exportResults(results, format, prefix) {
  if (!results?.length) {
    showToast("Run a search first.", "error");
    return;
  }
  if (format === "csv") {
    const headers = [
      "rank",
      "passage_id",
      "score",
      "confidence",
      "display_title",
      "short_title",
      "document_title",
      "language",
      "document_type",
      "date_filed",
      "source_url",
      "passage_text",
      "model",
      "query",
    ];
    const rows = results.map((r) =>
      headers.map((h) => `"${String(r[h] ?? "").replace(/"/g, '""')}"`).join(",")
    );
    downloadBlob(
      `ph_legal_search_${prefix}.csv`,
      [headers.join(","), ...rows].join("\n"),
      "text/csv"
    );
  } else {
    downloadBlob(
      `ph_legal_search_${prefix}.json`,
      JSON.stringify(results, null, 2),
      "application/json"
    );
  }
}

function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    const active = btn.dataset.tab === tab;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", active);
  });
  document.querySelectorAll(".pane[data-panel]").forEach((pane) => {
    if (pane.dataset.panel === "analytics") return;
    pane.classList.toggle("active", pane.dataset.panel === tab);
  });
}

function openAnalytics() {
  previousTab = currentTab;
  document.querySelectorAll(".pane[data-panel]").forEach((p) => {
    if (p.dataset.panel !== "analytics") p.classList.remove("active");
  });
  document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
  $("main-topbar") && ($("main-topbar").style.display = "none");
  $("page-footer") && ($("page-footer").style.display = "none");
  $("pane-analytics")?.classList.add("active");
  LegalTechAnalytics.showInlinePane();
}

function closeAnalytics() {
  $("pane-analytics")?.classList.remove("active");
  $("main-topbar") && ($("main-topbar").style.display = "");
  $("page-footer") && ($("page-footer").style.display = "");
  switchTab(previousTab);
}

function openPassageModal(passageId) {
  const p = passageCache.get(passageId);
  if (!p) return;

  const sc = scoreColor(p.score);
  $("modal-passage-id").textContent = p.passage_id;
  $("modal-doc-title").textContent =
    p.display_title ||
    p.short_title ||
    p.document_title ||
    p.document_type ||
    "";
  $("modal-text").textContent = p.passage_text || "";
  $("modal-meta").innerHTML = `
    <span class="tag-score" style="background:${sc.bg};color:${sc.tx}">${Number(p.score).toFixed(4)}</span>
    ${p.confidence ? `<span class="tag-conf ${confClass(p.confidence)}">${escapeHtml(p.confidence)}</span>` : ""}
    <span class="tag-pill">${escapeHtml(p.language)}</span>
    <span class="tag-rect">${escapeHtml(p.document_type)}</span>`;

  $("modal-overlay")?.classList.add("open");
  document.body.style.overflow = "hidden";
}

function closePassageModal(e) {
  if (e?.target && e.target !== $("modal-overlay") && !e.target.closest?.(".modal-close")) {
    return;
  }
  $("modal-overlay")?.classList.remove("open");
  document.body.style.overflow = "";
}

async function handleSearch() {
  const query = $("search-query")?.value?.trim();
  if (!query) {
    showToast("Please enter a search query.", "error");
    return;
  }

  const model = $("sidebar-model")?.value || "BGE-M3";
  const top_k = parseInt($("top-k")?.value || "5", 10);

  $("search-empty") && ($("search-empty").style.display = "none");
  $("search-meta") && ($("search-meta").style.display = "none");
  $("search-results").innerHTML = "";
  setLoading(true);

  const t0 = performance.now();
  try {
    const data = await apiPost("/api/search", { query, model, top_k });
    const ms = Math.round(performance.now() - t0);

    data.results.forEach((r) => {
      r.query = data.query;
    });

    lastExportResults = data.results;
    window._lastSearchResults = data.results;

    $("meta-query").textContent = `"${data.query}"`;
    $("meta-topk").textContent = String(data.count);
    $("search-meta").style.display = "flex";

    renderResultsList("search-results", data.results, data.query);
    if ($("search-empty")) {
      $("search-empty").style.display = data.results.length ? "none" : "flex";
    }
    if ($("search-meta")) {
      $("search-meta").style.display = data.results.length ? "flex" : "none";
    }
    showToast(`Found ${data.count} passages with ${model} (${ms} ms)`);
  } catch (e) {
    showToast(e.message, "error");
    renderResultsList("search-results", [], query);
  } finally {
    setLoading(false);
  }
}

async function handleCompare() {
  const query = $("compare-query")?.value?.trim();
  if (!query) {
    showToast("Please enter a query to compare.", "error");
    return;
  }

  const model_a = $("compare-model-a")?.value;
  const model_b = $("compare-model-b")?.value;
  const top_k = parseInt($("top-k")?.value || "5", 10);

  if (model_a === model_b) {
    showToast("Select two different models for comparison.", "error");
    return;
  }

  $("compare-empty") && ($("compare-empty").style.display = "none");
  $("compare-columns").style.display = "none";
  setLoading(true);

  try {
    const data = await apiPost("/api/compare", { query, model_a, model_b, top_k });

    data.results_a.forEach((r) => {
      r.query = data.query;
    });
    data.results_b.forEach((r) => {
      r.query = data.query;
    });

    $("col-title-a").textContent = `Model A — ${data.model_a}`;
    $("col-title-b").textContent = `Model B — ${data.model_b}`;
    $("col-sub-a").textContent = `Top ${top_k} results for "${data.query}"`;
    $("col-sub-b").textContent = `Top ${top_k} results for "${data.query}"`;

    renderResultsList("compare-results-a", data.results_a, data.query, true);
    renderResultsList("compare-results-b", data.results_b, data.query, true);

    $("compare-columns").style.display = "grid";
    $("compare-export") && ($("compare-export").style.display = "flex");
    lastExportResults = [...data.results_a, ...data.results_b];
    window._lastCompareResults = lastExportResults;
    showToast("Comparison complete");
  } catch (e) {
    showToast(e.message, "error");
  } finally {
    setLoading(false);
  }
}

async function handleFilteredSearch() {
  const query = $("filter-query")?.value?.trim();
  if (!query) {
    showToast("Please enter a query.", "error");
    return;
  }

  const model = $("sidebar-model")?.value || "BGE-M3";
  const top_k = parseInt($("top-k")?.value || "5", 10);
  const language = $("filter-lang")?.value;
  const document_type = $("filter-type")?.value;
  const date_from = $("filter-date-from")?.value || null;
  const date_to = $("filter-date-to")?.value || null;

  $("filter-empty") && ($("filter-empty").style.display = "none");
  setLoading(true);

  try {
    const data = await apiPost("/api/search", {
      query,
      model,
      top_k,
      language: language !== "All" ? language : null,
      document_type: document_type !== "All" ? document_type : null,
      date_from,
      date_to,
    });

    data.results.forEach((r) => {
      r.query = data.query;
    });

    const meta = $("filter-meta");
    if (meta) {
      meta.style.display = "block";
      meta.innerHTML = `Top <strong>${data.count}</strong> results for <strong>"${escapeHtml(data.query)}"</strong> with active filters.`;
    }

    renderResultsList("filter-results", data.results, data.query);
    if ($("filter-empty")) {
      $("filter-empty").style.display = data.results.length ? "none" : "flex";
    }
    $("filter-export") && ($("filter-export").style.display = data.results.length ? "flex" : "none");
    lastExportResults = data.results;
    window._lastFilterResults = data.results;
    showToast(`Found ${data.count} passages with filters applied`);
  } catch (e) {
    showToast(e.message, "error");
  } finally {
    setLoading(false);
  }
}

function updateCorpusStats() {
  const lang = $("filter-lang")?.value || "All";
  const typeSel = $("filter-type");
  const typeLabel =
    typeSel?.options[typeSel.selectedIndex]?.text || typeSel?.value || "All";
  if ($("f-lang-disp")) $("f-lang-disp").textContent = lang;
  if ($("f-doc-disp")) $("f-doc-disp").textContent = typeLabel;
}

function initTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });
}

async function initSearchPage() {
  initTabs();
  updateTopKSlider($("top-k")?.value || "5");

  $("top-k")?.addEventListener("input", (e) => updateTopKSlider(e.target.value));

  try {
    const [modelsRes, filtersRes] = await Promise.all([
      apiGet("/api/models"),
      apiGet("/api/filters"),
    ]);
    models = modelsRes.models;
    filters = filtersRes;

    ["sidebar-model", "compare-model-a", "compare-model-b"].forEach((id) => {
      const sel = $(id);
      if (!sel) return;
      sel.innerHTML = models
        .map(
          (m) =>
            `<option value="${escapeHtml(m.key)}"${m.key === "BGE-M3" ? " selected" : ""}>${escapeHtml(m.label)}</option>`
        )
        .join("");
      if (id === "compare-model-b" && models.length > 1) sel.value = models[1].key;
    });

    const langSel = $("filter-lang");
    const typeSel = $("filter-type");
    if (langSel) {
      langSel.innerHTML = filters.languages
        .map((l) => `<option value="${escapeHtml(l)}">${escapeHtml(l)}</option>`)
        .join("");
    }
    if (typeSel) {
      typeSel.innerHTML = filters.document_types
        .map((t) => `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`)
        .join("");
    }

    if (filters.date_from) {
      const from = $("filter-date-from");
      if (from) {
        from.min = filters.date_from;
        from.max = filters.date_to || new Date().toISOString().split("T")[0];
      }
    }
    if (filters.date_to) {
      const to = $("filter-date-to");
      if (to) {
        to.min = filters.date_from || "1900-01-01";
        to.max = filters.date_to;
      }
    }
  } catch (e) {
    showToast(`Could not load config: ${e.message}`, "error");
  }

  $("search-btn")?.addEventListener("click", handleSearch);
  $("search-query")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") handleSearch();
  });
  $("compare-btn")?.addEventListener("click", handleCompare);
  $("compare-query")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") handleCompare();
  });
  $("filter-btn")?.addEventListener("click", handleFilteredSearch);
  $("filter-query")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") handleFilteredSearch();
  });

  $("filter-lang")?.addEventListener("change", updateCorpusStats);
  $("filter-type")?.addEventListener("change", updateCorpusStats);

  $("export-csv-search")?.addEventListener("click", () =>
    exportResults(window._lastSearchResults, "csv", "search")
  );
  $("export-json-search")?.addEventListener("click", () =>
    exportResults(window._lastSearchResults, "json", "search")
  );
  $("export-csv-compare")?.addEventListener("click", () =>
    exportResults(window._lastCompareResults, "csv", "compare")
  );
  $("export-json-compare")?.addEventListener("click", () =>
    exportResults(window._lastCompareResults, "json", "compare")
  );
  $("export-csv-filter")?.addEventListener("click", () =>
    exportResults(window._lastFilterResults, "csv", "filtered")
  );
  $("export-json-filter")?.addEventListener("click", () =>
    exportResults(window._lastFilterResults, "json", "filtered")
  );
  $("btn-export-sidebar")?.addEventListener("click", () => {
    const data =
      window._lastSearchResults ||
      window._lastFilterResults ||
      window._lastCompareResults ||
      lastExportResults;
    exportResults(data, "json", "export");
  });

  $("btn-analytics")?.addEventListener("click", openAnalytics);
  $("btn-close-analytics")?.addEventListener("click", closeAnalytics);

  $("modal-close")?.addEventListener("click", () => closePassageModal());
  $("modal-overlay")?.addEventListener("click", (e) => closePassageModal(e));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closePassageModal();
  });

  updateCorpusStats();

  if (window.location.hash === "#analytics") {
    openAnalytics();
  }
}

document.addEventListener("DOMContentLoaded", initSearchPage);
