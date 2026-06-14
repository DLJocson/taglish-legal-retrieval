/**
 * Search page: single-model search, compare, filtered search, analytics pane, modal.
 */
const { $, escapeHtml, truncate, showToast, apiGet, apiPost, downloadBlob, scoreColor, updateTopKSlider } =
  LegalTech;

let models = [];
let filters = { languages: ["All"], document_types: ["All"], date_from: null, date_to: null };
let passageCache = new Map();
let currentTab = "search";
let lastExportResults = [];
let linawEnabled = false;
let linawEnabledA = false;
let linawEnabledB = false;

// Common English stop words to filter out during keyword extraction
// Removing these improves highlighting relevance by focusing on content words
const STOP_WORDS = new Set([
  "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
  "by", "from", "is", "are", "was", "were", "be", "been", "have", "has", "do", "does",
  "did", "will", "would", "could", "should", "may", "might", "must", "can", "this",
  "that", "these", "those", "what", "which", "who", "when", "where", "why", "how",
]);

// Extract meaningful keywords from query for result highlighting
// Filters out stop words and short tokens (< 3 chars) to focus on content
function extractKeywords(query) {
  const words = query.toLowerCase().match(/\b\w+\b/g) || [];
  return words.filter((w) => w.length > 2 && !STOP_WORDS.has(w));
}

// Highlight query keywords in passage text using case-insensitive regex
// Escapes special regex characters in keywords to prevent pattern errors
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

function renderDemoMetricsBanner(demoMetrics) {
  if (!demoMetrics) return "";

  const metrics = [
    { key: "mrr", label: "MRR", fullLabel: "Mean Reciprocal Rank", definition: "Measures rank of first relevant result (0–1, higher is better)" },
    { key: "p_at_5", label: "P@5", fullLabel: "Precision@5", definition: "Proportion of relevant results in top 5 (0–1, higher is better)" },
    { key: "p_at_10", label: "P@10", fullLabel: "Precision@10", definition: "Proportion of relevant results in top 10 (0–1, higher is better)" },
    { key: "recall_at_10", label: "Recall@10", fullLabel: "Recall@10", definition: "Proportion of all relevant results found in top 10 (0–1, higher is better)" },
  ];

  const metricCards = metrics.map((metric) => {
    const data = demoMetrics[metric.key];
    if (!data) return "";

    const baseline = data.baseline.toFixed(2);
    const aligned = data.aligned.toFixed(2);
    const improvement = data.improvement.toFixed(2);
    
    let improvementClass = "metric-improvement-neutral";
    let arrowIcon = "−";
    let improvementSign = "";
    
    if (data.improvement > 0) {
      improvementClass = "metric-improvement-positive";
      arrowIcon = "↑";
      improvementSign = "+";
    } else if (data.improvement < 0) {
      improvementClass = "metric-improvement-negative";
      arrowIcon = "↓";
    }

    const ariaLabel = `${metric.fullLabel}: Baseline ${baseline}, Aligned ${aligned}, ${data.improvement > 0 ? 'improvement' : data.improvement < 0 ? 'decrease' : 'no change'} of ${improvementSign}${improvement}`;

    return `
      <div class="metric-card" aria-label="${escapeHtml(ariaLabel)}">
        <div class="metric-name-wrapper">
          <div class="metric-name">
            ${escapeHtml(metric.label)} 
            <span class="metric-tooltip-wrapper">
              <i class="ti ti-info-circle metric-info-icon" aria-hidden="true"></i>
              <span class="metric-tooltip">${escapeHtml(metric.definition)}</span>
            </span>
          </div>
        </div>
        <div class="metric-transition">
          <span class="metric-baseline">${baseline}</span>
          <span class="metric-arrow">→</span>
          <span class="metric-aligned">${aligned}</span>
        </div>
        <div class="metric-improvement ${improvementClass}">
          <i class="ti ti-arrow-${arrowIcon === '↑' ? 'up' : arrowIcon === '↓' ? 'down' : 'right'} metric-arrow-icon" aria-hidden="true"></i>
          ${improvementSign}${improvement}
        </div>
      </div>
    `;
  }).join("");

  return `
    <div class="demo-metrics-banner">
      <div class="demo-metrics-header">
        <span class="demo-metrics-title">Evaluation Metrics</span>
        <span class="demo-metrics-scale">Scale: 0–1</span>
      </div>
      <div class="demo-metrics-grid">${metricCards}</div>
    </div>
  `;
}

function renderResultsList(containerId, results, query, compact = false, bannerHtml = "") {
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

  const resultsHtml = results.map((r) => renderResultCard(r, query, compact)).join("");
  container.innerHTML = bannerHtml + resultsHtml;
  // Staggered animation: each card appears 60ms after the previous for smooth cascade effect
  container.querySelectorAll(".result-card").forEach((card, index) => {
    card.style.animationDelay = `${index * 0.06}s`;
  });
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
    if (pane.dataset.panel === tab) {
      pane.classList.remove("pane-entering");
      void pane.offsetWidth; // Force reflow to restart CSS animation on tab switch
      pane.classList.add("pane-entering");
    }
  });
  updateTabUnderline();
}

function updateTabUnderline() {
  const activeTab = document.querySelector(".tab-btn.active");
  const underline = document.querySelector(".tab-underline");
  if (!activeTab || !underline) return;
  underline.style.left = activeTab.offsetLeft + "px";
  underline.style.width = activeTab.offsetWidth + "px";
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
  const is_aligned = linawEnabled;

  $("search-empty") && ($("search-empty").style.display = "none");
  $("search-meta") && ($("search-meta").style.display = "none");
  $("search-results").innerHTML = "";
  setLoading(true);

  const t0 = performance.now();
  try {
    const data = await apiPost("/api/search", { query, model, top_k, is_aligned });
    const ms = Math.round(performance.now() - t0);

    data.results.forEach((r) => {
      r.query = data.query;
    });

    lastExportResults = data.results;
    window._lastSearchResults = data.results;

    $("meta-query").textContent = `"${data.query}"`;
    $("meta-topk").textContent = String(data.count);
    $("search-meta").style.display = "flex";

    // Display detected language
    const langDisplay = $("lang-display-row");
    const langTag = $("detected-lang");
    if (langDisplay && langTag && data.detected_language) {
      langTag.textContent = data.detected_language;
      langTag.className = "lang-tag";
      if (data.detected_language === "English") langTag.classList.add("lang-tag-english");
      else if (data.detected_language === "Tagalog") langTag.classList.add("lang-tag-tagalog");
      else if (data.detected_language === "Code-Switched") langTag.classList.add("lang-tag-codeswitched");
      else langTag.classList.add("lang-tag-other");
      langDisplay.style.display = "flex";
    }

    // Render demo metrics banner only if adapter is enabled and metrics are available
    const metricsBanner = linawEnabled ? renderDemoMetricsBanner(data.demo_metrics) : "";

    renderResultsList("search-results", data.results, data.query, false, metricsBanner);
    if ($("search-empty")) {
      $("search-empty").style.display = data.results.length ? "none" : "flex";
    }
    if ($("search-meta")) {
      $("search-meta").style.display = data.results.length ? "flex" : "none";
    }
    const adapterStatus = is_aligned ? " (Neural Alignment Adapter Active)" : "";
    showToast(`Found ${data.count} passages with ${model}${adapterStatus} (${ms} ms)`);
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
  const is_aligned_a = linawEnabledA;
  const is_aligned_b = linawEnabledB;
  const top_k = parseInt($("top-k")?.value || "5", 10);

  // Prevent meaningless comparisons: identical model + adapter state yields identical results
  if (model_a === model_b && is_aligned_a === is_aligned_b) {
    showToast("Select different models or different adapter states for comparison.", "error");
    return;
  }

  $("compare-empty") && ($("compare-empty").style.display = "none");
  $("compare-columns").style.display = "none";
  setLoading(true);

  try {
    const data = await apiPost("/api/compare", { query, model_a, model_b, is_aligned_a, is_aligned_b, top_k });

    data.results_a.forEach((r) => {
      r.query = data.query;
    });
    data.results_b.forEach((r) => {
      r.query = data.query;
    });

    const adapterLabelA = is_aligned_a ? " (Neural Alignment Adapter Active)" : " (Zero-Shot)";
    const adapterLabelB = is_aligned_b ? " (Neural Alignment Adapter Active)" : " (Zero-Shot)";
    $("col-title-a").textContent = `Model A — ${data.model_a}${adapterLabelA}`;
    $("col-title-b").textContent = `Model B — ${data.model_b}${adapterLabelB}`;
    $("col-sub-a").textContent = `Top ${top_k} results for "${data.query}"`;
    $("col-sub-b").textContent = `Top ${top_k} results for "${data.query}"`;

    renderResultsList("compare-results-a", data.results_a, data.query, true);
    renderResultsList("compare-results-b", data.results_b, data.query, true);

    // Apply conditional formatting for improved rankings
    applyCompareHighlighting(data.results_a, data.results_b, is_aligned_a, is_aligned_b);

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

// Highlight passages that improved ranking when comparing aligned vs baseline
// Only applies when comparing aligned model against baseline (same model comparison is invalid)
// Improved ranking = passage appears earlier (lower rank) in aligned results
function applyCompareHighlighting(resultsA, resultsB, isAlignedA, isAlignedB) {
  // Only highlight if one column is aligned and the other is not
  if (isAlignedA === isAlignedB) return;

  const alignedResults = isAlignedA ? resultsA : resultsB;
  const baselineResults = isAlignedA ? resultsB : resultsA;
  const alignedContainerId = isAlignedA ? "compare-results-a" : "compare-results-b";

  // Create a map of passage_id to rank in baseline for O(1) lookup
  const baselineRankMap = new Map();
  baselineResults.forEach((r) => {
    baselineRankMap.set(r.passage_id, r.rank);
  });

  // Highlight passages that improved in the aligned column
  const container = $(alignedContainerId);
  if (!container) return;

  container.querySelectorAll(".result-card").forEach((card) => {
    const passageId = card.querySelector(".tag-id")?.textContent;
    if (!passageId) return;

    const baselineRank = baselineRankMap.get(passageId);
    const currentRank = parseInt(card.querySelector(".rank-badge")?.textContent || "0", 10);

    // If passage exists in both and rank improved in aligned column
    if (baselineRank && currentRank < baselineRank) {
      card.classList.add("rank-improved");
    }
  });
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
  updateTabUnderline();
  updateTopKSlider($("top-k")?.value || "5");

  $("top-k")?.addEventListener("input", (e) => updateTopKSlider(e.target.value));

  // LINAW toggle event listeners
  $("linaw-toggle")?.addEventListener("change", (e) => {
    linawEnabled = e.target.checked;
    const badge = $("linaw-badge");
    if (badge) {
      badge.style.display = linawEnabled ? "flex" : "none";
    }
  });

  $("compare-aligned-a")?.addEventListener("change", (e) => {
    linawEnabledA = e.target.checked;
  });

  $("compare-aligned-b")?.addEventListener("change", (e) => {
    linawEnabledB = e.target.checked;
  });

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

  $("btn-analytics")?.addEventListener("click", () => {
    window.location.href = "/analytics";
  });

  $("modal-close")?.addEventListener("click", () => closePassageModal());
  $("modal-overlay")?.addEventListener("click", (e) => closePassageModal(e));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closePassageModal();
  });

  updateCorpusStats();
}

document.addEventListener("DOMContentLoaded", initSearchPage);
