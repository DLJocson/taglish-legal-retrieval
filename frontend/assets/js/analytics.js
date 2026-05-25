/**
 * Analytics: evaluation metrics (CSS charts + tables).
 */
const LegalTechAnalytics = (() => {
  const { $, escapeHtml, truncate, apiGet } = LegalTech;

  const MODEL_COLORS = {
    BGE_M3: "#1D9E75",
    "BGE-M3": "#1D9E75",
    MSBERT: "#7F77DD",
    mSBERT: "#7F77DD",
    LEGAL_BERT: "#378ADD",
    "Legal-BERT": "#378ADD",
  };

  const FALLBACK = ["#1D9E75", "#7F77DD", "#378ADD", "#C9973A"];

  function colorFor(model, i) {
    const key = String(model).replace(/\s/g, "_");
    return MODEL_COLORS[model] || MODEL_COLORS[key] || FALLBACK[i % FALLBACK.length];
  }

  function bestModel(rows, modelCol, metricCol) {
    if (!metricCol || !rows?.length) return { model: "N/A", value: 0 };
    let best = rows[0];
    for (const r of rows) {
      if ((r[metricCol] ?? 0) > (best[metricCol] ?? 0)) best = r;
    }
    return { model: String(best[modelCol]), value: Number(best[metricCol] ?? 0) };
  }

  function renderBarChart(containerId, rows, modelCol, metricCol) {
    const el = $(containerId);
    if (!el || !metricCol || !rows?.length) {
      if (el) el.innerHTML = "<p style='font-size:11px;color:var(--an-muted)'>No data</p>";
      return;
    }

    const max = Math.max(...rows.map((r) => Number(r[metricCol] ?? 0)), 0.0001);
    el.innerHTML = rows
      .map((r, i) => {
        const v = Number(r[metricCol] ?? 0);
        const m = String(r[modelCol]);
        const c = colorFor(m, i);
        return `
          <div class="chart-row">
            <span class="chart-model-label" title="${escapeHtml(m)}">${escapeHtml(truncate(m, 10))}</span>
            <div class="chart-bar-track">
              <div class="chart-bar-fill" style="width:${((v / max) * 100).toFixed(1)}%;background:${c}"></div>
            </div>
            <span class="chart-bar-val">${v.toFixed(4)}</span>
          </div>`;
      })
      .join("");
  }

  function renderInsights(global, cols) {
    const grid = $("insight-grid");
    const overallTile = $("overall-leader");
    if (!grid) return;

    const modelCol = cols.model;
    const metrics = [
      { key: cols.mrr, label: "Best MRR" },
      { key: cols.p10, label: "Best P@10" },
      { key: cols.p5, label: "Best P@5" },
      { key: cols.recall, label: "Best Recall" },
    ].filter((m) => m.key);

    const winners = [];
    const tiles = metrics.map((m) => {
      const b = bestModel(global, modelCol, m.key);
      winners.push(b.model);
      return `
        <div class="metric-tile">
          <div class="metric-tile-label">${m.label}</div>
          <div class="metric-tile-value">${b.value.toFixed(4)}</div>
          <div class="metric-tile-model">${escapeHtml(b.model)}</div>
        </div>`;
    });

    grid.innerHTML = tiles.join("");

    const votes = {};
    winners.forEach((m) => {
      if (m !== "N/A") votes[m] = (votes[m] || 0) + 1;
    });
    const overall = Object.entries(votes).sort((a, b) => b[1] - a[1])[0];

    if (overallTile && overall) {
      overallTile.style.display = "";
      if ($("overall-leader-name")) $("overall-leader-name").textContent = overall[0];
      if ($("overall-leader-sub"))
        $("overall-leader-sub").textContent = `Best in ${overall[1]}/${metrics.length} metrics`;
    }
  }

  function renderRawTable(containerId, rows) {
    const el = $(containerId);
    if (!el) return;
    if (!rows?.length) {
      el.innerHTML = "<p style='padding:1rem;font-size:12px;color:var(--an-muted)'>No sample data</p>";
      return;
    }

    const cols = Object.keys(rows[0]);
    const head = cols.map((c) => `<th>${escapeHtml(c)}</th>`).join("");
    const body = rows
      .map(
        (r) =>
          `<tr>${cols
            .map((c) => {
              const v = r[c];
              const text =
                typeof v === "number" ? v.toFixed(4) : escapeHtml(String(v ?? ""));
              return `<td>${text}</td>`;
            })
            .join("")}</tr>`
      )
      .join("");

    el.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  }

  function renderDashboard(data) {
    const cols = data.columns;
    const global = data.global;

    renderInsights(global, cols);
    renderBarChart("chart-mrr", global, cols.model, cols.mrr);
    renderBarChart("chart-p10", global, cols.model, cols.p10);
    renderBarChart("chart-p5", global, cols.model, cols.p5);
    renderBarChart("chart-recall", global, cols.model, cols.recall);
    renderRawTable("table-raw", data.raw_sample || []);
  }

  async function loadMetrics() {
    return apiGet("/api/analytics/metrics");
  }

  async function showInlinePane() {
    const loading = $("analytics-loading");
    const content = $("analytics-content");
    const errEl = $("analytics-error");

    loading?.classList.add("visible");
    content && (content.style.display = "none");
    errEl && (errEl.style.display = "none");

    try {
      const data = await loadMetrics();
      renderDashboard(data);
      loading?.classList.remove("visible");
      if (content) content.style.display = "";
    } catch (e) {
      loading?.classList.remove("visible");
      if (errEl) {
        errEl.style.display = "";
        errEl.innerHTML = `<strong>Evaluation data not available.</strong><p style="margin-top:.5rem">${escapeHtml(e.message)}</p><p style="margin-top:.5rem;font-size:11px;color:var(--an-muted)">Run <code>06_evaluate_retrieval.py</code> to generate metrics.</p>`;
      }
    }
  }

  async function initStandalonePage() {
    if (!$("analytics-standalone")) return;
    await showInlinePane();
  }

  return { renderDashboard, loadMetrics, showInlinePane, initStandalonePage };
})();

document.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("analytics-standalone")) {
    LegalTechAnalytics.initStandalonePage();
  }
});
