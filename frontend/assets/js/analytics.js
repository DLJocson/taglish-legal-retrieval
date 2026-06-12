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

  function renderBarChart(containerId, rows, modelCol, metricCol, categoryCol = null) {
    const el = $(containerId);
    if (!el || !metricCol || !rows?.length) {
      if (el) el.innerHTML = "<p style='font-size:11px;color:var(--an-muted)'>No data</p>";
      return;
    }

    // If categoryCol is provided, group by category first
    if (categoryCol) {
      const categories = [...new Set(rows.map(r => r[categoryCol]))];
      el.innerHTML = categories.map((cat, index) => {
        const catRows = rows.filter(r => r[categoryCol] === cat);
        // Use category-specific max for proper scaling
        const catMax = Math.max(...catRows.map((r) => Number(r[metricCol] ?? 0)), 0.0001);
        const divider = index < categories.length - 1 ? '<hr style="border:0;border-top:1px solid var(--an-bdr);margin:0.75rem 0 0.5rem;">' : '';
        return `
          <div style="margin-bottom: 0.75rem;">
            <div style="font-size: 11px; color: var(--an-muted); margin-bottom: 0.25rem;">${escapeHtml(cat.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()))}</div>
            ${catRows.map((r, i) => {
              const v = Number(r[metricCol] ?? 0);
              const m = String(r[modelCol]);
              const c = colorFor(m, i);
              const isPerfectMRR = metricCol && metricCol.toLowerCase().includes('mrr') && v >= 0.9999;
              const cardClass = isPerfectMRR ? 'perfect-mrr' : '';
              const widthPercent = v === 0 ? 3 : ((v / catMax) * 100).toFixed(1);
              return `
                <div class="chart-row ${cardClass}">
                  <span class="chart-model-label" title="${escapeHtml(m)}">${escapeHtml(truncate(m, 10))}</span>
                  <div class="chart-bar-track">
                    <div class="chart-bar-fill" style="--bar-width:${widthPercent}%;width:${widthPercent}%;background:${c};min-width:2px;animation-delay:${i * 50}ms;"></div>
                  </div>
                  <span class="chart-bar-val">${v.toFixed(4)}</span>
                </div>
              `;
            }).join('')}
            ${divider}
          </div>
        `;
      }).join('');
      return;
    }

    const max = Math.max(...rows.map((r) => Number(r[metricCol] ?? 0)), 0.0001);

    // Defensive handling for single-item categories
    const isSingleItem = rows.length === 1;

    el.innerHTML = rows
      .map((r, i) => {
        const v = Number(r[metricCol] ?? 0);
        const m = String(r[modelCol]);
        const c = colorFor(m, i);

        // Check for perfect MRR (1.0000) for highlighting
        const isPerfectMRR = metricCol && metricCol.toLowerCase().includes('mrr') && v >= 0.9999;
        const cardClass = isPerfectMRR ? 'perfect-mrr' : '';
        const widthPercent = v === 0 ? 3 : ((v / max) * 100).toFixed(1);

        return `
          <div class="chart-row ${cardClass}">
            <span class="chart-model-label" title="${escapeHtml(m)}">${escapeHtml(truncate(m, 10))}</span>
            ${isSingleItem ? `<span class="chart-single-label">(1 query)</span>` : ''}
            <div class="chart-bar-track">
              <div class="chart-bar-fill" style="--bar-width:${widthPercent}%;width:${widthPercent}%;background:${c};min-width:2px;animation-delay:${i * 50}ms;"></div>
            </div>
            <span class="chart-bar-val">${v.toFixed(4)}</span>
          </div>`;
      })
      .join("");
  }

  function animateCountUp(element, startValue, endValue, duration = 800, deltaHtml = '') {
    const startTime = performance.now();
    const start = startValue;
    const end = endValue;
    const range = end - start;

    function update(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const easeProgress = 1 - Math.pow(1 - progress, 3);
      const current = start + (range * easeProgress);
      element.innerHTML = current.toFixed(4) + deltaHtml;

      if (progress < 1) {
        requestAnimationFrame(update);
      } else {
        // Store final value for next animation
        element.setAttribute('data-prev', end.toFixed(4));
      }
    }

    requestAnimationFrame(update);
  }

  function renderInsights(baselineGlobal, alignedGlobal, deltas, cols, viewMode = 'aligned') {
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

    const selectedGlobal = viewMode === 'baseline' ? baselineGlobal : alignedGlobal;
    const winners = [];
    const tileData = [];
    const tiles = metrics.map((m) => {
      const b = bestModel(selectedGlobal, modelCol, m.key);
      winners.push(b.model);

      // Find delta for this model and metric
      const deltaRow = deltas.find((d) => d[modelCol] === b.model);
      const deltaKey = `${m.key}_delta`;
      const deltaValue = deltaRow ? deltaRow[deltaKey] : 0;

      // Hide delta on Baseline view since it represents LINAW's effect on Aligned view
      let deltaHtml = "";
      if (viewMode === 'aligned') {
        const deltaSign = deltaValue >= 0 ? "+" : "";
        const deltaClass = deltaValue >= 0 ? "delta-positive" : "delta-negative";
        deltaHtml = ` <span class="${deltaClass}">${deltaSign}${deltaValue.toFixed(4)}</span>`;
      }

      tileData.push({ value: b.value, deltaHtml });
      return `
        <div class="metric-tile">
          <div class="metric-tile-label">${m.label}</div>
          <div class="metric-tile-value" data-prev="0">${b.value.toFixed(4)}${deltaHtml}</div>
          <div class="metric-tile-model">${escapeHtml(b.model)}</div>
        </div>`;
    });

    grid.innerHTML = tiles.join("");

    // Animate metric values
    const valueElements = grid.querySelectorAll('.metric-tile-value');
    valueElements.forEach((el, index) => {
      const prevValue = parseFloat(el.getAttribute('data-prev')) || 0;
      const endValue = tileData[index].value;
      const deltaHtml = tileData[index].deltaHtml;
      animateCountUp(el, prevValue, endValue, 800, deltaHtml);
    });

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
              let text;
              if (typeof v === "number") {
                text = v.toFixed(4);
              } else if (Array.isArray(v)) {
                // Show first 2 IDs with "+N more" label
                const display = v.slice(0, 2).map(item => String(item ?? ""));
                const remaining = v.length - 2;
                text = escapeHtml(display.join(", ") + (remaining > 0 ? ` (+${remaining} more)` : ""));
              } else {
                text = escapeHtml(String(v ?? ""));
              }
              return `<td>${text}</td>`;
            })
            .join("")}</tr>`
      )
      .join("");

    el.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  }

  // Store data globally for view switching
  let currentData = null;
  let currentView = 'aligned';

  function renderDashboard(data) {
    currentData = data;
    renderDashboardWithView(currentView);
  }

  async function renderDashboardWithView(viewMode) {
    if (!currentData) return;

    // Add fade+slide animation
    const content = $("analytics-content");
    if (content) {
      content.classList.remove("view-transition");
      // Trigger reflow to restart animation
      void content.offsetWidth;
      content.classList.add("view-transition");
    }

    // Small delay for visual feedback
    await new Promise(resolve => setTimeout(resolve, 150));

    currentView = viewMode;

    const cols = currentData.columns || { model: "model", mrr: "mrr", p5: "p@5", p10: "p@10", recall: "recall@10" };
    const baselineGlobal = currentData.baseline_global || currentData.baseline;
    const alignedGlobal = currentData.aligned_global || currentData.aligned;
    const deltas = currentData.deltas;
    const baselineLang = currentData.baseline_language || [];
    const alignedLang = currentData.aligned_language || [];
    const baselineEnglish = currentData.baseline_english || [];
    const alignedEnglish = currentData.aligned_english || [];
    const baselineTagalog = currentData.baseline_tagalog || [];
    const alignedTagalog = currentData.aligned_tagalog || [];
    const baselineSemantic = currentData.baseline_semantic || [];
    const alignedSemantic = currentData.aligned_semantic || [];
    const sampleData = currentData.sample_data || [];

    // Select dataset based on view mode
    const selectedGlobal = viewMode === 'baseline' ? baselineGlobal : alignedGlobal;
    const selectedLang = viewMode === 'baseline' ? baselineLang : alignedLang;
    const selectedEnglish = viewMode === 'baseline' ? baselineEnglish : alignedEnglish;
    const selectedTagalog = viewMode === 'baseline' ? baselineTagalog : alignedTagalog;
    const selectedSemantic = viewMode === 'baseline' ? baselineSemantic : alignedSemantic;

    renderInsights(baselineGlobal, alignedGlobal, deltas, cols, viewMode);
    renderBarChart("chart-mrr", selectedGlobal, cols.model, cols.mrr);
    renderBarChart("chart-p10", selectedGlobal, cols.model, cols.p10);
    renderBarChart("chart-p5", selectedGlobal, cols.model, cols.p5);
    renderBarChart("chart-recall", selectedGlobal, cols.model, cols.recall);

    // Render Code-Switched performance charts
    renderBarChart("chart-codeswitched-mrr", selectedLang, cols.model, cols.mrr);
    renderBarChart("chart-codeswitched-p10", selectedLang, cols.model, cols.p10);

    // Render English performance charts
    renderBarChart("chart-english-mrr", selectedEnglish, cols.model, cols.mrr);
    renderBarChart("chart-english-p10", selectedEnglish, cols.model, cols.p10);

    // Render Tagalog performance charts
    renderBarChart("chart-tagalog-mrr", selectedTagalog, cols.model, cols.mrr);
    renderBarChart("chart-tagalog-p10", selectedTagalog, cols.model, cols.p10);

    // Render Semantic Type performance charts
    renderBarChart("chart-semantic-mrr", selectedSemantic, cols.model, cols.mrr, cols.category);
    renderBarChart("chart-semantic-p10", selectedSemantic, cols.model, cols.p10, cols.category);

    // Render per-query sample
    renderRawTable("table-raw", sampleData);

    // Fade back in
    if (content) {
      content.style.opacity = "1";
    }
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
      console.log("Analytics data received:", data);
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

  function initScrollObserver() {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
        }
      });
    }, {
      threshold: 0.1,
      rootMargin: '0px 0px -50px 0px'
    });

    const sections = document.querySelectorAll('.an-section');
    sections.forEach(section => observer.observe(section));
  }

  async function initStandalonePage() {
    if (!$("analytics-standalone")) return;
    await showInlinePane();

    // Initialize scroll observer after content is loaded
    initScrollObserver();

    // Add event listeners for view toggle
    const toggleInputs = document.querySelectorAll('input[name="view-mode"]');
    toggleInputs.forEach(input => {
      input.addEventListener('change', (e) => {
        if (e.target.checked) {
          renderDashboardWithView(e.target.value);
        }
      });
    });
  }

  return { renderDashboard, loadMetrics, showInlinePane, initStandalonePage };
})();

document.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("analytics-standalone")) {
    LegalTechAnalytics.initStandalonePage();
  }
});
