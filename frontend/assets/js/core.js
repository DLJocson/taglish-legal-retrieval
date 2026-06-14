/** Shared API helpers and UI utilities. */
const LegalTech = {
  API_BASE: window.location.origin,

  // Shorthand for document.getElementById to reduce verbosity
  $(id) {
    return document.getElementById(id);
  },

  // Sanitize user input to prevent XSS attacks
  // Uses browser's built-in HTML escaping via textContent
  escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text ?? "";
    return div.innerHTML;
  },

  // Truncate strings with ellipsis for display in UI cards/tables
  truncate(str, len) {
    if (!str || str.length <= len) return str ?? "";
    return str.slice(0, len) + "…";
  },

  // Display transient toast notifications for user feedback
  // Auto-dismisses after 3.5 seconds
  showToast(message, type = "success") {
    const toast = document.getElementById("toast");
    if (!toast) return;
    toast.textContent = message;
    toast.className =
      type === "error" ? "toast-error" : "toast-success";
    toast.classList.remove("hidden");
    setTimeout(() => toast.classList.add("hidden"), 3500);
  },

  // Wrapper for GET requests with automatic error handling
  // Extracts FastAPI error detail from JSON response when available
  async apiGet(path) {
    const res = await fetch(`${LegalTech.API_BASE}${path}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(
        typeof err.detail === "string" ? err.detail : res.statusText
      );
    }
    return res.json();
  },

  // Wrapper for POST requests with automatic error handling
  // Sends JSON body and extracts FastAPI error detail when available
  async apiPost(path, body) {
    const res = await fetch(`${LegalTech.API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(
        typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail)
      );
    }
    return res.json();
  },

  // Client-side file download using Blob API
  // Creates temporary object URL and triggers browser download
  downloadBlob(filename, content, mime) {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);  // Clean up memory
  },

  // Return color scheme based on cosine similarity score (0-1 range)
  // Thresholds match backend confidence levels: 0.65+ (MEDIUM), 0.75+ (HIGH)
  scoreColor(s) {
    if (s >= 0.75) return { bar: "#1D9E75", bg: "#E1F5EE", tx: "#0F6E56" };
    if (s >= 0.68) return { bar: "#639922", bg: "#EAF3DE", tx: "#3B6D11" };
    if (s >= 0.65) return { bar: "#BA7517", bg: "#FAEEDA", tx: "#854F0B" };
    return { bar: "#888780", bg: "#F1EFE8", tx: "#5F5E5A" };
  },

  // Update range slider visual fill and label to reflect current value
  // Creates gradient fill effect proportional to slider position (1-10 range)
  updateTopKSlider(val) {
    const slider = LegalTech.$("top-k");
    const label = LegalTech.$("top-k-label");
    if (label) label.textContent = `Top ${val}`;
    if (slider) {
      const pct = ((parseInt(val, 10) - 1) / 9) * 100;
      slider.style.background = `linear-gradient(to right, var(--sb-gold) ${pct}%, var(--sb-border) ${pct}%)`;
    }
  },
};
