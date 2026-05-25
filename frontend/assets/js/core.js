/** Shared API helpers and UI utilities. */
const LegalTech = {
  API_BASE: window.location.origin,

  $(id) {
    return document.getElementById(id);
  },

  escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text ?? "";
    return div.innerHTML;
  },

  truncate(str, len) {
    if (!str || str.length <= len) return str ?? "";
    return str.slice(0, len) + "…";
  },

  showToast(message, type = "success") {
    const toast = document.getElementById("toast");
    if (!toast) return;
    toast.textContent = message;
    toast.className =
      type === "error" ? "toast-error" : "toast-success";
    toast.classList.remove("hidden");
    setTimeout(() => toast.classList.add("hidden"), 3500);
  },

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

  downloadBlob(filename, content, mime) {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  },

  scoreColor(s) {
    if (s >= 0.75) return { bar: "#1D9E75", bg: "#E1F5EE", tx: "#0F6E56" };
    if (s >= 0.68) return { bar: "#639922", bg: "#EAF3DE", tx: "#3B6D11" };
    if (s >= 0.65) return { bar: "#BA7517", bg: "#FAEEDA", tx: "#854F0B" };
    return { bar: "#888780", bg: "#F1EFE8", tx: "#5F5E5A" };
  },

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
