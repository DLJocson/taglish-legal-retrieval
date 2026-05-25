"""Resolve human-readable citation titles for search result cards."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

GENERIC_LABELS = frozenset(
    {
        "decisions / signed resolutions",
        "republic acts",
        "acts",
        "commonwealth act",
        "unknown",
        "unknown document",
    }
)

# Statute identifiers and optional descriptive line
_STATUTE_ID = re.compile(
    r"(?:\[\s*)?"
    r"(?P<kind>REPUBLIC\s+ACT|COMMONWEALTH\s+ACT|ACT)\s+NO\.?\s*(?P<num>[\d\-]+)"
    r"[^\]]*(?:\])?",
    re.IGNORECASE,
)
_AN_ACT = re.compile(
    r"(AN\s+ACT(?:\s+TO|\s+ESTABLISHING|\s+AUTHORIZING|\s+PROVIDING)[^.]{8,140})",
    re.IGNORECASE,
)
_GR_NO = re.compile(r"G\.?\s*R\.?\s*No\.?\s*(?P<num>[\d\-]+)", re.IGNORECASE)
_PARTIES_AFTER_GR = re.compile(
    r"\]\s*(?P<parties>.{12,200}?(?:PETITIONER|PLAINTIFF|APPELLANT|COMPLAINANT))"
    r".{0,80}?(?:RESPONDENT|DEFENDANT|APPELLEE)",
    re.IGNORECASE | re.DOTALL,
)
_PARTIES_VS = re.compile(
    r"([A-Z][A-Z0-9\s,'.&\-]{4,80})\s*,?\s*PETITIONER\s*,?\s*VS\.?\s*"
    r"([A-Z][A-Z0-9\s,'.&\-]{4,80})\s*,?\s*RESPONDENT",
    re.IGNORECASE,
)


def _truncate(text: str, max_len: int = 120) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def is_generic_title(value: Any, document_type: str | None = None) -> bool:
    if value is None:
        return True
    s = str(value).strip()
    if not s or s.lower() in ("none", "nan", "—", "-"):
        return True
    if document_type and s.lower() == str(document_type).strip().lower():
        return True
    return s.lower() in GENERIC_LABELS


def _title_case_act(phrase: str) -> str:
    phrase = re.sub(r"\s+", " ", phrase.strip())
    if phrase.isupper():
        return phrase.title()
    return phrase


def _clean_parties(raw: str) -> str:
    s = re.sub(r"\s+", " ", raw.strip())
    s = re.sub(r",?\s*PETITIONER.*", "", s, flags=re.IGNORECASE)
    s = re.sub(r",?\s*RESPONDENT.*", "", s, flags=re.IGNORECASE)
    s = re.sub(r",?\s*PLAINTIFF.*", "", s, flags=re.IGNORECASE)
    s = re.sub(r",?\s*DEFENDANT.*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*VS\.?\s*", " v. ", s, flags=re.IGNORECASE)
    return _truncate(s, 100)


def extract_title_from_passage(passage_text: str, document_type: str | None = None) -> str | None:
    """Heuristic titles from passage text (G.R. numbers, RA numbers, party names)."""
    if not passage_text or not isinstance(passage_text, str):
        return None

    head = passage_text[:2500]

    gr = _GR_NO.search(head)
    if gr:
        gr_label = f"G.R. No. {gr.group('num')}"
        m_vs = _PARTIES_VS.search(head)
        if m_vs:
            left = m_vs.group(1).strip().title()
            right = m_vs.group(2).strip().title()
            return _truncate(f"{gr_label} — {left} v. {right}", 120)
        m_parties = _PARTIES_AFTER_GR.search(head)
        if m_parties:
            return _truncate(f"{gr_label} — {_clean_parties(m_parties.group('parties'))}", 120)
        return gr_label

    statute = _STATUTE_ID.search(head)
    if statute:
        kind = statute.group("kind").upper()
        num = statute.group("num")
        if "REPUBLIC" in kind:
            short = f"Republic Act No. {num}"
        elif "COMMONWEALTH" in kind:
            short = f"Commonwealth Act No. {num}"
        else:
            short = f"Act No. {num}"
        act_line = _AN_ACT.search(head)
        if act_line:
            desc = _title_case_act(act_line.group(1))
            return _truncate(f"{short} — {desc}", 120)
        return short

    act_only = _AN_ACT.search(head)
    if act_only and document_type and "act" in document_type.lower():
        return _truncate(_title_case_act(act_only.group(1)), 120)

    return None


_STATUTE_SHORT = re.compile(
    r"^(Republic Act|Commonwealth Act|Act)\s+No\.?\s*[\d\-]+$",
    re.IGNORECASE,
)


def resolve_display_title(
    *,
    short_title: str | None = None,
    document_title: str | None = None,
    passage_text: str | None = None,
    document_type: str | None = None,
) -> str:
    """Pick the best card header: citation short title > extracted > long title > type label."""
    if short_title and not is_generic_title(short_title, document_type):
        st = str(short_title).strip()
        if (
            document_title
            and not is_generic_title(document_title, document_type)
            and _STATUTE_SHORT.match(st)
        ):
            return _truncate(f"{st} — {document_title.strip()}", 120)
        return _truncate(st, 120)

    extracted = extract_title_from_passage(passage_text or "", document_type)
    if extracted:
        return extracted

    if document_title and not is_generic_title(document_title, document_type):
        return _truncate(str(document_title).strip(), 120)

    if document_type and not is_generic_title(document_type):
        return str(document_type).strip()

    return "Unknown Document"


def load_doc_metadata_by_url(master_corpus_path: Path) -> dict[str, dict[str, str | None]]:
    """Map source URL → short_title and document_title from raw corpus citation JSON."""
    if not master_corpus_path.is_file():
        return {}

    import pandas as pd

    df = pd.read_csv(master_corpus_path, usecols=["url", "citation_information"])
    out: dict[str, dict[str, str | None]] = {}

    for _, row in df.iterrows():
        url = row.get("url")
        if url is None or (isinstance(url, float) and str(url) == "nan"):
            continue
        url = str(url).strip()
        if not url or url in out:
            continue
        try:
            citation = json.loads(row.get("citation_information") or "{}")
        except (json.JSONDecodeError, TypeError):
            citation = {}

        case_name = citation.get("case_name")
        short = citation.get("short_title") or case_name
        title = citation.get("title") or case_name

        # Court decisions: prefer "G.R. No. X — Case Name" when citation string has docket
        citation_str = citation.get("citation") or ""
        gr = _GR_NO.search(citation_str)
        if gr and case_name:
            short = f"G.R. No. {gr.group('num')} — {_normalize_case_name(case_name)}"
        elif case_name and not short:
            short = _normalize_case_name(case_name)

        out[url] = {
            "short_title": short,
            "document_title": title,
        }

    return out


def _normalize_case_name(name: str) -> str:
    s = re.sub(r"\s+", " ", str(name).strip())
    s = re.sub(r",?\s*Petitioner\s*,?", "", s, flags=re.IGNORECASE)
    s = re.sub(r",?\s*Respondent\s*\.?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*vs\.?\s*", " v. ", s, flags=re.IGNORECASE)
    return _truncate(s.strip(" ,."), 100)
