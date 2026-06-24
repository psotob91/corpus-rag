"""Acquire a source: DOI -> open-access PDF cascade (robots/ToS-respecting).

Given a DOI we try an ETHICAL open-access cascade and download the first source
that yields an OA full text. Priority (rescued from ms-rie
`orchestrator/fetch_source.py` + `learning/SOURCES_KB.md`):

    Europe PMC / NCBI  >  Unpaywall  >  Crossref

All network calls live inside `fetch()` (nothing at import time). Every request
carries an honest User-Agent with a contact (from the `email` arg or the
`SCRAPER_CONTACT` env var), a polite timeout, and respects HTTP status. We NEVER
bypass paywalls / anti-bot and NEVER use stealth headers. If nothing OA is found
we raise `FetchBlocked` with a message pointing at manual rescue.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import quote

# Polite, bounded network behaviour.
_TIMEOUT = 30  # seconds per request
_DEFAULT_CONTACT = "corpus-rag@example.org"
_PDF_MAGIC = b"%PDF-"


class FetchBlocked(Exception):
    """Raised when no open-access full text could be ethically resolved for a DOI.

    The message points the caller at manual rescue (download the PDF by hand and
    pass its local path to `fetch()` instead).
    """


def fetch(
    doi_or_path: str,
    cache_dir: str = "corpus/_fetched",
    email: str | None = None,
) -> str:
    """Resolve `doi_or_path` to a local PDF/HTML path.

    - If `doi_or_path` is an existing local path, return it unchanged.
    - Otherwise treat it as a DOI and run the OA cascade
      (Europe PMC -> Unpaywall -> Crossref), downloading the first OA full text.

    Raises FetchBlocked when the input is not a local file, not a usable DOI, or
    when no open-access full text is found.
    """
    if not doi_or_path or not str(doi_or_path).strip():
        raise FetchBlocked("empty input: pass a local PDF/HTML path or a DOI")

    raw = str(doi_or_path).strip()

    # 1) Existing local path -> return as-is (no network).
    p = Path(raw)
    if p.exists() and p.is_file():
        return str(p)

    # 2) Must look like a DOI to proceed with network resolution.
    doi = _normalize_doi(raw)
    if doi is None:
        raise FetchBlocked(
            f"not a local file and not a recognizable DOI: {raw!r}. "
            "Pass an existing PDF/HTML path, or a DOI like '10.1371/journal.pone.0173955'."
        )

    contact = _resolve_contact(email)
    headers = {
        "User-Agent": f"corpus-rag/0.1 (open-access fetch; mailto:{contact})",
        "Accept": "application/json",
    }

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    slug = _doi_slug(doi)

    import requests  # lazy: keep import-time light

    session = requests.Session()
    session.headers.update(headers)

    attempts: list[str] = []

    # ---- Cascade, in priority order. Each resolver returns a URL or None. ----
    for resolver in (_resolve_europepmc, _resolve_unpaywall, _resolve_crossref):
        try:
            url = resolver(session, doi, contact)
        except Exception as exc:  # network/JSON hiccup: record and continue
            attempts.append(f"{resolver.__name__}: error {type(exc).__name__}: {exc}")
            continue
        if not url:
            attempts.append(f"{resolver.__name__}: no OA full text")
            continue
        # Try to download what the resolver pointed us at.
        try:
            dest = _download(session, url, cache, slug)
        except Exception as exc:
            attempts.append(f"{resolver.__name__}: url={url} download failed: {exc}")
            continue
        if dest is not None:
            return str(dest)
        attempts.append(f"{resolver.__name__}: url={url} not a usable full text")

    raise FetchBlocked(
        "no open-access full text resolved for DOI "
        f"{doi} via Europe PMC/Unpaywall/Crossref. "
        "Manual rescue: download the article PDF yourself (respecting the "
        "publisher's terms) and call fetch() with the local file path. "
        "Tried -> " + " | ".join(attempts)
    )


# ---------------------------- resolvers ----------------------------


def _resolve_europepmc(session, doi: str, contact: str) -> str | None:
    """Europe PMC: search by DOI -> PMCID -> OA full-text PDF URL.

    Priority source (NCBI/EuropePMC > Unpaywall > Crossref) per SOURCES_KB.
    """
    search = (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        f"?query={quote('DOI:' + doi)}&format=json&pageSize=1&resultType=core"
    )
    r = session.get(search, timeout=_TIMEOUT, headers={"Accept": "application/json"})
    if r.status_code != 200:
        return None
    data = r.json()
    results = (data.get("resultList") or {}).get("result") or []
    if not results:
        return None
    rec = results[0]

    # Prefer an explicit PDF link in the fullTextUrlList.
    for url in (rec.get("fullTextUrlList") or {}).get("fullTextUrl", []):
        if not isinstance(url, dict):
            continue
        avail = (url.get("availability") or "").lower()
        doc_style = (url.get("documentStyle") or "").lower()
        link = url.get("url")
        if link and doc_style == "pdf" and avail in ("open access", "free"):
            return link

    # Fall back to the Europe PMC OA PDF render endpoint when a PMCID exists.
    pmcid = rec.get("pmcid")
    is_oa = (rec.get("isOpenAccess") or "").upper() == "Y" or (rec.get("inEPMC") or "").upper() == "Y"
    if pmcid and is_oa:
        return (
            "https://www.ebi.ac.uk/europepmc/webservices/rest/"
            f"{quote(pmcid)}/fullTextPDF"
        )
    return None


def _resolve_unpaywall(session, doi: str, contact: str) -> str | None:
    """Unpaywall: best_oa_location.url_for_pdf (email is REQUIRED by their API)."""
    api = f"https://api.unpaywall.org/v2/{quote(doi)}?email={quote(contact)}"
    r = session.get(api, timeout=_TIMEOUT, headers={"Accept": "application/json"})
    if r.status_code != 200:
        return None
    data = r.json()
    if not data.get("is_oa"):
        return None
    loc = data.get("best_oa_location") or {}
    url = loc.get("url_for_pdf") or loc.get("url")
    if url:
        return url
    # Try any OA location with a PDF.
    for loc in data.get("oa_locations") or []:
        if isinstance(loc, dict) and loc.get("url_for_pdf"):
            return loc["url_for_pdf"]
    return None


def _resolve_crossref(session, doi: str, contact: str) -> str | None:
    """Crossref: a `link` entry with content-type application/pdf (when OA)."""
    api = f"https://api.crossref.org/works/{quote(doi)}"
    r = session.get(
        api,
        timeout=_TIMEOUT,
        headers={"Accept": "application/json", "User-Agent": f"corpus-rag/0.1 (mailto:{contact})"},
    )
    if r.status_code != 200:
        return None
    msg = (r.json() or {}).get("message") or {}
    links = msg.get("link") or []
    # Prefer explicit application/pdf links.
    for link in links:
        if not isinstance(link, dict):
            continue
        if (link.get("content-type") or "").lower() == "application/pdf" and link.get("URL"):
            return link["URL"]
    # Some publishers tag the full-text PDF as 'unspecified' content-type.
    for link in links:
        if not isinstance(link, dict):
            continue
        url = link.get("URL")
        if url and url.lower().endswith(".pdf"):
            return url
    return None


# ---------------------------- helpers ----------------------------


def _download(session, url: str, cache: Path, slug: str) -> Path | None:
    """Download `url` to the cache. Return the path iff it is a real PDF/HTML.

    Respects HTTP status; returns None (rather than raising) when the body is not
    usable full text, so the caller can continue the cascade.
    """
    r = session.get(
        url,
        timeout=_TIMEOUT,
        stream=True,
        allow_redirects=True,
        headers={"Accept": "application/pdf, text/html;q=0.5, */*;q=0.1"},
    )
    if r.status_code != 200:
        r.close()
        return None

    ctype = (r.headers.get("Content-Type") or "").lower()
    body = r.content  # bounded by server; OA articles are small enough
    r.close()

    if not body:
        return None

    # PDF: detect by content-type or magic bytes.
    if "application/pdf" in ctype or body[:5] == _PDF_MAGIC:
        dest = cache / f"{slug}.pdf"
        dest.write_bytes(body)
        return dest

    # HTML/XML full text is acceptable (Europe PMC / publisher landing full text).
    if "text/html" in ctype or "application/xml" in ctype or "text/xml" in ctype:
        # Guard against tiny error/landing stubs masquerading as full text.
        if len(body) < 512:
            return None
        ext = ".xml" if ("xml" in ctype) else ".html"
        dest = cache / f"{slug}{ext}"
        dest.write_bytes(body)
        return dest

    return None


def _normalize_doi(raw: str) -> str | None:
    """Extract a bare DOI from a raw string or DOI URL. Return None if not a DOI."""
    s = raw.strip()
    # Strip common URL / scheme prefixes.
    s = re.sub(r"(?i)^https?://(dx\.)?doi\.org/", "", s)
    s = re.sub(r"(?i)^doi:\s*", "", s)
    s = s.strip()
    m = re.match(r"(?i)(10\.\d{4,9}/\S+)", s)
    if not m:
        return None
    return m.group(1)


def _resolve_contact(email: str | None) -> str:
    """Pick a contact for the User-Agent / Unpaywall email, honestly."""
    if email and email.strip():
        return email.strip()
    env = os.environ.get("SCRAPER_CONTACT")
    if env and env.strip():
        return env.strip()
    return _DEFAULT_CONTACT


def _doi_slug(doi: str) -> str:
    """Filesystem-safe slug for a DOI (for the cached filename)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", doi).strip("_") or "fetched"
