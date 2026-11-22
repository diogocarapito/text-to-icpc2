"""Page text, loaded from content/<lang>.md.

- YAML front matter (between the two `---` lines): the short interface strings.
- Body: Markdown. Each `# Heading` starts a collapsible section under the search box
  (e.g. "Sobre o projeto", "API"); add or remove sections by adding or removing headings.

Files are re-read when they change on disk, so text edits show up on the next page load.
"""
import re
from pathlib import Path

import yaml
from markdown_it import MarkdownIt

DIR = Path(__file__).parent / "content"
LANGS = ("pt", "en")
_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)
_H1 = re.compile(r"^# +(.+?)\s*$", re.MULTILINE)
_md = MarkdownIt("commonmark", {"html": False}).enable("table")
_cache: dict[str, tuple[float, dict]] = {}


def _render(md: str) -> str:
    html = _md.render(md)
    # external links open in a new tab; site-relative ones (/docs) stay in place
    return html.replace('<a href="http', '<a target="_blank" rel="noopener" href="http')


def _sections(body: str) -> list[dict]:
    """Split on `# Heading` lines, ignoring `#` lines inside ``` code blocks."""
    out, title, buf, fenced = [], None, [], False
    for line in body.splitlines(keepends=True):
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
        h = None if fenced else _H1.match(line)
        if h:
            if title is not None or "".join(buf).strip():
                out.append({"title": title or "", "html": _render("".join(buf))})
            title, buf = h.group(1), []
        else:
            buf.append(line)
    if title is not None or "".join(buf).strip():
        out.append({"title": title or "", "html": _render("".join(buf))})
    return out


def load(lang: str) -> dict:
    """{"ui": {key: text}, "sections": [{"title", "html"}]} for one language (falls back to pt)."""
    lang = lang if lang in LANGS else "pt"
    p = DIR / f"{lang}.md"
    mtime = p.stat().st_mtime
    hit = _cache.get(lang)
    if hit and hit[0] == mtime:
        return hit[1]
    m = _FRONT.match(p.read_text(encoding="utf-8"))
    if not m:
        raise ValueError(f"{p}: expected front matter between two '---' lines")
    ui = {k: str(v) for k, v in (yaml.safe_load(m.group(1)) or {}).items()}
    data = {"ui": ui, "sections": _sections(m.group(2))}
    _cache[lang] = (mtime, data)
    return data


def ui(lang: str) -> dict:
    """UI strings for lang, with pt filling any key missing from that language."""
    base = load("pt")["ui"]
    return {**base, **load(lang)["ui"]} if lang != "pt" else base


def text(lang: str, key: str, **fmt) -> str:
    return ui(lang).get(key, key).format(**fmt)
