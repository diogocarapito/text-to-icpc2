"""Page text, loaded from content/<lang>.md (YAML front matter = UI strings, body = About section).

Files are re-read when they change on disk, so text edits show up on the next page load.
"""
import re
from pathlib import Path

import yaml
from markdown_it import MarkdownIt

DIR = Path(__file__).parent / "content"
LANGS = ("pt", "en")
_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.S)
_md = MarkdownIt("commonmark", {"html": False}).enable("table")
_cache: dict[str, tuple[float, dict]] = {}


def load(lang: str) -> dict:
    """{"ui": {key: text}, "about_html": str} for one language (falls back to pt)."""
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
    about = _md.render(m.group(2)).replace("<a href=", '<a target="_blank" rel="noopener" href=')
    data = {"ui": ui, "about_html": about}
    _cache[lang] = (mtime, data)
    return data


def ui(lang: str) -> dict:
    """UI strings for lang, with pt filling any key missing from that language."""
    base = load("pt")["ui"]
    return {**base, **load(lang)["ui"]} if lang != "pt" else base


def text(lang: str, key: str, **fmt) -> str:
    return ui(lang).get(key, key).format(**fmt)
