"""The ICPC-2 label space (components 1 + 7 -> 686 codes) and per-code metadata."""

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from icpc2.config import load_config, path

CHAPTERS_PT = {
    "A": "Geral e inespecífico", "B": "Sangue e órgãos hematopoiéticos", "D": "Aparelho digestivo",
    "F": "Olhos", "H": "Ouvidos", "K": "Aparelho circulatório", "L": "Sistema musculoesquelético",
    "N": "Sistema nervoso", "P": "Psicológico", "R": "Aparelho respiratório", "S": "Pele",
    "T": "Endócrino, metabólico e nutricional", "U": "Aparelho urinário",
    "W": "Gravidez e planeamento familiar", "X": "Aparelho genital feminino",
    "Y": "Aparelho genital masculino", "Z": "Problemas sociais",
}


@dataclass(frozen=True)
class Code:
    code: str
    chapter: str
    component: str
    title_pt: str
    short_pt: str
    inclusion: tuple[str, ...]
    exclusion: str
    criteria: str
    icd10: tuple[str, ...]          # WHO ICD-10 codes without dot, e.g. ("A000", "A001")
    title_en: str = ""


def _split(s, sep=";") -> tuple[str, ...]:
    if not isinstance(s, str) or not s.strip():
        return ()
    return tuple(p.strip() for p in s.split(sep) if p.strip())


@lru_cache(maxsize=1)
def load_codes() -> dict[str, Code]:
    cfg = load_config("data")
    df = pd.read_csv(path(cfg["raw"]["icpc2"]), dtype=str).fillna("")
    df = df[df["componente"].isin(cfg["components"])]

    en_titles: dict[str, str] = {}
    en_path = path(cfg["raw"]["icpc2_en_titles"])
    if en_path.exists():
        en = pd.read_csv(en_path, dtype=str).fillna("")
        en_titles = dict(zip(en["code"], en["title_en"]))

    codes = {}
    for r in df.itertuples(index=False):
        codes[r.cod] = Code(
            code=r.cod,
            chapter=r.cod[0],
            component=r.componente,
            title_pt=r.nome.strip(),
            short_pt=r.nome_curto.strip(),
            inclusion=_split(r.incl),
            exclusion=r.excl.strip(),
            criteria=r.crit.strip(),
            icd10=tuple(c.replace(".", "") for c in _split(r.icd10.replace(" ", ""))),
            title_en=en_titles.get(r.cod, ""),
        )
    return dict(sorted(codes.items()))


def label_list() -> list[str]:
    """Stable, sorted label order. Index in this list == classifier label id."""
    return list(load_codes().keys())


def codes_table() -> list[dict]:
    """Compact metadata shipped inside every model artifact (used by the web app)."""
    return [
        {"code": c.code, "chapter": c.chapter, "title_pt": c.title_pt, "title_en": c.title_en,
         "inclusion": list(c.inclusion)}
        for c in load_codes().values()
    ]
