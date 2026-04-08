"""
Агрегирование первичных меток arXiv в обобщённые классы.

Основано на официальной таксономии arXiv (названия и смысл class ID):
https://arxiv.org/category_taxonomy

И на разъяснениях по пересекающимся ML-направлениям (cs.LG vs stat.ML и др.):
https://blog.arxiv.org/2019/12/05/arxiv-machine-learning-classification-guide/

Смысл superclass (кратко):
- Компьютерное зрение — зрение, графика, изображения/видео (cs.CV, eess.IV, cs.GR, cs.MM)
- Обработка естественного языка — язык и NLP (cs.CL, устар. cmp-lg)
- Машинное обучение — ML и нейросети (cs.LG, stat.ML, cs.NE; устар. cs.DL)
- ИИ-агенты — ИИ вне «ядра» ML/NLP/Vision: рассуждения, планирование, мультиагенты (cs.AI, cs.MA)
- Инженерия данных — данные, ИР, сети, распределённые системы (cs.IR, cs.DB, cs.DC, cs.NI)
- Алгоритмы — алгоритмы и теория КС (cs.DS, cs.DM, cs.CG, cs.GT, cs.CC, cs.LO, cs.FL, cs.IT, cs.CE)
- Програмная инженерия — ПО, языки, ОС, архитектура, перформанс (cs.SE, cs.PL, cs.OS, cs.AR, cs.PF, cs.MS)
- Информационная безопасность — криптография и безопасность (cs.CR)
- Робототехника — робототехника и управление (cs.RO, cs.SY, eess.SY)
- Человеко-ориентированные системы — HCI, соц. сети, общество/этика (cs.HC, cs.SI, cs.CY)
- Аудио и обработка сигналов — речь, звук, сигналы (cs.SD, eess.AS, eess.SP)
- Математика и статистика — без ML (math.*, cs.NA, stat.* кроме stat.ML)
- Физика и STEM — физика, астро, материя, ЯФ и смежные архивы
- Экономика и финансы — q-bio, q-fin, econ
- Прочие компьютерные науки — прочий CS, явно не попавший в группы выше
"""

from __future__ import annotations

import re

# Порядок = id классификатора (0…14), строки как в model config id2label.
SUPERCLASS_ORDER = (
    "ИИ-агенты",
    "Алгоритмы",
    "Аудио и обработка сигналов",
    "Инженерия данных",
    "Человеко-ориентированные системы",
    "Экономика и финансы",
    "Математика и статистика",
    "Машинное обучение",
    "Обработка естественного языка",
    "Прочие компьютерные науки",
    "Робототехника",
    "Информационная безопасность",
    "Програмная инженерия",
    "Физика и STEM",
    "Компьютерное зрение",
)

_TO_IDX = {name: i for i, name in enumerate(SUPERCLASS_ORDER)}


def label_id(name: str) -> int:
    return _TO_IDX[name]


def label_maps() -> tuple[dict[str, int], dict[int, str]]:
    label2id = {name: i for i, name in enumerate(SUPERCLASS_ORDER)}
    id2label = {i: name for i, name in enumerate(SUPERCLASS_ORDER)}
    return label2id, id2label


SUPERCLASS_DESCRIPTIONS: dict[str, str] = {
    "ИИ-агенты": "ИИ вне «ядра» ML/NLP/Vision: рассуждения, планирование, мультиагенты (cs.AI, cs.MA)",
    "Алгоритмы": "алгоритмы и теория КС (cs.DS, cs.DM, cs.CG, cs.GT, cs.CC, cs.LO, cs.FL, cs.IT, cs.CE)",
    "Аудио и обработка сигналов": "речь, звук, сигналы (cs.SD, eess.AS, eess.SP)",
    "Инженерия данных": "данные, ИР, сети, распределённые системы (cs.IR, cs.DB, cs.DC, cs.NI)",
    "Человеко-ориентированные системы": "HCI, соц. сети, общество/этика (cs.HC, cs.SI, cs.CY)",
    "Экономика и финансы": "q-bio, q-fin, econ",
    "Математика и статистика": "без ML (math.*, cs.NA, stat.* кроме stat.ML)",
    "Машинное обучение": "ML и нейросети (cs.LG, stat.ML, cs.NE; устар. cs.DL)",
    "Обработка естественного языка": "язык и NLP (cs.CL, устар. cmp-lg)",
    "Прочие компьютерные науки": "прочий CS, явно не попавший в группы выше",
    "Робототехника": "робототехника и управление (cs.RO, cs.SY, eess.SY)",
    "Информационная безопасность": "криптография и безопасность (cs.CR)",
    "Програмная инженерия": "ПО, языки, ОС, архитектура, перформанс (cs.SE, cs.PL, cs.OS, cs.AR, cs.PF, cs.MS)",
    "Физика и STEM": "физика, астро, материя, ЯФ и смежные архивы",
    "Компьютерное зрение": "зрение, графика, изображения/видео (cs.CV, eess.IV, cs.GR, cs.MM)",
}


def _pairs(pairs: list[tuple[list[str], str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for ids, sup in pairs:
        for i in ids:
            out[i] = sup
    return out


ARXIV_TO_SUPER: dict[str, str] = _pairs(
    [
        (
            ["cs.CV", "eess.IV", "cs.GR", "cs.MM"],
            "Компьютерное зрение",
        ),
        (
            ["cs.CL", "cmp-lg"],
            "Обработка естественного языка",
        ),
        (
            ["cs.LG", "stat.ML", "cs.NE", "cs.DL"],
            "Машинное обучение",
        ),
        (
            ["cs.AI", "cs.MA"],
            "ИИ-агенты",
        ),
        (
            ["cs.IR", "cs.DB", "cs.DC", "cs.NI"],
            "Инженерия данных",
        ),
        (
            [
                "cs.DS",
                "cs.DM",
                "cs.CG",
                "cs.GT",
                "cs.CC",
                "cs.LO",
                "cs.FL",
                "cs.IT",
                "cs.CE",
            ],
            "Алгоритмы",
        ),
        (
            ["cs.SE", "cs.PL", "cs.OS", "cs.AR", "cs.PF", "cs.MS"],
            "Програмная инженерия",
        ),
        (
            ["cs.CR"],
            "Информационная безопасность",
        ),
        (
            ["cs.RO", "cs.SY", "eess.SY"],
            "Робототехника",
        ),
        (
            ["cs.HC", "cs.SI", "cs.CY"],
            "Человеко-ориентированные системы",
        ),
        (
            ["cs.SD", "eess.AS", "eess.SP"],
            "Аудио и обработка сигналов",
        ),
        (
            ["cs.NA"],
            "Математика и статистика",
        ),
        (
            ["cs.OH", "cs.ET", "cs.GL", "cs.SC"],
            "Прочие компьютерные науки",
        ),
    ]
)

_MATH_PREFIX = "math."
_STAT_PREFIX = "stat."

_PHYSICS_ASTRONOMY = [
    "adap-org",
    "astro-ph",
    "astro-ph.CO",
    "astro-ph.EP",
    "astro-ph.GA",
    "astro-ph.IM",
    "astro-ph.SR",
    "cond-mat",
    "cond-mat.dis-nn",
    "cond-mat.mtrl-sci",
    "cond-mat.other",
    "cond-mat.soft",
    "cond-mat.stat-mech",
    "cond-mat.supr-con",
    "gr-qc",
    "hep-ex",
    "hep-lat",
    "hep-ph",
    "hep-th",
    "nucl-th",
    "nlin.AO",
    "nlin.CD",
    "nlin.CG",
    "nlin.PS",
    "quant-ph",
    "physics.ao-ph",
    "physics.bio-ph",
    "physics.chem-ph",
    "physics.class-ph",
    "physics.comp-ph",
    "physics.data-an",
    "physics.gen-ph",
    "physics.geo-ph",
    "physics.hist-ph",
    "physics.ins-det",
    "physics.med-ph",
    "physics.optics",
    "physics.soc-ph",
]

_LIFE_FIN_ECON = [
    "econ.EM",
    "q-bio.BM",
    "q-bio.CB",
    "q-bio.GN",
    "q-bio.MN",
    "q-bio.NC",
    "q-bio.PE",
    "q-bio.QM",
    "q-bio.TO",
    "q-fin.CP",
    "q-fin.EC",
    "q-fin.GN",
    "q-fin.PM",
    "q-fin.RM",
    "q-fin.ST",
    "q-fin.TR",
]

_MATH_CATS = [
    "math.AG",
    "math.AP",
    "math.AT",
    "math.CA",
    "math.CO",
    "math.CT",
    "math.DG",
    "math.DS",
    "math.FA",
    "math.GM",
    "math.GN",
    "math.GR",
    "math.GT",
    "math.HO",
    "math.LO",
    "math.MG",
    "math.NA",
    "math.NT",
    "math.OC",
    "math.PR",
    "math.RA",
    "math.RT",
    "math.ST",
]

_STAT_NON_ML = ["stat.AP", "stat.CO", "stat.ME", "stat.OT", "stat.TH"]

for _id in _PHYSICS_ASTRONOMY:
    ARXIV_TO_SUPER[_id] = "Физика и STEM"
for _id in _LIFE_FIN_ECON:
    ARXIV_TO_SUPER[_id] = "Экономика и финансы"
for _id in _MATH_CATS:
    ARXIV_TO_SUPER[_id] = "Математика и статистика"
for _id in _STAT_NON_ML:
    ARXIV_TO_SUPER[_id] = "Математика и статистика"


def arxiv_primary_to_superclass(arxiv_id: str) -> str:
    """Первичная категория arXiv → обобщённый класс обучения."""
    if arxiv_id in ARXIV_TO_SUPER:
        return ARXIV_TO_SUPER[arxiv_id]

    if arxiv_id.startswith(_MATH_PREFIX):
        return "Математика и статистика"
    if arxiv_id.startswith(_STAT_PREFIX):
        if arxiv_id == "stat.ML":
            return "Машинное обучение"
        return "Математика и статистика"
    if arxiv_id.startswith("cs."):
        return "Прочие компьютерные науки"
    if arxiv_id.startswith("physics.") or arxiv_id.startswith("astro-ph"):
        return "Физика и STEM"
    if arxiv_id.startswith("cond-mat"):
        return "Физика и STEM"
    if arxiv_id.startswith("hep-") or arxiv_id.startswith("nucl-th"):
        return "Физика и STEM"
    if arxiv_id.startswith("nlin."):
        return "Физика и STEM"
    if arxiv_id.startswith("q-bio."):
        return "Экономика и финансы"
    if arxiv_id.startswith("q-fin."):
        return "Экономика и финансы"
    if arxiv_id.startswith("econ."):
        return "Экономика и финансы"
    if arxiv_id.startswith("eess."):
        if arxiv_id in ("eess.AS", "eess.SP"):
            return "Аудио и обработка сигналов"
        if arxiv_id == "eess.IV":
            return "Компьютерное зрение"
        if arxiv_id == "eess.SY":
            return "Робототехника"
        return "Аудио и обработка сигналов"
    return "Физика и STEM"


_TAG_TERM = re.compile(r"'term':\s*'([^']+)'")


def primary_arxiv_code_from_record(record: dict) -> str | None:
    """Первый primary category в поле tag (repr списка Atom)."""
    m = _TAG_TERM.search(record["tag"])
    return m.group(1) if m else None


def raw_records_labeled_pairs(records: list[dict]) -> list[tuple[dict, str]]:
    """raw_ds записи → (record, superclass); записи без разобранного tag пропускаются."""
    out: list[tuple[dict, str]] = []
    for rec in records:
        code = primary_arxiv_code_from_record(rec)
        if not code:
            continue
        out.append((rec, arxiv_primary_to_superclass(code)))
    return out
