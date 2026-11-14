"""
Inconsistent-value detection and repair proposals (Phase 14). Method fixed in docs/PHASE14_SPEC.md.

An "inconsistency" is one real-world value written several ways inside one categorical column:
'Male' / 'male' / ' Male' / 'M' / 'Mael'. The older detect.detect_inconsistency only caught case and whitespace
(its docstring promised 'M' too, which it never did). This module finds three kinds, each with its own
confidence tier, and never invents a value that is not already present in the column:

  1. format variants   strict  proposed repair   same value apart from case, spacing, separators, trailing '.', accents
                                                 (a symbol that carries meaning, such as '+', '<', '>', is NEVER ignored)
  2. abbreviations     strict  proposed repair   'M' -> 'Male'; 'NY' -> 'New York'. Only when the match is
                                                 UNIQUE and the smaller form is a minority (<= 30%)
  3. typos             loose   flag only         'Prvate' vs 'Private'. Never repaired: a one-character edit is
                                                 also how two genuinely different categories differ

Whether a strict proposal is applied automatically is NOT decided here. That is the policy's job (measured harm
risk 0, repair precision >= 0.95, ...); this module only reports what it found.

Pure functions, no dependency on taskclean.py (so there is no import cycle).
"""

import re
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd

from detect import DetectionResult

# --- fixed in docs/PHASE14_SPEC.md before any validation run -----------------------------------------------
MAX_LEVELS = 500                    # more distinct KEYS than this: not a categorical column. Keys, not raw spellings:
                                    # heavy corruption turned 14 occupations into 691 spellings and the column was
                                    # skipped (docs/PHASE14_SPEC.md, Amendment 3)
FREE_TEXT_MIN_LEVELS = 50           # ... and, above this many distinct keys, a column where more than
FREE_TEXT_UNIQUE_SHARE = 0.5        # half the rows are distinct is free text / an identifier: skipped
MAX_RAW_VALUES = 10_000             # cost guard only: never build keys for more distinct raw strings than this
ABBREV_MAX_LEN = 3                  # an abbreviation key has at most this many letters
ABBREV_MAX_MINORITY_SHARE = 0.30    # the smaller of the two forms must be at most this share of the pair
TYPO_MIN_LEN = 5                    # both keys at least this long for a typo suggestion
TYPO_MAX_SHARE = 0.05               # the typo form must be at most this share of the correct form
TYPO_MAX_LEVELS = 300               # pairwise typo search is quadratic in distinct values

# Only DECORATION is ignored. A symbol that carries meaning must survive in the key, otherwise '>50K' and '<=50K', or
# 'A+' and 'A-', would be "the same value" (found by the real-data check, docs/PHASE14_SPEC.md Amendment 2).
_SEPARATOR_RUN = re.compile(r"(?<=\w)[\s_\-.]+(?=\w)", re.UNICODE)     # 'Self-emp-inc', 'new_york', 'St. Louis'
_END_DECORATION = " \t\r\n.,;:'\"`\u2019"                                # trailing '.', stray quotes, padding
_SPACES = re.compile(r"\s+")

CELL_COLUMNS = ["row_id", "column", "original_value", "proposed_value", "rule", "tier", "detail", "share"]


def value_key(v: str) -> str:
    """Case-folded, accent-free form of a value in which spacing and separators are normalised but meaningful
    symbols are kept: 'Self-emp-inc' == 'self emp inc' == ' Self_emp_inc.', yet '>50K' != '<=50K' and 'A+' != 'A-'."""
    s = unicodedata.normalize("NFKD", str(v))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).casefold()
    s = s.strip(_END_DECORATION)
    s = _SEPARATOR_RUN.sub(" ", s)
    return _SPACES.sub(" ", s)


def _one_edit(a: str, b: str):
    """None if a and b are not exactly one edit apart (insert, delete, substitute or adjacent swap);
    otherwise True/False for 'the edit involves a digit' (Grade 3 vs Grade 4 is a different value, not a typo)."""
    if a == b:
        return None
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return None
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        if len(diff) == 1:
            i = diff[0]
            return a[i].isdigit() or b[i].isdigit()
        if len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]:
            return any(ch.isdigit() for ch in (a[diff[0]], a[diff[1]]))
        return None
    s, l = (a, b) if la < lb else (b, a)
    i = 0
    while i < len(s) and s[i] == l[i]:
        i += 1
    return l[i].isdigit() if s[i:] == l[i + 1:] else None


def _analyse_column(counts: pd.Series) -> dict[str, dict]:
    """{raw value -> {proposed, rule, tier, detail}} for one column's value counts (index = raw strings)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for raw in counts.index:
        k = value_key(raw)
        if k:
            groups[k].append(raw)
    ent = {}
    for k, raws in groups.items():
        raws = sorted(raws, key=lambda r: (-int(counts[r]), r))      # most frequent first, ties by string: deterministic
        ent[k] = {"total": int(sum(counts[r] for r in raws)), "canon": raws[0], "raws": raws}

    found: dict[str, dict] = {}

    # 1. format variants: several raw spellings of one key
    for k, e in ent.items():
        for r in e["raws"][1:]:
            found[r] = {"proposed": e["canon"], "rule": "format_variant", "tier": "strict",
                        "detail": f"same value as '{e['canon']}' apart from case, spacing, punctuation or accents"}

    # 2. abbreviations: a short key that is a prefix, or the initials, of exactly one other key
    keys = list(ent)
    merge_to: dict[str, str] = {}
    for a in keys:
        flat = a.replace(" ", "")
        if len(flat) > ABBREV_MAX_LEN or flat.isdigit():
            continue
        cands = [b for b in keys if b != a and len(b) > len(a)
                 and (b.startswith(a) or (len(b.split()) >= 2 and "".join(w[0] for w in b.split()) == flat))]
        if not cands:
            continue
        ta = ent[a]["total"]
        if len(cands) == 1:
            b = cands[0]
            tb = ent[b]["total"]
            if min(ta, tb) / (ta + tb) <= ABBREV_MAX_MINORITY_SHARE:
                small, big = (a, b) if ta <= tb else (b, a)
                merge_to[small] = big
        else:
            tmax = max(ent[b]["total"] for b in cands)
            if ta / (ta + tmax) <= ABBREV_MAX_MINORITY_SHARE:
                names = ", ".join(f"'{ent[b]['canon']}'" for b in sorted(cands)[:4])
                for r in ent[a]["raws"]:
                    found.setdefault(r, {"proposed": None, "rule": "ambiguous_abbreviation", "tier": "loose",
                                         "detail": f"could abbreviate more than one value ({names}); not repaired"})

    def final(k):
        for _ in range(4):                       # follow small -> larger chains; strictly growing totals => no cycles
            if k in merge_to:
                k = merge_to[k]
            else:
                break
        return k

    for small in merge_to:
        big = final(small)
        for r in ent[small]["raws"]:
            found[r] = {"proposed": ent[big]["canon"], "rule": "abbreviation", "tier": "strict",
                        "detail": f"'{r}' looks like an abbreviation of '{ent[big]['canon']}' "
                                  "(the only value it could stand for)"}

    # 3. typos: flag only. Entities already merged are skipped; the typo form must be rare and the match unique
    if len(keys) <= TYPO_MAX_LEVELS:
        open_keys = [k for k in keys if k not in merge_to and len(k) >= TYPO_MIN_LEN]
        for a in open_keys:
            hits = []
            for b in open_keys:
                if b != a:
                    r = _one_edit(a, b)
                    if r is False:
                        hits.append(b)
            if len(hits) != 1:
                continue
            b = hits[0]
            if ent[a]["total"] <= TYPO_MAX_SHARE * ent[b]["total"]:
                for r in ent[a]["raws"]:
                    found.setdefault(r, {"proposed": None, "rule": "possible_typo", "tier": "loose",
                                         "detail": f"possible typo of '{ent[b]['canon']}' (flagged, not repaired)"})
    return found


def detect_inconsistency_v2(X: pd.DataFrame) -> DetectionResult:
    """Row-level rate = share of rows holding at least one flagged (non-canonical) cell. `extra["cells"]` is one
    row per flagged cell: row_id, column, original_value, proposed_value (None for flag-only), rule, tier, detail."""
    cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    frames, analysed, skipped = [], [], {}
    for col in cols:
        s = X[col].astype(object)
        nn = s.dropna()
        if nn.nunique() < 2:
            continue
        counts = nn.astype(str).value_counts()
        if len(counts) > MAX_RAW_VALUES:
            skipped[col] = f"{len(counts):,} distinct values: looks like free text or an identifier"
            continue
        n_keys = len({value_key(v) for v in counts.index})
        if n_keys > MAX_LEVELS or (n_keys > FREE_TEXT_MIN_LEVELS and n_keys > FREE_TEXT_UNIQUE_SHARE * len(nn)):
            skipped[col] = f"{n_keys:,} distinct values: looks like free text or an identifier"
            continue
        analysed.append(col)
        found = _analyse_column(counts)
        if not found:
            continue
        sv = s.where(s.isna(), s.astype(str))
        hit = sv.map(lambda v: v in found if isinstance(v, str) else False)
        if not hit.any():
            continue
        orig = sv[hit]
        info = orig.map(found)
        frames.append(pd.DataFrame({
            "row_id": orig.index.values, "column": col, "original_value": orig.values,
            "proposed_value": [i["proposed"] for i in info], "rule": [i["rule"] for i in info],
            "tier": [i["tier"] for i in info], "detail": [i["detail"] for i in info],
            "share": (orig.map(counts) / len(nn)).values}))
    cells = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CELL_COLUMNS)
    row_mask = pd.Series(False, index=X.index)
    if len(cells):
        row_mask.loc[cells["row_id"].unique()] = True
    return DetectionResult("inconsistency", float(row_mask.mean()) if len(X) else 0.0, row_mask,
                           {"cells": cells, "columns": analysed, "skipped_columns": skipped,
                            "n_strict": int((cells.tier == "strict").sum()) if len(cells) else 0,
                            "n_loose": int((cells.tier == "loose").sum()) if len(cells) else 0})
