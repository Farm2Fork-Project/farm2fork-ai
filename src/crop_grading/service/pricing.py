"""Price references from dated official wholesale prices (AMIS Punjab).

Each answer is an observed mandi price range, never a modelled estimate: the
snapshot in configs/official_prices.json (scripts/update_official_prices.py)
records the market, trading date, minimum, maximum and fair-quality price
(FQP) for each commodity. A farmer's own market is used when it reported;
otherwise the middle half of fair-quality prices across Punjab markets.

Products the snapshot does not cover get no price rather than a guess.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import quantiles


class PriceNotEstimableError(ValueError):
    """No official price covers this product or unit."""


@dataclass(frozen=True)
class PriceEstimate:
    min_price: float
    max_price: float
    unit: str
    commodity: str
    observed_on: date
    market_scope: str  # market name, or "Punjab"
    market_count: int
    version: str


# Farmer wording -> AMIS commodity names. Varieties are checked before crops.
VARIETY_ALIASES = {
    "basmati": ("Rice Basmati Super (New)",),
    "irri": ("Rice (IRRI)",),
    "chaunsa": ("Mango(Chounsa)",),
    "chounsa": ("Mango(Chounsa)",),
    "sindhri": ("Mango(Sindhri)",),
    "dasehri": ("Mango(Desahri)",),
    "dusehri": ("Mango(Desahri)",),
    "dussehri": ("Mango(Desahri)",),
    "ratol": ("Mango(Anwer Ratol)",),
    "ratul": ("Mango(Anwer Ratol)",),
}
CROP_ALIASES = {
    "wheat": ("Wheat",),
    "gandum": ("Wheat",),
    "rice": ("Rice Basmati Super (New)", "Rice (IRRI)"),
    "chawal": ("Rice Basmati Super (New)", "Rice (IRRI)"),
    "maize": ("Maize",),
    "corn": ("Maize",),
    "makai": ("Maize",),
    "cotton": ("Seed Cotton(Phutti)",),
    "phutti": ("Seed Cotton(Phutti)",),
    "kapas": ("Seed Cotton(Phutti)",),
    "mango": ("Mango*",),
    "aam": ("Mango*",),
}
# First words that name a family, not a commodity ("Green Chilli", "Gram Pulse").
AMBIGUOUS_FIRST_WORDS = {
    "green", "red", "sweet", "lady", "bitter", "bottle", "gram", "mash", "moong",
    "masoor", "rice", "paddy", "wheat", "seed", "banola", "brown", "mango",
}

# Rupees per kilogram-type AMIS unit, per requested unit.
UNIT_FACTORS = {
    "100kg": {"kg": 0.01, "ton": 10.0},
    "dozen": {"dozen": 1.0},
    "100pcs": {"piece": 0.01},
}


def _words(text: str) -> set[str]:
    words = set(re.findall(r"[a-z]+", text.lower()))
    return words | {w[:-2] for w in words if w.endswith("es")} | {w[:-1] for w in words if w.endswith("s")}


def _base_words(name: str) -> list[str]:
    return re.findall(r"[a-z]+", re.sub(r"\(.*?\)", " ", name.lower()))


class OfficialPrices:
    def __init__(self, path: Path) -> None:
        with path.open(encoding="utf-8") as handle:
            document = json.load(handle)
        self.source: str = document["source"]
        self.source_url: str = document["source_url"]
        self.fetched_at: str = document["fetched_at_pkt"]
        self.version = "amis-punjab-" + self.fetched_at[:10]
        self._commodities: dict = document["commodities"]

    def _match(self, product_name: str) -> list[str]:
        words = _words(product_name)
        for aliases in (VARIETY_ALIASES, CROP_ALIASES):
            names = [n for word, found in aliases.items() if word in words for n in found]
            if names:
                return self._expand(names)
        exact = [n for n in self._commodities if set(_base_words(n)) <= words]
        if exact:
            return exact
        return [
            n
            for n in self._commodities
            if (base := _base_words(n))
            and base[0] not in AMBIGUOUS_FIRST_WORDS
            and base[0] in words
        ]

    def _expand(self, names: list[str]) -> list[str]:
        found = []
        for name in names:
            if name.endswith("*"):
                found += [n for n in self._commodities if n.startswith(name[:-1])]
            elif name in self._commodities:
                found.append(name)
        return list(dict.fromkeys(found))

    def estimate(self, *, product_name: str, unit: str, location: str | None = None) -> PriceEstimate:
        unit = unit.lower()
        names = self._match(product_name)
        names = [n for n in names if unit in UNIT_FACTORS[self._commodities[n]["unit"]]]
        if not names:
            raise PriceNotEstimableError(
                f"No official wholesale price covers '{product_name}' sold per {unit}."
            )
        # Prefer the most recent trading day among the matched commodities.
        latest = max(self._commodities[n]["observed_on"] for n in names)
        names = [n for n in names if self._commodities[n]["observed_on"] == latest]
        factor = UNIT_FACTORS[self._commodities[names[0]]["unit"]][unit]
        rows = [(market, row) for n in names for market, row in self._commodities[n]["markets"].items()]

        local = self._local_rows(rows, location)
        if local:
            low = min(row["min"] for _, row in local)
            high = max(row["max"] for _, row in local)
            scope, count = local[0][0], 1
        else:
            fair = sorted(row["fqp"] for _, row in rows)
            if len(fair) >= 4:
                q1, _, q3 = quantiles(fair, n=4, method="inclusive")
                low, high = q1, q3
            else:
                low, high = fair[0], fair[-1]
            scope, count = "Punjab", len({market for market, _ in rows})

        return PriceEstimate(
            min_price=round(low * factor, 2),
            max_price=round(high * factor, 2),
            unit=unit,
            commodity=", ".join(names),
            observed_on=date.fromisoformat(latest),
            market_scope=scope,
            market_count=count,
            version=self.version,
        )

    @staticmethod
    def _local_rows(rows: list[tuple[str, dict]], location: str | None) -> list[tuple[str, dict]]:
        if not location:
            return []
        wanted = re.sub(r"[^a-z]", "", location.lower())
        if not wanted:
            return []
        return [(m, r) for m, r in rows if re.sub(r"[^a-z]", "", m.lower()) == wanted]
