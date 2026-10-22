"""Snapshot official wholesale prices from Punjab's AMIS into configs/official_prices.json.

AMIS (Agriculture Marketing Information Service, Government of the Punjab) publishes
daily wholesale minimum, maximum and fair-quality prices per commodity and market.
Every value kept here carries its market and trading date; nothing is estimated.

Today's prices are read for every listed commodity. Core crops that did not trade
recently (e.g. mango out of season) are walked back until their last trading day.

    ./.venv/bin/python scripts/update_official_prices.py
"""

from __future__ import annotations

import argparse
import html
import http.cookiejar
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://www.amis.pk"
COMMODITY_URL = BASE + "/ViewPrices.aspx?searchType=0&commodityId={id}"
MARKET_LIST_URL = BASE + "/ViewPrices.aspx?searchType=1&commodityId=1"
PKT = ZoneInfo("Asia/Karachi")

# Commodities the grader and listings care about most: walk back to their last trade.
CORE = ("Wheat", "Rice Basmati Super (New)", "Rice (IRRI)", "Maize", "Seed Cotton(Phutti)")
CORE_PREFIXES = ("Mango",)


def _cells(row: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", cell))).strip()
        for cell in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
    ]


def _price(value: str) -> float | None:
    value = value.replace(",", "")
    return float(value) if re.fullmatch(r"\d+(\.\d+)?", value) else None


def unit_of(name: str) -> str:
    lowered = name.lower().replace(" ", "")
    if "(dozen)" in lowered:
        return "dozen"
    if "100pcs" in lowered:
        return "100pcs"
    return "100kg"  # AMIS: "WholeSalePrices Of Agricultural Commodities (Rs/100Kg)"


class Amis:
    def __init__(self, delay: float) -> None:
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        )
        self._delay = delay

    def _get(self, url: str, data: dict | None = None) -> str:
        time.sleep(self._delay)
        body = urllib.parse.urlencode(data).encode() if data else None
        for attempt in range(3):
            try:
                with self._opener.open(url, data=body, timeout=40) as response:
                    return response.read().decode("utf-8", "ignore")
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        raise AssertionError("unreachable")

    def commodities(self) -> dict[str, int]:
        page = self._get(MARKET_LIST_URL)
        found = {}
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", page, re.S):
            link = re.search(r"CommodityChart\.aspx\?cmd=(\d+)", row)
            cells = _cells(row)
            if link and cells:
                found[re.sub(r"^\d+\s+", "", cells[0])] = int(link.group(1))
        return found

    def prices(self, commodity_id: int, day: date | None = None) -> tuple[date, dict]:
        url = COMMODITY_URL.format(id=commodity_id)
        page = self._get(url)
        if day is not None:
            form = dict(re.findall(r'name="(__[A-Z]+)"[^>]*value="([^"]*)"', page))
            form["ctl00$cphPage$DateTextBox"] = day.strftime("%m/%d/%Y")
            form["ctl00$cphPage$ReminderButton"] = "Show prices"
            page = self._get(url, form)
        dated = re.search(r"Dated:(\d{2})-(\d{2})-(\d{4})", page)
        if not dated:
            raise ValueError(f"No dated price table for commodity {commodity_id}.")
        on = date(int(dated.group(3)), int(dated.group(2)), int(dated.group(1)))
        markets = {}
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", page, re.S):
            cells = _cells(row)
            if len(cells) != 6 or not re.match(r"\d+\s", cells[0]):
                continue
            low, high, fair = (_price(v) for v in cells[2:5])
            if low and high and fair and 0 < low <= fair <= high:
                markets[re.sub(r"^\d+\s+", "", cells[0])] = {"min": low, "max": high, "fqp": fair}
        return on, markets

    def latest(
        self, commodity_id: int, today: date, max_days: int, min_markets: int
    ) -> tuple[date, dict] | None:
        """Most recent day on which at least min_markets markets traded.

        Off-season, one market may still quote a stray price; such days are
        skipped so the reference reflects a real trading day. Searches daily for
        two weeks, then weekly (refining the week), then falls back to the most
        recent day with any trade.
        """
        fallback = None

        def check(back: int) -> tuple[date, dict] | None:
            nonlocal fallback
            day = None if back == 0 else today - timedelta(days=back)
            on, markets = self.prices(commodity_id, day)
            if markets and fallback is None:
                fallback = (on, markets)
            return (on, markets) if len(markets) >= min_markets else None

        for back in range(0, 15):
            if found := check(back):
                return found
        for back in range(21, max_days + 1, 7):
            if found := check(back):
                # A later day in the skipped week may also qualify.
                for later in range(back - 6, back):
                    if earlier_week := check(later):
                        return earlier_week
                return found
        return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "configs/official_prices.json")
    parser.add_argument("--recent-days", type=int, default=3, help="Look-back for non-core commodities")
    parser.add_argument("--core-days", type=int, default=200, help="Look-back for core crops")
    parser.add_argument("--core-min-markets", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    amis = Amis(args.delay)
    today = datetime.now(PKT).date()
    snapshot = {}
    for name, commodity_id in sorted(amis.commodities().items(), key=lambda item: item[1]):
        core = name in CORE or name.startswith(CORE_PREFIXES)
        if core:
            found = amis.latest(commodity_id, today, args.core_days, args.core_min_markets)
        else:
            found = None
            for back in range(args.recent_days):
                on, markets = amis.prices(commodity_id, None if back == 0 else today - timedelta(days=back))
                if markets:
                    found = (on, markets)
                    break
        if found:
            on, markets = found
            snapshot[name] = {
                "amis_id": commodity_id,
                "unit": unit_of(name),
                "observed_on": on.isoformat(),
                "markets": dict(sorted(markets.items())),
            }
        print(f"{name}: {found[0] if found else 'no recent trade'}", flush=True)

    document = {
        "source": "AMIS Punjab (Agriculture Marketing Information Service, Government of the Punjab)",
        "source_url": BASE,
        "price_basis": "Wholesale mandi prices in PKR: minimum, maximum and fair-quality price (FQP)",
        "fetched_at_pkt": datetime.now(PKT).isoformat(timespec="seconds"),
        "commodities": snapshot,
    }
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"saved {len(snapshot)} commodities to {args.output}")


if __name__ == "__main__":
    main()
