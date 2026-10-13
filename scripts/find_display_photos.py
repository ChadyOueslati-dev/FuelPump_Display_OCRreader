"""Find freely licensed photos of fuel pump displays on the internet.

Step 1, collect candidates (small previews only) for someone to look through:

    python -X utf8 scripts/find_display_photos.py search --work C:/ocr-train/internet_candidates

Step 2, after choosing which candidates really show a display with the
liters pumped, download those in full:

    python -X utf8 scripts/find_display_photos.py download --work C:/ocr-train/internet_candidates --keep 3 17 42 --out data/internet_displays

Sources are Openverse (a search engine for Creative Commons pictures, which
covers Flickr and many others) and Wikimedia Commons. Every picture they
list carries a free licence. Stock photo sites are not used: their pictures
are copyrighted.

Step 2 writes sources.csv (file, page, licence, author): the attribution the
licences require. The pictures come without labels.
"""

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

AGENT = "PumpDisplayOCR-testing/0.2 (non-commercial research; small one-off download)"
PAUSE = 2.5  # seconds between requests, to stay well inside the rate limits

# Short queries work best: both sites match every word. People photograph pump
# displays mostly to show what a fill-up cost, so that is what to search for.
QUERIES = [
    # Saudi Arabia and the Gulf first
    "Saudi petrol pump", "Saudi gas station", "Riyadh petrol", "Riyadh gas station", "Jeddah gas station",
    "Aramco station", "Saudi Arabia fuel", "Kuwait petrol pump", "Kuwait petrol", "Dubai petrol pump",
    "UAE petrol pump", "Qatar petrol pump", "Oman petrol pump", "Bahrain petrol", "petrol dirhams litres",
    "محطة بنزين", "مضخة بنزين",
    # displays showing liters, anywhere
    "petrol pump display", "fuel pump display", "gas pump display", "fuel dispenser display",
    "petrol pump litres", "fuel pump litres", "petrol pump liters", "diesel pump litres",
    "petrol pump price litres", "petrol pump total", "petrol pump reading", "fuel pump meter",
    "pump display litres", "filling up petrol litres", "fill up cost petrol pump", "expensive petrol pump",
    "petrol price pump display", "fuel price pump litres", "full tank litres", "litres sale pump",
    "bowser litres", "servo pump litres", "petrol pump close up",
    "Zapfsäule", "Zapfsäule Anzeige", "Zapfsäule Liter", "tanken Liter Anzeige", "volltanken",
    "pompe essence litres", "pompe à essence prix", "surtidor gasolina litros", "gasolinera surtidor",
    "benzinepomp", "tanken liters", "pompa bensin liter", "distributore benzina litri",
    "bensinpump liter", "benzin liter", "pompa paliwo litry", "bomba gasolina litros",
]


def get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    time.sleep(PAUSE)
    with urllib.request.urlopen(request, timeout=40) as response:
        return json.loads(response.read().decode("utf-8"))


def get_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    for attempt in range(4):
        time.sleep(1.0 + 4.0 * attempt)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except ConnectionError:  # Flickr drops a connection now and then; an HTTP refusal is not retried
            if attempt == 3:
                raise
        except urllib.error.URLError as e:
            if not isinstance(e.reason, ConnectionError) or attempt == 3:
                raise
    raise AssertionError("unreachable")


def openverse(query: str, page: int = 1) -> list[dict]:
    url = "https://api.openverse.org/v1/images/?" + urllib.parse.urlencode(
        {"q": query, "page_size": 20, "page": page}
    )
    items = []
    for r in get_json(url).get("results", []):
        items.append({
            "key": r.get("foreign_landing_url") or r["url"],
            "title": (r.get("title") or "")[:100],
            "licence": f"CC {r.get('license', '').upper()} {r.get('license_version') or ''}".strip(),
            "author": (r.get("creator") or "")[:100],
            "page": r.get("foreign_landing_url") or "",
            "full": r["url"],
            # Openverse's own thumbnails are slow; Flickr serves its pictures directly
            "preview": r["url"] if r.get("source") == "flickr" else r.get("thumbnail") or r["url"],
            "source": "openverse/" + (r.get("source") or ""),
        })
    return items


def commons(query: str, page: int = 1) -> list[dict]:
    if page > 1:
        return []  # one page of 30 is all that is asked of Commons
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "generator": "search", "gsrsearch": f"{query} filetype:bitmap",
        "gsrnamespace": 6, "gsrlimit": 30, "prop": "imageinfo",
        "iiprop": "url|extmetadata", "iiurlwidth": 330, "format": "json",  # 330 and 1920: sizes Commons serves freely
    })
    items = []
    for page in get_json(url).get("query", {}).get("pages", {}).values():
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        strip = lambda t: re.sub(r"<[^>]+>", "", t or "").strip()  # noqa: E731
        items.append({
            "key": info.get("descriptionurl", page["title"]),
            "title": page["title"][5:105],
            "licence": strip(meta.get("LicenseShortName", {}).get("value", "")),
            "author": strip(meta.get("Artist", {}).get("value", ""))[:100],
            "page": info.get("descriptionurl", ""),
            "full": re.sub(r"/(\d+)px-", "/1920px-", info.get("thumburl", "")) or info.get("url", ""),
            "original": info.get("url", ""),
            "preview": info.get("thumburl") or info.get("url", ""),
            "source": "commons",
        })
    return items


def search(work: Path, pages: int, resume: bool = False) -> int:
    """Adds to what an earlier search already collected in the same folder."""
    (work / "previews").mkdir(parents=True, exist_ok=True)
    listing = work / "candidates.json"
    candidates = json.loads(listing.read_text(encoding="utf-8")) if listing.is_file() else []
    known = {c["key"] for c in candidates}
    found: dict[str, dict] = {}
    for query in [] if resume else QUERIES:
        for name, source in (("openverse", openverse), ("commons", commons)):
            for page in range(1, pages + 1):
                try:
                    items = source(query, page)
                except Exception as e:  # rate limit, last page, or network: report and move on
                    print(f"{name} stopped [{query}] page {page}: {e}", flush=True)
                    break
                fresh = [i for i in items if i["key"] not in found and i["key"] not in known]
                for item in fresh:
                    found[item["key"]] = {**item, "query": query}
                if items:
                    print(f"{name} [{query}] page {page}: {len(items)} result(s), {len(fresh)} new", flush=True)
                if len(items) < 20:
                    break

    # The search results are kept, so an interrupted run carries on from here.
    waiting = work / "pending.json"
    pending = json.loads(waiting.read_text(encoding="utf-8")) if waiting.is_file() else []
    pending += [i for i in found.values() if i["key"] not in {p["key"] for p in pending}]
    pending = [i for i in pending if i["key"] not in known]
    pending.sort(key=lambda i: i["source"] != "openverse/flickr")  # Flickr has most of the close-ups
    waiting.write_text(json.dumps(pending, ensure_ascii=False), encoding="utf-8")
    refused = 0
    for item in pending:
        number = len(candidates)
        if item["source"] == "commons":
            if refused >= 5:
                continue  # Commons is rate limiting: leave its pictures for a later run
            time.sleep(2.0)
        try:
            (work / "previews" / f"{number:04d}.jpg").write_bytes(get_bytes(item["preview"]))
        except Exception as e:
            print(f"preview failed [{item['title']}]: {e}", flush=True)
            refused += item["source"] == "commons" and "429" in str(e)
            continue
        candidates.append({**item, "number": number})
        listing.write_text(json.dumps(candidates, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(candidates)} candidate preview(s) in {work / 'previews'}")
    return 0


def download(work: Path, keep: list[int], out: Path) -> int:
    candidates = {c["number"]: c for c in json.loads((work / "candidates.json").read_text(encoding="utf-8"))}
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for number in keep:
        item = candidates[number]
        name = f"display_{number:04d}.jpg"
        if (out / name).is_file():  # saved by an earlier run: keep it and its row in sources.csv
            rows.append([name, item["page"], item["licence"], item["author"], item["title"]])
            continue
        full, original = item["full"], item.get("original", "")
        in_commons = re.search(r"/commons/thumb/(.+?)/\d+px-", full)
        if in_commons:
            full = re.sub(r"/\d+px-", "/1920px-", full)
            original = "https://upload.wikimedia.org/wikipedia/commons/" + in_commons.group(1)
        try:
            (out / name).write_bytes(get_bytes(full))
        except Exception as e:
            # Commons has no 1920 px version of a picture smaller than that
            try:
                (out / name).write_bytes(get_bytes(original))
            except Exception:
                print(f"download failed [{number} {item['title']}]: {e}", flush=True)
                continue
        rows.append([name, item["page"], item["licence"], item["author"], item["title"]])
        print(f"saved {name}  ({item['licence']})  {item['title'][:60]}", flush=True)
    with open(out / "sources.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "source_page", "licence", "author", "title"])
        writer.writerows(rows)
    print(f"{len(rows)} picture(s) saved to {out}, with sources.csv")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    s = commands.add_parser("search")
    s.add_argument("--work", type=Path, required=True)
    s.add_argument("--pages", type=int, default=2, help="result pages to fetch per query from Openverse")
    s.add_argument("--resume", action="store_true", help="skip the search; fetch previews an earlier run left")
    d = commands.add_parser("download")
    d.add_argument("--work", type=Path, required=True)
    d.add_argument("--keep", type=int, nargs="+", required=True)
    d.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "search":
        return search(args.work, args.pages, args.resume)
    return download(args.work, args.keep, args.out)


if __name__ == "__main__":
    sys.exit(main())
