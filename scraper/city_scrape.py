import asyncio
import csv
import json
import os
import re
from pathlib import Path

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright


CATALOG_URL = "https://bankofgeorgia.ge/ka/offers-hub/catalog"


HERE = Path(__file__).resolve().parent
SNAPSHOT_FILES = [HERE / "location.txt", HERE / "cityid_html.txt"]


def _extract_from_snapshot_text(text: str) -> list[dict]:
    items = []
    for m in re.finditer(r"<bg-bog-filter-item\\b([^>]+)>", text, flags=re.IGNORECASE | re.DOTALL):
        attrs = m.group(1)
        label_m = re.search(r"\\blabel=\"([^\"]+)\"", attrs)
        value_m = re.search(r"\\bvalue=\"([^\"]+)\"", attrs)
        count_m = re.search(r"\\bitems-count=\"([^\"]+)\"", attrs)
        if not label_m or not value_m:
            continue
        label = label_m.group(1).strip()
        value = value_m.group(1).strip()
        items_count = (count_m.group(1).strip() if count_m else "")
        if not label or not value:
            continue
        try:
            value_parsed = int(value)
        except ValueError:
            value_parsed = value
        try:
            items_count_parsed = int(items_count) if items_count else ""
        except ValueError:
            items_count_parsed = items_count
        items.append({"label": label, "value": value_parsed, "items_count": items_count_parsed})

    seen = set()
    out = []
    for r in items:
        key = (r["label"], r["value"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    out.sort(key=lambda d: str(d["label"]))
    return out


def extract_from_snapshots() -> list[dict]:
    for p in SNAPSHOT_FILES:
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if "<bg-bog-filter-item" not in text:
            continue
        cities = _extract_from_snapshot_text(text)
        if cities:
            return cities
    return []


async def accept_cookies(page) -> bool:
    candidates = [
        page.locator("bd-cookie-consent bd-standard-button[type='primary']").first,
        page.get_by_role("button", name="დადასტურება").first,
        page.get_by_role("button", name="Confirm").first,
    ]

    for locator in candidates:
        try:
            await locator.wait_for(state="visible", timeout=5000)
            await locator.click(timeout=5000)
            return True
        except PlaywrightTimeoutError:
            continue
    return False


async def scroll_a_bit(page) -> None:
    for _ in range(3):
        try:
            await page.mouse.wheel(0, 1200)
        except Exception:
            await page.evaluate("window.scrollBy(0, 1200)")
        await page.wait_for_timeout(500)


async def extract_cities(page) -> list[dict]:
    await page.goto(CATALOG_URL, wait_until="domcontentloaded")
    await page.wait_for_timeout(1500)

    await accept_cookies(page)
    await page.wait_for_timeout(800)

    await scroll_a_bit(page)

    # Open city filter accordion (title: "აირჩიე ქალაქი")
    filter_root = page.locator('bg-bog-product-filter-with-search[title="აირჩიე ქალაქი"]').first
    await filter_root.wait_for(timeout=20000)

    # Some pages ship it expanded, but clicking header is safe.
    try:
        await filter_root.locator("bd-accordion bd-list-item.header").first.click(timeout=3000)
    except Exception:
        pass

    # Ensure items appear
    items = filter_root.locator("bg-bog-filter-item")
    await items.first.wait_for(timeout=20000)

    # Pull attributes directly off the custom element.
    rows = await items.evaluate_all(
        """
        nodes => nodes.map(n => ({
            label: (n.getAttribute('label') || '').trim(),
            value: (n.getAttribute('value') || '').trim(),
            items_count: (n.getAttribute('items-count') || '').trim(),
        }))
        """
    )

    out = []
    seen = set()
    for r in rows:
        label = (r.get("label") or "").strip()
        value = (r.get("value") or "").strip()
        items_count = (r.get("items_count") or "").strip()

        if not label or not value:
            continue

        try:
            value_parsed = int(value)
        except ValueError:
            value_parsed = value

        try:
            items_count_parsed = int(items_count) if items_count else ""
        except ValueError:
            items_count_parsed = items_count

        key = (label, value)
        if key in seen:
            continue
        seen.add(key)

        out.append(
            {
                "label": label,
                "value": value_parsed,
                "items_count": items_count_parsed,
            }
        )

    out.sort(key=lambda d: (str(d["label"])))
    return out


async def main() -> None:
    headless = os.getenv("HEADLESS", "0") == "1"
    only_snapshot = os.getenv("CITY_SCRAPE_SNAPSHOT_ONLY", "0") == "1"

    cities: list[dict] = []

    if not only_snapshot:
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=headless)
                context = await browser.new_context()
                page = await context.new_page()
                await page.set_viewport_size({"width": 1280, "height": 720})

                print("Loading catalog and extracting city filter...")
                cities = await extract_cities(page)

                await context.close()
                await browser.close()
        except Exception as e:
            print(f"Playwright city extraction failed ({type(e).__name__}: {e}). Falling back to snapshots...")

    if not cities:
        cities = extract_from_snapshots()
        if cities:
            print(f"Loaded {len(cities)} cities from local snapshot file")
        else:
            raise RuntimeError("Could not extract cities (neither live scrape nor snapshot parse succeeded)")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "cities.json")
    csv_path = os.path.join(out_dir, "cities.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cities, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "value", "items_count"])
        w.writeheader()
        w.writerows(cities)

    print(f"Extracted {len(cities)} cities")
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
