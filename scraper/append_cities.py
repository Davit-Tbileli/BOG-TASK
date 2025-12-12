import asyncio
import json
import os
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright


BASE_URL = "https://bankofgeorgia.ge"
CATALOG_URL = f"{BASE_URL}/ka/offers-hub/catalog"


def normalize_href(href: str) -> str:
    if not href:
        return href
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if not href.startswith("/"):
        href = "/" + href
    return BASE_URL + href


def with_current_page(url: str, page_num: int) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["currentPage"] = str(page_num)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query, doseq=True), parts.fragment))


def with_city(url: str, city_id: int) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["cityId"] = str(city_id)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query, doseq=True), parts.fragment))


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


async def get_total_pages(page) -> int:
    pagination = page.locator("bd-pagination").first
    try:
        await pagination.wait_for(timeout=8000)
        total_count = int((await pagination.get_attribute("total-count")) or "0")
        page_size = int((await pagination.get_attribute("page-size")) or "10")
        pages = (total_count + page_size - 1) // page_size
        return max(1, pages)
    except Exception:
        # Some city filters may produce 0 results or render without pagination.
        return 1


async def scrape_city_page_hrefs(page, base_url: str, page_num: int) -> list[str]:
    await page.goto(with_current_page(base_url, page_num), wait_until="domcontentloaded")
    await page.wait_for_timeout(700)
    await scroll_a_bit(page)
    await page.evaluate("window.scrollTo(0, 0)")
    await page.wait_for_timeout(250)

    cards = page.locator("bd-new-ecommerce-card")
    try:
        await cards.first.wait_for(timeout=20000)
    except Exception:
        return []

    hrefs = await cards.evaluate_all("""
        cards => cards.map(c => (c.getAttribute('href') || '').trim()).filter(Boolean)
    """)
    return hrefs


async def city_worker(browser, storage_state_path: str, base_url: str, page_numbers: list[int]) -> list[str]:
    context = await browser.new_context(storage_state=storage_state_path)
    page = await context.new_page()
    await page.set_viewport_size({"width": 1280, "height": 720})

    out: list[str] = []
    for pn in page_numbers:
        try:
            out.extend(await scrape_city_page_hrefs(page, base_url, pn))
        except PlaywrightTimeoutError:
            continue
        except Exception:
            continue

    await context.close()
    return out


def add_city(offer: dict, city_label: str, city_id: int) -> None:
    cities = offer.get("cities")
    if not isinstance(cities, list):
        cities = []
    if city_label and city_label not in cities:
        cities.append(city_label)
    offer["cities"] = cities

    city_ids = offer.get("city_ids")
    if not isinstance(city_ids, list):
        city_ids = []
    if isinstance(city_id, int) and city_id not in city_ids:
        city_ids.append(city_id)
    offer["city_ids"] = city_ids


async def main() -> None:
    headless = os.getenv("HEADLESS", "0") == "1"
    concurrency = int(os.getenv("CONCURRENCY", "2"))
    city_concurrency = int(os.getenv("CITY_CONCURRENCY", "3"))
    city_limit = int(os.getenv("CITY_LIMIT", "0"))

    root = os.path.join(os.path.dirname(__file__), "..")
    cities_path = os.path.join(root, "data", "raw", "cities.json")
    offers_in_path = os.path.join(root, "data", "raw", "found_offers.json")

    if not os.path.exists(cities_path):
        raise FileNotFoundError(f"Missing cities.json: {cities_path} (run: py -m scraper.city_scrape)")
    if not os.path.exists(offers_in_path):
        raise FileNotFoundError(f"Missing found_offers.json: {offers_in_path} (run main scraper first)")

    cities = json.load(open(cities_path, "r", encoding="utf-8"))
    offers = json.load(open(offers_in_path, "r", encoding="utf-8"))

    # Index existing offers by normalized href (best stable key available in this dataset)
    offers_by_href: dict[str, dict] = {}
    for o in offers:
        raw = (o.get("href", "") or "").strip()
        href_abs = normalize_href(raw)
        if href_abs:
            offers_by_href[href_abs] = o
        if raw:
            offers_by_href[raw] = o
            # Ensure fields exist
            if "cities" not in o:
                o["cities"] = []
            if "city_ids" not in o:
                o["city_ids"] = []

    if city_limit and city_limit > 0:
        cities = cities[:city_limit]

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)

        browser_closed = False

        def _on_disconnected() -> None:
            nonlocal browser_closed
            browser_closed = True

        browser.on("disconnected", _on_disconnected)

        # Bootstrap once, accept cookies, and cache storage state for all city runs
        bootstrap_context = await browser.new_context()
        bootstrap_page = await bootstrap_context.new_page()
        await bootstrap_page.set_viewport_size({"width": 1280, "height": 720})
        await bootstrap_page.goto(CATALOG_URL, wait_until="domcontentloaded")
        await bootstrap_page.wait_for_timeout(1200)
        await accept_cookies(bootstrap_page)
        await bootstrap_page.wait_for_timeout(600)

        storage_state_path = os.path.join(os.path.dirname(__file__), ".storage_state.json")
        await bootstrap_context.storage_state(path=storage_state_path)
        await bootstrap_context.close()

        async def process_one_city(city: dict, idx: int) -> None:
            if browser_closed:
                return
            city_label = (city.get("label") or "").strip()
            city_id = city.get("value")
            if not isinstance(city_id, int):
                try:
                    city_id = int(str(city_id))
                except Exception:
                    return

            city_url = with_city(CATALOG_URL, city_id)
            print(f"[{idx}/{len(cities)}] City: {city_label} (cityId={city_id})")

            # Determine pages for this city
            ctx = None
            total_pages = 1
            try:
                ctx = await browser.new_context(storage_state=storage_state_path)
                pg = await ctx.new_page()
                await pg.set_viewport_size({"width": 1280, "height": 720})
                await pg.goto(city_url, wait_until="domcontentloaded")
                await pg.wait_for_timeout(800)
                await scroll_a_bit(pg)
                total_pages = await get_total_pages(pg)
            except Exception as e:
                # Any failure for one city shouldn't kill the full job.
                print(f"  city bootstrap failed: {type(e).__name__}: {e}")
                return
            finally:
                if ctx is not None:
                    try:
                        await ctx.close()
                    except Exception:
                        pass

            concurrency_eff = max(1, min(concurrency, total_pages))
            page_lists = [list(range(i + 1, total_pages + 1, concurrency_eff)) for i in range(concurrency_eff)]

            tasks = [
                city_worker(browser, storage_state_path, city_url, page_lists[i])
                for i in range(concurrency_eff)
            ]
            try:
                results = await asyncio.gather(*tasks)
            except Exception as e:
                print(f"  city page workers failed: {type(e).__name__}: {e}")
                return
            # Keep both raw and normalized; main dataset stores raw relative hrefs today.
            hrefs_raw = set(h for sub in results for h in sub if h)
            hrefs = hrefs_raw | set(normalize_href(h) for h in hrefs_raw)

            matched = 0
            for href in hrefs:
                if href in offers_by_href:
                    add_city(offers_by_href[href], city_label, city_id)
                    matched += 1
            print(f"  matched offers: {matched} / hrefs in city: {len(hrefs)}")

        async def run_in_pool() -> None:
            sem = asyncio.Semaphore(max(1, city_concurrency))

            async def wrapped(city: dict, idx: int) -> None:
                async with sem:
                    try:
                        await process_one_city(city, idx)
                    except Exception as e:
                        # Catch-all so one task doesn't cancel others.
                        print(f"  city task crashed: {type(e).__name__}: {e}")

            await asyncio.gather(*[wrapped(city, idx) for idx, city in enumerate(cities, start=1)])

        await run_in_pool()

        await browser.close()

    with open(offers_in_path, "w", encoding="utf-8") as f:
        json.dump(offers, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {offers_in_path}")


if __name__ == "__main__":
    asyncio.run(main())
