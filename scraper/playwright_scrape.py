import asyncio
import json
import os
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

from .detail_workers import run_detail_workers


CATALOG_URL = "https://bankofgeorgia.ge/ka/offers-hub/catalog"


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
        await page.mouse.wheel(0, 1200)
        await page.wait_for_timeout(500)


def with_current_page(url: str, page_num: int) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["currentPage"] = str(page_num)
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query, doseq=True), parts.fragment)
    )


async def get_total_pages(page) -> int:
    pagination = page.locator("bd-pagination").first
    await pagination.wait_for(timeout=15000)
    total_count = int((await pagination.get_attribute("total-count")) or "0")
    page_size = int((await pagination.get_attribute("page-size")) or "10")
    return (total_count + page_size - 1) // page_size


async def scrape_one_page(page, base_url: str, page_num: int) -> list[dict]:
    await page.goto(with_current_page(base_url, page_num), wait_until="domcontentloaded")
    await page.wait_for_timeout(700)

    await scroll_a_bit(page)
    await page.evaluate("window.scrollTo(0, 0)")
    await page.wait_for_timeout(300)

    cards_locator = page.locator("bd-new-ecommerce-card")
    await cards_locator.first.wait_for(timeout=30000)

    return await cards_locator.evaluate_all(
        """
        cards => cards.map(card => ({
            image_url: card.getAttribute('image-url') || '',
            href: card.getAttribute('href') || '',
            benef_name: card.getAttribute('benefname') || '',
            benef_badge: card.getAttribute('benefbadge') || '',
            brand_name: card.getAttribute('brandname') || '',
            segment_type: card.getAttribute('segmenttype') || '',
            product_code: card.getAttribute('productcode') || '',
            category_id: card.getAttribute('categoryid') || '',
            category_desc: card.getAttribute('categorydesc') || '',
            section_types: card.getAttribute('sectiontypes') || '',
            start_date: card.getAttribute('startdate') || '',
            end_date: card.getAttribute('enddate') || '',
            title: card.querySelector('bd-font[slot="title"]')?.textContent.trim() || ''
        }))
        """
    )


async def worker(browser, storage_state_path: str, base_url: str, page_numbers: list[int], worker_id: int) -> dict[int, list[dict]]:
    context = await browser.new_context(storage_state=storage_state_path)
    page = await context.new_page()
    await page.set_viewport_size({"width": 1280, "height": 720})

    results: dict[int, list[dict]] = {}
    for page_num in page_numbers:
        try:
            results[page_num] = await scrape_one_page(page, base_url, page_num)
            print(f"Worker {worker_id}: page {page_num} -> {len(results[page_num])} cards")
        except PlaywrightTimeoutError:
            print(f"Worker {worker_id}: page {page_num} -> ERROR (no cards)")
        except Exception as e:
            print(f"Worker {worker_id}: page {page_num} -> ERROR ({type(e).__name__}: {e})")

    await context.close()
    return results


async def main() -> None:
    concurrency = int(os.getenv("CONCURRENCY", "1"))
    detail_concurrency = int(os.getenv("DETAIL_CONCURRENCY", "4"))
    headless = os.getenv("HEADLESS", "0") == "1"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)

        # ertjeradi tema - cookieebis dadastureba da pagination-is naxva
        bootstrap_context = await browser.new_context()
        bootstrap_page = await bootstrap_context.new_page()
        await bootstrap_page.set_viewport_size({"width": 1280, "height": 720})

        print("Loading page...")
        await bootstrap_page.goto(CATALOG_URL, wait_until="domcontentloaded")
        await bootstrap_page.wait_for_timeout(1500)

        print("Clicking cookie button...")
        if await accept_cookies(bootstrap_page):
            print("Cookie accepted")
        else:
            print("No cookie dialog (or already accepted)")
        await bootstrap_page.wait_for_timeout(1000)

        print("Scrolling to trigger lazy loading...")
        await scroll_a_bit(bootstrap_page)
        await bootstrap_page.evaluate("window.scrollTo(0, 0)")
        await bootstrap_page.wait_for_timeout(300)

        try:
            await bootstrap_page.locator("bd-new-ecommerce-card").first.wait_for(timeout=30000)
        except PlaywrightTimeoutError:
            debug_path = os.path.join(os.path.dirname(__file__), "debug_no_cards.html")
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(await bootstrap_page.content())
            await bootstrap_context.close()
            await browser.close()
            raise RuntimeError(f"No cards found. Saved debug HTML to: {debug_path}")

        print("Getting pagination...")
        total_pages = await get_total_pages(bootstrap_page)
        print(f"Total pages: {total_pages}")

        base_url = bootstrap_page.url

        storage_state_path = os.path.join(os.path.dirname(__file__), ".storage_state.json")
        await bootstrap_context.storage_state(path=storage_state_path)
        await bootstrap_context.close()

        # workerebis gadanacileba - k,2k... k+1, 2(k+1)...
        concurrency = max(1, min(concurrency, total_pages))
        page_lists = [list(range(i + 1, total_pages + 1, concurrency)) for i in range(concurrency)]
        print(f"Starting {concurrency} workers")

        tasks = [
            worker(browser, storage_state_path, base_url, page_lists[i], i + 1)
            for i in range(concurrency)
        ]
        worker_results = await asyncio.gather(*tasks)

        # resultebis megingi 
        results_by_page: dict[int, list[dict]] = {}
        for d in worker_results:
            results_by_page.update(d)
        all_offers: list[dict] = []
        for page_num in sorted(results_by_page.keys()):
            all_offers.extend(results_by_page[page_num])

        hrefs = [o.get("href", "") for o in all_offers if o.get("href")]
        print(
            f"\nFetching descriptions for {len(hrefs)} offers using {min(4, max(1, detail_concurrency))} workers..."
        )

        try:
            details_by_href = await run_detail_workers(
                browser,
                storage_state_path=storage_state_path,
                hrefs=hrefs,
                concurrency=detail_concurrency,
            )
        except Exception as e:
            print(f"Description enrichment failed: {type(e).__name__}: {e}")
            details_by_href = {}

        for o in all_offers:
            href = o.get("href")
            if not href:
                o["description"] = ""
                o["address"] = ""
                continue
            d = details_by_href.get(href) or {}
            o["description"] = d.get("description", "")
            o["details_url"] = d.get("details_url", "")
            o["address"] = d.get("address", "")

        output_file = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "found_offers.json")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_offers, f, ensure_ascii=False, indent=2)

        await browser.close()

    print(f"\n{'='*50}")
    print("✓ DONE!")
    print(f"{'='*50}")
    print(f"Total offers scraped: {len(all_offers)}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
