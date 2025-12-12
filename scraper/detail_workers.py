from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urljoin

from playwright.async_api import TimeoutError as PlaywrightTimeoutError


BASE_URL = "https://bankofgeorgia.ge"


@dataclass(frozen=True)
class DetailSelectors:
    description_candidates: tuple[str, ...] = (
        "[data-testid='description']",
        "[data-testid='offer-description']",
        "[data-testid='offer-details']",
        "main [class*='description']",
        "main [class*='content']",
        ".offer-description",
        ".offer-details",
        "div[class*='description']",
        "bd-font[type='text']",
        "bd-font[type='small-text']",
    )

    reject_exact: tuple[str, ...] = (
        "მისამართები",
        "პირობები",
        "წესები",
        "დეტალები",
    )


def normalize_href(href: str, base_url: str = BASE_URL) -> str:
    if not href:
        return href
    return urljoin(base_url, href)


async def _extract_description_from_dom(page, selectors: DetailSelectors) -> str:
    for sel in selectors.description_candidates:
        loc = page.locator(sel).first
        try:
            await loc.wait_for(state="attached", timeout=2500)
            text = (await loc.inner_text()).strip()
            if text:
                normalized = " ".join(text.split())
                if normalized in selectors.reject_exact:
                    continue
                if len(normalized) >= 80:
                    return normalized
        except PlaywrightTimeoutError:
            continue
        except Exception:
            continue

    try:
        data = await page.evaluate(
            r"""
            () => {
              const norm = (s) => (s || '').replace(/\s+/g,' ').trim();
              const main = document.querySelector('main') || document.body;
              const allText = norm(main.innerText || '');
              if (!allText) return { allText: '' };

              const markers = [
                '\nმისამართები\n',
                '\nმისამართი\n',
                '\nპირობები\n',
                '\nწესები\n',
              ];

              const raw = (main.innerText || '');
              const idxs = markers
                .map(m => ({ m, i: raw.indexOf(m) }))
                .filter(x => x.i !== -1)
                .sort((a,b) => a.i - b.i);

              let before = raw;
              if (idxs.length) before = raw.slice(0, idxs[0].i);

              let lines = before.split(/\r?\n/).map(norm).filter(Boolean);
              lines = lines.filter(l => l.length >= 12);
              if (lines.length >= 2 && lines[0].length < 40) {
                lines = lines.slice(1);
              }
                            const desc = lines.join('\n').replace(/\n{3,}/g, '\n\n').trim();

              return { desc };
            }
            """
        )
        if isinstance(data, dict):
            desc = (data.get("desc") or "").strip()
            if desc and desc not in selectors.reject_exact and len(desc) >= 80:
                return desc
    except Exception:
        pass

    try:
        meta = page.locator("head meta[name='description']").first
        content = (await meta.get_attribute("content")) or ""
        content = content.strip()
        if content and content not in selectors.reject_exact:
            return content
    except Exception:
        pass

    return ""


async def _extract_address_from_dom(page) -> str:
        try:
                data = await page.evaluate(
                        r"""
                        () => {
                            const norm = (s) => (s || '').replace(/\s+/g,' ').trim();
                            const main = document.querySelector('main') || document.body;
                            const raw = main.innerText || '';
                            const markers = ['\nმისამართი\n', '\nმისამართები\n'];
                            for (const m of markers) {
                                const i = raw.indexOf(m);
                                if (i !== -1) {
                                    const after = raw.slice(i + m.length);
                                    const lines = after.split(/\r?\n/).map(norm).filter(Boolean);
                                    return { address: lines[0] || '' };
                                }
                            }
                            return { address: '' };
                        }
                        """
                )
                if isinstance(data, dict):
                        return (data.get("address") or "").strip()
        except Exception:
                pass
        return ""


async def scrape_one_detail_page(page, href: str, selectors: Optional[DetailSelectors] = None) -> Dict[str, Any]:
    selectors = selectors or DetailSelectors()
    url = normalize_href(href)

    try:
        await page.goto(url, wait_until="domcontentloaded")

        # Detail pages are dynamic; wait for main body to be present and stabilize.
        try:
            await page.locator("main").first.wait_for(state="attached", timeout=15000)
        except PlaywrightTimeoutError:
            # Some pages might not have <main>; fall back to body.
            await page.locator("body").first.wait_for(state="attached", timeout=15000)

        # Give the SPA a moment to render text.
        await page.wait_for_timeout(900)

        # Trigger lazy sections.
        for _ in range(3):
            try:
                await page.mouse.wheel(0, 1200)
                await page.wait_for_timeout(350)
            except Exception:
                break
        try:
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(250)
        except Exception:
            pass

        # Now that dynamic content should be in the DOM, extract fields.
        description = await _extract_description_from_dom(page, selectors)
        address = await _extract_address_from_dom(page)
        return {
            "href": href,
            "details_url": url,
            "description": description,
            "address": address,
            "status": "ok",
        }
    except PlaywrightTimeoutError as e:
        return {"href": href, "details_url": url, "description": "", "status": "error", "error": str(e)}
    except Exception as e:
        return {"href": href, "details_url": url, "description": "", "status": "error", "error": f"{type(e).__name__}: {e}"}


async def detail_worker(browser, storage_state_path: str, hrefs: List[str], worker_id: int) -> Dict[str, Dict[str, Any]]:
    context = await browser.new_context(storage_state=storage_state_path)
    page = await context.new_page()
    await page.set_viewport_size({"width": 1280, "height": 720})

    out: Dict[str, Dict[str, Any]] = {}
    selectors = DetailSelectors()

    for href in hrefs:
        res = await scrape_one_detail_page(page, href, selectors=selectors)
        out[href] = {**res, "worker": worker_id}

    await context.close()
    return out


def _split_round_robin(items: List[str], buckets: int) -> List[List[str]]:
    buckets = max(1, buckets)
    out = [[] for _ in range(buckets)]
    for i, item in enumerate(items):
        out[i % buckets].append(item)
    return out


async def run_detail_workers(browser, storage_state_path: str, hrefs: Iterable[str], concurrency: int = 5) -> Dict[str, Dict[str, Any]]:
    href_list = [h for h in hrefs if h]
    if not href_list:
        return {}

    concurrency = max(1, min(concurrency, 8, len(href_list)))
    href_lists = _split_round_robin(href_list, concurrency)

    tasks = [
        detail_worker(browser, storage_state_path, href_lists[i], i + 1)
        for i in range(concurrency)
    ]
    results = await asyncio.gather(*tasks)

    merged: Dict[str, Dict[str, Any]] = {}
    for d in results:
        merged.update(d)
    return merged
