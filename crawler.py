import sys
import asyncio

# Set Windows Proactor event loop for subprocess support
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

async def fetch_markdown(url: str) -> str:
    browser_conf = BrowserConfig(headless=True)
    run_conf = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    async with AsyncWebCrawler(config=browser_conf) as crawler:
        result = await crawler.arun(url=url, config=run_conf)
        return result.markdown

def get_markdown(url: str) -> str:
    """Synchronous wrapper to fetch markdown data from the URL."""
    return asyncio.run(fetch_markdown(url))
