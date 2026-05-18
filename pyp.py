import asyncio
import aiohttp
from datetime import datetime, timedelta

BASE = "https://aapt.org/physicsteam/2026/upload/2026-USAPhO-Qualifiers_{}.pdf"

start = datetime(2026, 4, 1)
end   = datetime(2026, 12, 31)

headers = {
    "User-Agent": "Mozilla/5.0"
}

async def check(session, url):
    try:
        async with session.head(url, timeout=5) as r:
            if r.status == 200:
                print("FOUND:", url)
                return url
            else:
                print("Checked:", url, "->", r.status)
    except:
        pass

async def main():
    urls = []
    current = start
    while current <= end:
        date_str = current.strftime("%m%d%Y")
        urls.append(BASE.format(date_str))
        current += timedelta(days=1)

    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [check(session, u) for u in urls]
        await asyncio.gather(*tasks)

asyncio.run(main())