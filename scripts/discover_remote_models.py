import asyncio
import aiohttp

ENDPOINTS = ["http://localhost:1234", "http://sams-macbook-pro:1234"]


async def discover_models():
    """Queries the /v1/models endpoint on all specified hosts."""
    print("--- Discovering Models on All Endpoints ---")
    async with aiohttp.ClientSession() as session:
        for endpoint in ENDPOINTS:
            try:
                url = f"{endpoint}/v1/models"
                print(f"\nQuerying: {url}...")
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("data", [])
                        if models:
                            print(f"  ✅ Success! Found {len(models)} model(s):")
                            for model in models:
                                print(f"    - {model.get('id')}")
                        else:
                            print("  🟡 Success, but no models found.")
                    else:
                        print(f"  ❌ Failed with status: {response.status}")
            except Exception as e:
                print(f"  ❌ Failed to connect: {e}")


if __name__ == "__main__":
    asyncio.run(discover_models())
