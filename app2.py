import asyncio
from crawl4ai import AsyncWebCrawler

async def main():
    # Get URL and filename from user input
    url = input("Enter the URL you want to crawl: ")
    filename = input("Enter the output filename (with extension, e.g., output.md): ")
    
    # Perform web crawling
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(url=url)
    
    # Save result to the specified file
    with open(filename, 'w') as file:
        file.write(result.markdown)
    
    print(f"Content successfully saved to {filename}")

if __name__ == "__main__":
    asyncio.run(main())