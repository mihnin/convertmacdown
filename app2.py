import asyncio
from crawl4ai import AsyncWebCrawler

async def main():
    try:
        # Get URL and filename from user input
        url = input("Enter the URL you want to crawl: ")
        filename = input("Enter the output filename (with extension, e.g., output.md): ")
        
        # Perform web crawling
        async with AsyncWebCrawler(verbose=True) as crawler:
            result = await crawler.arun(url=url)
        
        # Save result to the specified file with UTF-8 encoding
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(result.markdown)
        
        print(f"Content successfully saved to {filename}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())