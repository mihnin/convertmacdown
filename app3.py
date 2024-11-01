import streamlit as st
import asyncio
import sys
from crawl4ai import AsyncWebCrawler

# Set the event loop policy to WindowsProactorEventLoopPolicy on Windows
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

async def crawl_url(url):
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(url=url)
    return result

def crawl_url_sync(url):
    # Create a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    # Run the async function in the new event loop
    return loop.run_until_complete(crawl_url(url))

st.title("Async Web Crawler")

# Get URL from user input
url = st.text_input("Enter the URL you want to crawl:")

# Input for the output filename
filename = st.text_input("Enter the output filename (e.g., output.md):", value="output.md")

# Button to start the crawling process
if st.button("Process"):
    if url:
        # Perform web crawling with a loading spinner
        with st.spinner("Processing..."):
            # Run the synchronous wrapper function
            result = crawl_url_sync(url)
        
        # Display the result as Markdown
        st.markdown(result.markdown)
        
        # Provide a download button to save the result
        st.download_button(
            label="Download Markdown",
            data=result.markdown,
            file_name=filename,
            mime="text/markdown"
        )
    else:
        st.error("Please enter a valid URL.")
