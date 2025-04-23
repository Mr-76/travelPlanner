import requests
from bs4 import BeautifulSoup
from langchain_community.utilities import SearxSearchWrapper

# Initialize the SearxSearchWrapper with the Searx instance
searx = SearxSearchWrapper(searx_host="http://localhost:8888")

# Define your query
query = "what is a large language model?"

# Fetch search results, limiting to 5 results
results = searx.results(query, num_results=5)

# Function to fetch full text from a URL
def fetch_full_text(url):
    try:
        # Make a request to the page
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract the text from the page
            text = soup.get_text(separator=' ', strip=True)
            return text
        else:
            return f"Failed to fetch page: {url}"
    except Exception as e:
        return f"Error fetching page {url}: {e}"

# Iterate through results and fetch the full text
for index, result in enumerate(results, 1):
    print(f"Result {index}:")
    title = result.get('title', 'No title')
    link = result.get('link', 'No URL')
    print(f"Title: {title}")
    print(f"Link: {link}")
    
    # Fetch and print the full text from the URL
    full_text = fetch_full_text(link)
    print(f"Full Text: {full_text[:1000]}...")  # Show first 1000 characters of full text as a preview
    print("-" * 50)
