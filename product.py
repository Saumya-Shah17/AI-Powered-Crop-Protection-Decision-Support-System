from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
import requests
import json
import re
import os
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

def is_agriculture_related(query: str) -> bool:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
    )

    prompt = """
    You are an agriculture query validator. Your task is to determine if the user's query is related to agriculture or farming.

    Examples of agriculture-related queries:
    - "wheat seeds"
    - "organic fertilizers"
    - "tractor parts"
    - "crop irrigation systems"

    Examples of non-agriculture-related queries:
    - "smartphones"
    - "laptops"
    - "cars"
    - "movies"

    Return "yes" if the query is related to agriculture, otherwise return "no".

    Query: {query}
    """

    response = llm.invoke(prompt.format(query=query))
    return response.content.lower()

def web_search_product(query: str, num_results=10) -> list:
    """Search for agricultural products in India and return product details in INR."""
    SERPAPI_KEY = os.getenv("SERPAPI_KEY")
    
    # Check if API key is available
    if not SERPAPI_KEY:
        # SERPAPI_KEY missing; return fallback products without noisy terminal output
        # Return fallback products with Google search links
        return [{
            "title": f"{query} - Agricultural Product",
            "price": "Price varies",
            "link": f"https://www.google.com/search?q={query.replace(' ', '+')}+buy+online+india",
            "source": "Google Search",
            "image": "",
            "description": f"Search for {query} products online"
        }]
    
    url = "https://serpapi.com/search"
    params = {
        "q": f"{query} buy online india",
        "api_key": SERPAPI_KEY,
        "num": num_results,
        "tbm": "shop",  # Use Google Shopping for product search
        "gl": "in",  # Restrict results to India
        "hl": "en",  # Language: English
        "currency": "INR"  # Currency: Indian Rupees
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()
        
        # Debug printing removed to avoid dumping large JSON to terminal
        
        # Extract product details from shopping results
        products = []
        shopping_results = data.get("shopping_results", [])
        
        def _normalize_url(possible_url: str) -> str:
            if not possible_url:
                return "#"
            # Ensure scheme exists
            if possible_url.startswith("//"):
                return "https:" + possible_url
            if not possible_url.startswith("http://") and not possible_url.startswith("https://"):
                return "https://" + possible_url
            return possible_url

        def _get_domain(url: str) -> str:
            try:
                return urlparse(url).netloc.lower()
            except Exception:
                return ""

        def _prefer_direct_merchant_link(title: str) -> str:
            """If we only have a Google link, try to find a direct merchant URL via a quick web search restricted to common marketplaces."""
            try:
                # If no API key, we can't do a secondary search
                if not SERPAPI_KEY:
                    return ""
                merchant_query = f"{title} buy online"
                merchant_params = {
                    "q": merchant_query,
                    "api_key": SERPAPI_KEY,
                    "num": 5,
                    "gl": "in",
                    "hl": "en",
                    # No tbm here, use normal organic results
                    "safe": "active"
                }
                merchant_resp = requests.get(url, params=merchant_params)
                merchant_resp.raise_for_status()
                merchant_data = merchant_resp.json()
                preferred_domains = {
                    "www.amazon.in", "amazon.in",
                    "www.flipkart.com", "flipkart.com",
                    "www.indiamart.com", "indiamart.com",
                    "www.moglix.com", "moglix.com",
                    "www.agribegri.com", "agribegri.com",
                    "www.agrostar.in", "agrostar.in",
                    "www.croma.com", "croma.com",
                    "www.reliancedigital.in", "reliancedigital.in"
                }
                for org in merchant_data.get("organic_results", [])[:5]:
                    candidate = org.get("link")
                    if not candidate:
                        continue
                    domain = _get_domain(candidate)
                    if domain in preferred_domains:
                        return _normalize_url(candidate)
                # If none matched, return the first organic link as a last resort
                if merchant_data.get("organic_results"):
                    fallback = merchant_data["organic_results"][0].get("link", "")
                    return _normalize_url(fallback)
            except Exception as _e:
                pass
            return ""

        for result in shopping_results[:num_results]:
            # Prefer direct merchant link when available
            raw_link = (
                result.get("product_link")
                or result.get("link")
                or ""
            )
            product_link = _normalize_url(raw_link)
            # If link still points to Google, try to replace with a direct merchant link
            if _get_domain(product_link).endswith("google.com") or _get_domain(product_link).endswith("google.co.in"):
                direct = _prefer_direct_merchant_link(result.get("title", ""))
                if direct:
                    product_link = direct

            products.append({
                "title": result.get("title", "No title available"),
                "price": result.get("price", "Price not available"),
                "link": product_link,
                "source": result.get("source", "Unknown source"),
                "image": result.get("thumbnail", ""),  # Fetch product image
                "description": result.get("description", "No description available.")
            })
        
        # If no shopping results found, try organic results as fallback
        if not products:
            organic_results = data.get("organic_results", [])
            for result in organic_results[:num_results]:
                raw_link = result.get("product_link") or result.get("link", "")
                if not raw_link or raw_link == "#":
                    product_title = result.get("title", "")
                    if product_title:
                        raw_link = f"https://www.google.com/search?q={product_title.replace(' ', '+')}+buy+online+india"
                    else:
                        raw_link = "#"
                product_link = _normalize_url(raw_link)
                if _get_domain(product_link).endswith("google.com") or _get_domain(product_link).endswith("google.co.in"):
                    direct = _prefer_direct_merchant_link(result.get("title", ""))
                    if direct:
                        product_link = direct
                
                products.append({
                    "title": result.get("title", "No title available"),
                    "price": "Price varies",  # Organic results don't have price
                    "link": product_link,
                    "source": result.get("source", "Unknown source"),
                    "image": result.get("thumbnail", ""),
                    "description": result.get("snippet", "No description available.")
                })
        
        # If still no products found, return fallback products
        if not products:
            return [{
                "title": f"{query} - Agricultural Product",
                "price": "Price varies",
                "link": f"https://www.google.com/search?q={query.replace(' ', '+')}+buy+online+india",
                "source": "Google Search",
                "image": "",
                "description": f"Search for {query} products online"
            }]
        
        return products
    except Exception as e:
        # Suppress verbose errors in terminal; still return graceful fallback
        # Return fallback products with Google search links
        return [{
            "title": f"{query} - Agricultural Product",
            "price": "Price varies", 
            "link": f"https://www.google.com/search?q={query.replace(' ', '+')}+buy+online+india",
            "source": "Google Search",
            "image": "",
            "description": f"Search for {query} products online"
        }]
    
