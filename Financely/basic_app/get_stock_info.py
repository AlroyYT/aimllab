import urllib.request
import json
import time
import random
from django.core.cache import cache
import logging
import urllib.parse

logger = logging.getLogger(__name__)

# List of common user agents to rotate through
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
]

def getStockInfo(var):
    """
    Get stock information from Yahoo Finance with caching and retry mechanism to handle rate limits.
    """
    # Clean up input and validate
    if not var:
        return []
    
    var = var.replace(' ', '')
    if not var:
        return []
        
    # Check cache first
    cache_key = f"stock_info_{var}"
    cached_result = cache.get(cache_key)
    if cached_result:
        logger.info(f"Cache hit for {var}")
        return cached_result
    
    # Prepare URL - using an updated Yahoo Finance API endpoint
    # This endpoint is more reliable for getting stock search results
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(var)}&quotesCount=6&newsCount=0"
    
    # Add retry mechanism with exponential backoff
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Create request with random user agent
            req = urllib.request.Request(url)
            req.add_header('User-Agent', random.choice(USER_AGENTS))
            req.add_header('Accept', 'application/json')
            
            # Make the request
            logger.info(f"Fetching stock info for {var} (attempt {attempt+1}/{max_retries})")
            response = urllib.request.urlopen(req, timeout=10)
            
            # Parse response
            data = json.loads(response.read())
            
            # Transform the data to match your expected format
            result = []
            if 'quotes' in data and len(data['quotes']) > 0:
                for quote in data['quotes']:
                    # Only include stocks, ETFs, etc. (not currencies, etc.)
                    if 'symbol' in quote and 'shortname' in quote:
                        result.append({
                            'symbol': quote.get('symbol', ''),
                            'name': quote.get('shortname', ''),
                            'exch': quote.get('exchange', ''),
                            'type': quote.get('quoteType', ''),
                            'exchDisp': quote.get('exchDisp', ''),
                            'typeDisp': quote.get('typeDisp', '')
                        })
            
            # Cache the result for 30 minutes (1800 seconds)
            cache.set(cache_key, result, 1800)
            logger.info(f"Successfully fetched and cached stock info for {var}")
            
            return result
            
        except urllib.error.HTTPError as e:
            if e.code == 429:  # Too Many Requests
                logger.warning(f"Rate limit exceeded (attempt {attempt+1}/{max_retries}). Waiting for {retry_delay} seconds.")
                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch stock info for '{var}' after {max_retries} attempts due to rate limiting.")
                    return []
            else:
                logger.error(f"HTTP Error {e.code}: {e.reason} when fetching stock info for '{var}'")
                return []
        except Exception as e:
            logger.error(f"Error fetching stock info for '{var}': {str(e)}")
            return []