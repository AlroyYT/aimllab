import requests
import random
from basic_app.sentiment_analysis import predict_sentiment
def getNews(key):
    r = requests.get(f"https://newsapi.org/v2/everything?q={key}&pageSize=12&apiKey=9b23adeb6a634a0ba1f62e76dcbc54de")
    res = r.json()
    news = {}

    if res['status'] == 'ok':
        articles = res['articles']
        random_news=random.sample(articles, 12)
        for i in range(12):
            #random_news[i]['sentiment'] = predict_sentiment([random_news[i]['description'][:100]])[0]
            news[i]=random_news[i]


    return news

def getNewsWithSentiment(query):
    """
    Get news articles with sentiment analysis for a given query
    
    Args:
        query (str): The search query for news articles
        
    Returns:
        list: List of news articles with sentiment analysis
    """
    try:
        # Import necessary functions - using a direct import path
        from basic_app.get_news import getNews  # Import your existing getNews function
        
        try:
            # Try to import the sentiment module
            from basic_app.sentiment import predict_sentiment
        except ImportError:
            # Create a simple fallback sentiment function if the module doesn't exist
            def predict_sentiment(texts):
                return ["neutral"] * len(texts)
            print("Using fallback neutral sentiment (sentiment module not found)")
        
        # Get raw news
        news = getNews(query)
        
        # Check if news is empty or None
        if not news:
            print(f"Warning: No news found for query '{query}'")
            return [{'title': 'No news available', 'description': 'No news found for this query', 'sentiment': 'neutral'}] * 12
        
        # Process news with sentiment
        random_news = news[:12]  # Take first 12 articles or fewer if less available
        
        # Fill to 12 items if needed
        while len(random_news) < 12:
            random_news.append({'title': 'No additional news', 'description': 'No additional news available', 'sentiment': 'neutral'})
        
        # Add sentiment to each news item
        for i in range(len(random_news)):
            try:
                # Check if the news item is valid
                if random_news[i] is None:
                    random_news[i] = {'title': 'Invalid news item', 'description': 'No content available', 'sentiment': 'neutral'}
                    continue
                
                # Check if description exists and is not None
                description = random_news[i].get('description')
                if description is None or description == '':
                    random_news[i]['description'] = 'No description available'
                    random_news[i]['sentiment'] = 'neutral'
                    continue
                
                # Truncate description to first 100 chars for sentiment analysis
                truncated_description = description[:100]
                
                # Get sentiment
                sentiment_result = predict_sentiment([truncated_description])
                
                # Check if sentiment prediction worked
                if sentiment_result and len(sentiment_result) > 0:
                    random_news[i]['sentiment'] = sentiment_result[0]
                else:
                    random_news[i]['sentiment'] = 'neutral'
                    
            except Exception as e:
                print(f"Error processing news item {i}: {e}")
                # Provide default values for this item
                random_news[i] = random_news[i] if random_news[i] else {}
                random_news[i]['sentiment'] = 'neutral'
            
        return random_news
        
    except Exception as e:
        print(f"Error in getNewsWithSentiment: {e}")
        # Return a default list of empty news items with neutral sentiment
        return [{'title': 'Error fetching news', 'description': 'There was an error fetching news', 'sentiment': 'neutral'}] * 12