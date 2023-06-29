
import json
import requests
import datetime
import pandas as pd
import numpy as np
import time
from helpers import gpt3_completion, gpt4_completion, obtain_article_text




#functions

def convert_unix_to_datetime(unix_timestamp):

    unix_timestamp = unix_timestamp / 1000
    # Convert the Unix timestamp to a datetime object
    dt = datetime.datetime.fromtimestamp(unix_timestamp)

    # Format the datetime object as a readable date and time
    formatted_datetime = dt.strftime('%Y-%m-%d %H:%M:%S')

    return formatted_datetime



def fetch_historical_data(start: int, end: int, exchange: str = "coinbase-pro", pair: str = "btcusd", period: int = 3600) -> list:
    """
    Fetch and print historical data for a specific cryptocurrency ticker.

    Parameters
    ----------
    exchange : str
        The exchange to fetch data for.
    pair : str
        The trading pair to fetch data for.
    start : int
        The start date for the data fetch in UNIX timestamp format.
    end : int
        The end date for the data fetch in UNIX timestamp format.
    period : int
        The period to fetch data for, in seconds.

    Returns
    -------
    list
        A list of the desired wick data for bitcoin [Open, Close, High, Low, timestamp, volume]

    Notes
    -----
    This function makes a request to a specific API which does not require an API key.
    """
    url = f"https://api.cryptowat.ch/markets/{exchange}/{pair}/ohlc?after={start}&before={end}&periods={period}"
    response = requests.get(url)
    response = response.json()
    if response.get('result') is not None:   # Ensure the request was successful
        data = response['result']['3600']
        return data
    else:
        print(f"Error fetching data: {response.get('error')}")




def fetch_SMA_data(cryptoTicker: str = "X:BTCUSD", timestamp: int = 1230940800, timespan: str = "day", window: int = 9, order: str = 'asc', limit: int = 24, series_type: str = "close") -> list:
    """
    Fetch Simple Moving Average (SMA) data from a specific API.

    Parameters
    ----------
    api_key : str
        -The API key for the specific service.
    cryptoTicker : str, optional
        -The cryptocurrency ticker to fetch data for. 
        -Default is 'X:BTCUSD' (Bitcoin to USD).
    timestamp: int, optional
        -A unix based timestamp in integer format
        -Data returned will be data that which is greater than the entered timestamp
        -Default is set to January 3rd, 2009
    timespan : str, optional
        -The timespan to consider for the SMA. 
        -Can be one of: ['minute', 'hour', 'day', 'week', 'month', 'quarter', 'year'].
        -Default is 'day'.
    window : int, optional
        -The window of time to consider for the SMA. 
        -Must be an integer between 0 and 200. 
        -Default is 9.
    order: str, optional
        -Whether you want the data return in descending (desc) or ascending (asc) order based on timestamp
        -Default is asc
    limit : int, optional
        -The number of datapoints to be returned
        -The default is set to 24
        -Integer must be greater than 0, but no greater than 5000
    series_type : str, optional
        -The series type to consider for the SMA. 
        -Can be one of: ['open', 'close', 'high', 'low'].
        -Default is 'close'.

    Returns
    -------
    list
        A list of SMA data points if the request is successful; an empty list otherwise.
    """

    endpoint = f"/v1/indicators/sma/{cryptoTicker}?timestamp.gt={timestamp}&timespan={timespan}&window={window}&order=desc&limit={limit}&series_type={series_type}&"
    response = requests.get(BASE_URL + endpoint + api_key)
    response = response.json()
    
    if response.get('status') == 'OK':
        data = response['results']['values']
        return data
    else:
        print(f"Error fetching data: {response.get('error')}")
        return []
    
def fetch_EMA_data(cryptoTicker: str = "X:BTCUSD", timestamp: int = 1230940800, timespan: str = "day", window: int = 9, limit: int = 24, series_type: str = "close", expand_underlying: bool = False) -> list:
    """
    Fetch Exponential Moving Average (EMA) data from a specific API.

    Parameters
    ----------
    cryptoTicker : str, optional
        - The cryptocurrency ticker to fetch data for. 
        - Default is 'X:BTCUSD' (Bitcoin to USD).
    timestamp: int, optional
        - A unix based timestamp in integer format.
        - Data returned will be data that which is greater than the entered timestamp.
        - Default is set to January 3rd, 2009.
    timespan : str, optional
        - The timespan to consider for the EMA. 
        - Can be one of: ['minute', 'hour', 'day', 'week', 'month', 'quarter', 'year'].
        - Default is 'day'.
    window : int, optional
        - The window of time to consider for the EMA. 
        - Must be an integer between 0 and 200. 
        - Default is 9.
    limit : int, optional
        - The number of datapoints to be returned.
        - The default is set to 24.
        - Integer must be greater than 0, but no greater than 5000.
    series_type : str, optional
        - The series type to consider for the EMA. 
        - Can be one of: ['open', 'close', 'high', 'low'].
        - Default is 'close'.
    expand_underlying : bool, optional
        - Whether or not to include the aggregates used to calculate this indicator in the response.
        - Default is False.

    Returns
    -------
    list
        A list of EMA data points if the request is successful; an empty list otherwise.
    """

    endpoint = f"/v1/indicators/ema/{cryptoTicker}?timestamp.gt={timestamp}&timespan={timespan}&window={window}&limit={limit}&series_type={series_type}&expand_underlying={str(expand_underlying).lower()}&order=desc&"

    response = requests.get(BASE_URL + endpoint + api_key)
    response = response.json()
    
    if response.get('status') == 'OK':
        data = response['results']['values']
        return data
    else:
        print(f"Error fetching data: {response.get('error')}")
        return []




def fetch_MACD_data(cryptoTicker: str = "X:BTCUSD", timestamp: int = 1230940800, timespan: str = "day", short_window: int = 12, long_window: int = 26, signal_window: int = 9, series_type: str = "close", expand_underlying: bool = False, limit: int = 24) -> list:
    """
    Fetch Moving Average Convergence/Divergence (MACD) data from a specific API.

    Parameters
    ----------
    cryptoTicker : str, optional
        - The cryptocurrency ticker to fetch data for. 
        - Default is 'X:BTCUSD' (Bitcoin to USD).
    timestamp: int, optional
        - A unix based timestamp in integer format.
        - Data returned will be data that which is greater than the entered timestamp.
        - Default is set to January 3rd, 2009.
    timespan : str, optional
        - The timespan to consider for the MACD. 
        - Can be one of: ['minute', 'hour', 'day', 'week', 'month', 'quarter', 'year'].
        - Default is 'day'.
    short_window : int, optional
        - The short window of time to consider for the MACD. 
        - Default is 12.
    long_window : int, optional
        - The long window of time to consider for the MACD. 
        - Default is 26.
    signal_window : int, optional
        - The window of time used to calculate the MACD signal line. 
        - Default is 9.
    series_type : str, optional
        - The series type to consider for the MACD. 
        - Can be one of: ['open', 'close', 'high', 'low'].
        - Default is 'close'.
    expand_underlying : bool, optional
        - Whether or not to include the aggregates used to calculate this indicator in the response.
        - Default is False.
    limit : int, optional
        - The number of datapoints to be returned.
        - The default is set to 24.
        - Integer must be greater than 0, but no greater than 5000.

    Returns
    -------
    list
        A list of MACD data points if the request is successful; an empty list otherwise.
    """

    endpoint = f"/v1/indicators/macd/{cryptoTicker}?timestamp.gt={timestamp}&timespan={timespan}&short_window={short_window}&long_window={long_window}&signal_window={signal_window}&series_type={series_type}&expand_underlying={str(expand_underlying).lower()}&order=desc&limit={limit}&"

    response = requests.get(BASE_URL + endpoint + api_key)
    response = response.json()
    
    if response.get('status') == 'OK':
        data = response['results']['values']
        return data
    else:
        print(f"Error fetching data: {response.get('error')}")
        return []
    

def fetch_RSI_data(cryptoTicker: str = "X:BTCUSD", timestamp: int = 1230940800, timespan: str = "day", window: int = 14, series_type: str = "close", expand_underlying: bool = False, limit: int = 24) -> list:
    """
    Fetch Relative Strength Index (RSI) data from a specific API.

    Parameters
    ----------
    cryptoTicker : str, optional
        - The cryptocurrency ticker to fetch data for. 
        - Default is 'X:BTCUSD' (Bitcoin to USD).
    timestamp: int, optional
        - A unix based timestamp in integer format.
        - Data returned will be data that which is greater than the entered timestamp.
        - Default is set to January 3rd, 2009.
    timespan : str, optional
        - The timespan to consider for the RSI. 
        - Can be one of: ['minute', 'hour', 'day', 'week', 'month', 'quarter', 'year'].
        - Default is 'day'.
    window : int, optional
        - The window of time to consider for the RSI. 
        - Default is 14.
    series_type : str, optional
        - The series type to consider for the RSI. 
        - Can be one of: ['open', 'close', 'high', 'low'].
        - Default is 'close'.
    expand_underlying : bool, optional
        - Whether or not to include the aggregates used to calculate this indicator in the response.
        - Default is False.
    limit : int, optional
        - The number of datapoints to be returned.
        - The default is set to 24.
        - Integer must be greater than 0, but no greater than 5000.

    Returns
    -------
    list
        A list of RSI data points if the request is successful; an empty list otherwise.
    """

    endpoint = f"/v1/indicators/rsi/{cryptoTicker}?timestamp.gt={timestamp}&timespan={timespan}&window={window}&series_type={series_type}&expand_underlying={str(expand_underlying).lower()}&order=desc&limit={limit}&"

    response = requests.get(BASE_URL + endpoint + api_key)
    response = response.json()
    
    if response.get('status') == 'OK':
        data = response['results']['values']
        return data
    else:
        print(f"Error fetching data: {response.get('error')}")
        return []



def fetch_bitcoin_stats() -> dict:
    """
    Fetch Bitcoin statistics from Blockchair API.

    Returns
    -------
    dict
        A dictionary containing various statistics about the Bitcoin network. If the request is unsuccessful, returns an empty dictionary.
    """

    endpoint = "https://api.blockchair.com/bitcoin/stats"
    
    response = requests.get(endpoint)
    response = response.json()
    
    if response.get('context').get('code') == 200:
        data = response['data']

        # Selecting the necessary fields
        selected_data = {
            'blocks': data['blocks'],
            'transactions': data['transactions'],
            'outputs': data['outputs'],
            'circulation': data['circulation'],
            'difficulty': data['difficulty'],
            'volume_24h': data['volume_24h'],
            'best_block_height': data['best_block_height'],
            'best_block_hash': data['best_block_hash'],
            'best_block_time': data['best_block_time'],
            'average_transaction_fee_24h': data['average_transaction_fee_24h'],
            'median_transaction_fee_24h': data['median_transaction_fee_24h'],
            'market_price_usd': data['market_price_usd'],
            'market_price_btc': data['market_price_btc'],
            'hashrate_24h': data['hashrate_24h'],
            'hodling_addresses' : data['hodling_addresses'],
            'market_cap_usd' : data['market_cap_usd']
        }
        
        return selected_data
    else:
        print(f"Error fetching data: {response.get('error')}")
        return {}



def find_MA_trend(data):
    """
    Finds the most recent trend change based on moving averages (MA) of price data.

    Args:
        data (list): A list of dictionaries containing timestamp and value pairs.

    Returns:
        int
            A unix based timestamp for the most recent determined trend start

    Raises:
        None.

    """
    
    df = pd.DataFrame(data)

    # Assuming df is your DataFrame and 'price' is the column with price data
    df['MA_short'] = df['value'].rolling(window=10).mean()
    df['MA_long'] = df['value'].rolling(window=50).mean()

    # Create a column 'MA_diff' such that it is 1 where MA_short is above MA_long and -1 otherwise
    df['MA_diff'] = np.where(df['MA_short'] > df['MA_long'], 1, -1)

    # Now, the points where 'MA_diff' changes from 1 to -1 or -1 to 1 would be the trend changes
    df['trend_change'] = df['MA_diff'].diff()

    # Find the most recent trend change
    most_recent_trend_change = df[df['trend_change'] != 0].iloc[-1]

    trend_timestamp = int(most_recent_trend_change['timestamp'])
    # Print the timestamp of the most recent trend change
    print(f"The most recent trend change occurred at: {trend_timestamp}")

    return trend_timestamp



def determine_fibonacci_retracement(data):
     """Determine Fibonacci retracement levels based on the provided data.

    Args:
        data (list): A list of tuples representing the data, where each tuple contains a timestamp and value.

    Returns:
        None

    Prints:
        Fibonacci Retracement Levels: Prints the Fibonacci retracement levels and their corresponding values.

    """

    # Convert to pandas DataFrame
     df = pd.DataFrame(data, columns=['timestamp', 'value'])
    
     # Convert 'timestamp' to datetime
     df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
    
     # Sort DataFrame by 'time'
     df = df.sort_values('time')
    
     # Identify the high and low prices
     high_price = df['value'].max()
     low_price = df['value'].min()
    
     # Calculate Fibonacci levels
     difference = high_price - low_price
     fibonacci_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 1.0]
     fibo_values = [high_price - difference * level for level in fibonacci_levels]
    
     # Print Fibonacci retracement values
     print("Fibonacci Retracement Levels:")
     for level, value in zip(fibonacci_levels, fibo_values):
        print(f"Level {level*100}%: {value}")



def calculate_stochastic_oscillator(data, period=14):
    # Convert to pandas DataFrame
    df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'quoteVolume'])

    # Convert 'time' to datetime
    df['time'] = pd.to_datetime(df['time'], unit='s') # I'm assuming the time is in seconds

    # Sort DataFrame by 'time'
    df = df.sort_values('time')

    # Calculate highest high and lowest low for each period
    df['high'] = df['high'].rolling(window=period).max()
    df['low'] = df['low'].rolling(window=period).min()

    # Calculate %K and %D for Stochastic Oscillator
    df['%K'] = (df['close'] - df['low']) / (df['high'] - df['low']) * 100
    df['%D'] = df['%K'].rolling(window=3).mean() # commonly used smoothing is 3 periods

    return df[['time', '%K', '%D']]



def get_bitcoin_news(news_api_key: str):
    
    # Set up the API request URL
    url = f"https://gnews.io/api/v4/search?q=bitcoin&token={news_api_key}&max=10&lang=en&from={datetime.date.today()}"
    # Make the request
    response = requests.get(url)

    # Check if request was successful
    if response.status_code != 200:
        print(f"Request failed with status {response.status_code}")
        return []

    # Parse the response JSON
    data = response.json()

    # Extract articles from the data
    articles = data.get('articles', [])

    # If we want to extract only the title and description of the news
    news_data = []
    for article in articles:
        title = article.get('title', '')
        description = article.get('description', '')
        url = article.get('url', '')
        news_data.append({'title': title, 'description': description, 'url': url})

    # return the news data
    return news_data


def analyze_sentiment_and_extract_themes(articles):


    news_sentiment_data = []
    for article in articles:
        print(article['url'])
        #obtain the news article text
        article_text = obtain_article_text(article['url'])

    
        prompt = f"""
        {{
            "task": "analyze_text",
            "text": "{article_text}"
        }}
        Please perform a sentiment analysis on the text, and also identify the key points or topics present in the text. Return as json format {{"sentiment": {{"score": sentiment_score_here(float between -1 to 1, +is positive),"label": label_here(positive, neutral or negative), 'reasoning':reasoning_for_sentiment_score_here}}, "keypoints": ["trade-war", "legal issues for bitcoin", "bitcoin sees adoption"]}}
        """

        # Make a request to the API
        response = gpt4_completion(prompt)
        
        try:
            output = json.loads(response)
            news_sentiment_data.append(output)
        except json.JSONDecodeError:
            print("Failed to parse JSON output.")
            

    return news_sentiment_data


def calculate_overall_sentiment(articles):
    total_weighted_score = 0
    total_weights = 0
    weight_label = {'positive': 1, 'negative': 1.5, 'neutral': 0.5} # customize weights as needed

    for article in articles:
        sentiment_score = article['sentiment']['score']
        sentiment_label = article['sentiment']['label']
        num_points = len(article['keypoints'])

        # Calculate a weight for the current article based on the label and the number of key points
        weight = weight_label[sentiment_label] * num_points

        total_weighted_score += weight * abs(sentiment_score) # Abs value emphasizes extreme sentiments
        total_weights += weight

    if total_weights == 0:
        return None

    overall_sentiment = total_weighted_score / total_weights
    return overall_sentiment

##VARIABLES 
BASE_URL = "https://api.polygon.io"
api_key = ""

cryptoTicker = "X:BTCUSD"
multiplier = 1
timespan = "day"
current_timestamp = int(datetime.datetime.now().timestamp())
start_timestamp = current_timestamp - (1 * 7 * 24 * 60 * 60) ## make this a function we can just enter a day variable into and then returns the proper timestamp

moving_average_data = fetch_SMA_data(limit=168, timespan='hour')
trend_timestamp = find_MA_trend(moving_average_data)
trend_data = fetch_SMA_data(timestamp=trend_timestamp, timespan='hour')
fibonacci = determine_fibonacci_retracement(trend_data)
historical_data = fetch_historical_data(start=start_timestamp, end=current_timestamp)
print(calculate_stochastic_oscillator(historical_data))

##################TESTING#########################

news = get_bitcoin_news()

news_count = len(news)
if news_count < 30:
    sentiment_report = analyze_sentiment_and_extract_themes(news)
    print(sentiment_report)
    print(calculate_overall_sentiment(sentiment_report))
else:
    print(f'Sorry the article count is: {news_count}')
