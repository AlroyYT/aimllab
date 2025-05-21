from prophet import Prophet
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from datetime import datetime, timedelta

def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph

def forecast(ticker):
    # Get the data from Yahoo Finance
    try:
        df = yf.Ticker(ticker).history(period='5y', interval='1d')
        if df.empty:
            return None, None
    except Exception as e:
        print(f"Error retrieving data: {str(e)}")
        return None, None
    
    df = df[['Close']]
    
    # Create Prophet dataframe
    dfx = pd.DataFrame()
    # Convert index to datetime and remove timezone information
    dfx['ds'] = pd.to_datetime(df.index).tz_localize(None)
    dfx['y'] = df.Close.values
    
    # Fit Prophet model
    fbp = Prophet(daily_seasonality=True)
    fbp.fit(dfx)
    
    # Make future predictions
    fut = fbp.make_future_dataframe(periods=365)
    forecast = fbp.predict(fut)
    
    # Create the plot
    plot = fbp.plot(forecast)
    plt.xlabel("Date")
    plt.ylabel("Price")
    graph = get_graph()
    
    # Calculate metrics for the dashboard
    metrics = calculate_metrics(df, forecast, ticker, fbp, fut)
    
    return graph, metrics

def calculate_metrics(df, forecast, ticker, fbp=None, fut=None):
    try:
        # Get the latest historical price
        last_price = round(df['Close'].iloc[-1], 2)
        last_date = df.index[-1].strftime('%b %d, %Y')
        
        # Calculate forecast metrics
        forecast_end_price = round(forecast['yhat'].iloc[-1], 2)
        growth_pct = round(((forecast_end_price - last_price) / last_price) * 100, 1)
        growth_class = "positive" if growth_pct >= 0 else "negative"
        growth_icon = "arrow-up" if growth_pct >= 0 else "arrow-down"
        
        # Calculate peak and min prices
        future_forecast = forecast.iloc[-365:]  # Only consider the forecast part
        peak_price = round(future_forecast['yhat_upper'].max(), 2)
        min_price = round(future_forecast['yhat_lower'].min(), 2)
        
        peak_growth = round(((peak_price - last_price) / last_price) * 100, 1)
        min_decline = round(((min_price - last_price) / last_price) * 100, 1)
        min_class = "positive" if min_decline >= 0 else "negative"
        min_icon = "arrow-up" if min_decline >= 0 else "arrow-down"
        
        # Calculate volatility
        volatility = round(future_forecast['yhat_upper'].std() / future_forecast['yhat'].mean() * 100, 1)
        
        # Support and resistance levels
        support_level = round(min_price * 0.95, 2)  # Simple approximation
        resistance_level = round(peak_price * 1.05, 2)  # Simple approximation
        
        # Determine trend
        if growth_pct > 15:
            trend = "strongly bullish"
            trend_class = "strongly-bullish"
            trend_icon = "arrow-up"
        elif growth_pct > 5:
            trend = "moderately bullish"
            trend_class = "moderately-bullish"
            trend_icon = "arrow-up"
        elif growth_pct > -5:
            trend = "neutral"
            trend_class = "neutral-trend"
            trend_icon = "minus"
        elif growth_pct > -15:
            trend = "moderately bearish"
            trend_class = "moderately-bearish"
            trend_icon = "arrow-down"
        else:
            trend = "strongly bearish"
            trend_class = "strongly-bearish"
            trend_icon = "arrow-down"
        
        # Quarterly breakdown
        quarterly_forecast = []
        
        # Starting from today, divide the 365-day forecast into quarters
        current_date = datetime.now()
        
        for i in range(4):
            start_date = current_date + timedelta(days=i*90)
            end_date = current_date + timedelta(days=(i+1)*90 - 1)
            
            # Convert to string format for Prophet's ds column
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # Filter forecast for this quarter
            quarter_forecast = forecast[(forecast['ds'] >= start_date_str) & 
                                      (forecast['ds'] <= end_date_str)]
            
            # Calculate quarter metrics
            if not quarter_forecast.empty:
                quarter_data = {
                    'period': f"Q{i+1}",
                    'avg_price': round(quarter_forecast['yhat'].mean(), 2),
                    'min_price': round(quarter_forecast['yhat_lower'].min(), 2),
                    'max_price': round(quarter_forecast['yhat_upper'].max(), 2),
                    'end_date': end_date.strftime('%b %d, %Y')
                }
                quarterly_forecast.append(quarter_data)
        
        # Generate insights
        try:
            stock_info = yf.Ticker(ticker).info
            company_name = stock_info.get('shortName', ticker)
        except:
            company_name = ticker
        
        # Time horizon analysis
        future_30d = forecast[(forecast['ds'] >= (datetime.now()).strftime('%Y-%m-%d')) & 
                            (forecast['ds'] <= (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'))]
        future_90d = forecast[(forecast['ds'] >= (datetime.now()).strftime('%Y-%m-%d')) & 
                            (forecast['ds'] <= (datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d'))]
        future_365d = forecast[(forecast['ds'] >= (datetime.now()).strftime('%Y-%m-%d'))]
        
        short_term_growth = round(((future_30d['yhat'].iloc[-1] - last_price) / last_price) * 100, 1)
        mid_term_growth = round(((future_90d['yhat'].iloc[-1] - last_price) / last_price) * 100, 1)
        long_term_growth = round(((future_365d['yhat'].iloc[-1] - last_price) / last_price) * 100, 1)
        
        # Generate text and classes for time horizons
        if short_term_growth > 5:
            short_term_class = "positive"
            short_term_icon = "arrow-up"
            short_term_text = f"Expected increase of {short_term_growth}% in the next month."
        elif short_term_growth < -5:
            short_term_class = "negative"
            short_term_icon = "arrow-down"
            short_term_text = f"Expected decrease of {abs(short_term_growth)}% in the next month."
        else:
            short_term_class = "neutral"
            short_term_icon = "minus"
            short_term_text = f"Relatively stable price expected with {short_term_growth}% change."
        
        if mid_term_growth > 8:
            mid_term_class = "positive"
            mid_term_icon = "arrow-up"
            mid_term_text = f"Strong upward trend of {mid_term_growth}% expected in the next quarter."
        elif mid_term_growth < -8:
            mid_term_class = "negative"
            mid_term_icon = "arrow-down"
            mid_term_text = f"Significant downward trend of {abs(mid_term_growth)}% expected in the next quarter."
        else:
            mid_term_class = "neutral"
            mid_term_icon = "minus"
            mid_term_text = f"Moderate fluctuations with net {mid_term_growth}% change expected in mid-term."
        
        if long_term_growth > 15:
            long_term_class = "positive"
            long_term_icon = "arrow-up"
            long_term_text = f"Strong bullish outlook with {long_term_growth}% growth projected for the year."
        elif long_term_growth < -15:
            long_term_class = "negative"
            long_term_icon = "arrow-down"
            long_term_text = f"Bearish long-term outlook with {abs(long_term_growth)}% decline projected."
        else:
            long_term_class = "neutral"
            long_term_icon = "minus"
            long_term_text = f"Moderate long-term trend with {long_term_growth}% change over the year."
        
        # Generate insight text based on the forecasted trend
        if growth_pct > 10:
            insight_text = f"The forecast for {company_name} shows strong bullish potential over the next 12 months, with an expected price appreciation of {growth_pct}%. Technical indicators suggest sustained momentum could push prices beyond the projected high of ${peak_price}."
        elif growth_pct > 0:
            insight_text = f"The {company_name} forecast indicates moderate upside potential of {growth_pct}% over the next year. While the overall trajectory is positive, periods of consolidation are likely, with support established around ${support_level}."
        elif growth_pct > -10:
            insight_text = f"The {company_name} price forecast suggests a relatively flat trajectory with a slight {abs(growth_pct)}% decline over the next 12 months. Significant volatility ({volatility}%) may create both risks and opportunities within this overall neutral trend."
        else:
            insight_text = f"The forecast for {company_name} indicates a bearish outlook with a projected {abs(growth_pct)}% decline over the next year. Investors should be cautious of resistance around ${resistance_level} and potential accelerated downside movements."
        
        # Seasonal patterns detection
        # Only attempt seasonal pattern analysis if fbp and fut are provided
        if fbp is not None and fut is not None:
            try:
                components = fbp.predict_components(fut)
                weekly_pattern = components['weekly'].iloc[-7:].values
                
                if np.std(weekly_pattern) > 0.01 * last_price:
                    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    strongest_day = weekdays[np.argmax(weekly_pattern)]
                    weakest_day = weekdays[np.argmin(weekly_pattern)]
                    seasonal_pattern = f"Weekly seasonality detected: {strongest_day} typically shows the strongest performance, while {weakest_day} tends to underperform."
                else:
                    seasonal_pattern = "No significant weekly seasonality detected in the historical data."
            except:
                seasonal_pattern = "Seasonality analysis not available for this forecast."
        else:
            seasonal_pattern = "Seasonality analysis not available for this forecast."
        
        # Additional insights based on volatility
        if volatility > 30:
            additional_insights = f"High volatility ({volatility}%) suggests significant price swings are expected. Consider adjusting position sizes and implementing risk management strategies accordingly."
        elif volatility > 15:
            additional_insights = f"Moderate volatility ({volatility}%) is expected, typical for this asset class. Regular rebalancing may be beneficial to manage risk."
        else:
            additional_insights = f"Relatively low volatility ({volatility}%) is predicted, suggesting more stable price action compared to historical patterns."
        
        # Compile all metrics
        metrics = {
            'last_price': last_price,
            'last_date': last_date,
            'forecast_end_price': forecast_end_price,
            'growth_pct': growth_pct,
            'growth_class': growth_class,
            'growth_icon': growth_icon,
            'peak_price': peak_price,
            'peak_growth': peak_growth,
            'min_price': min_price,
            'min_decline': min_decline,
            'min_class': min_class,
            'min_icon': min_icon,
            'volatility': volatility,
            'support_level': support_level,
            'resistance_level': resistance_level,
            'trend': trend,
            'trend_class': trend_class,
            'trend_icon': trend_icon,
            'quarterly_forecast': quarterly_forecast,
            'insight_text': insight_text,
            'short_term_class': short_term_class,
            'short_term_icon': short_term_icon,
            'short_term_text': short_term_text,
            'mid_term_class': mid_term_class,
            'mid_term_icon': mid_term_icon,
            'mid_term_text': mid_term_text,
            'long_term_class': long_term_class,
            'long_term_icon': long_term_icon,
            'long_term_text': long_term_text,
            'seasonal_pattern': seasonal_pattern,
            'additional_insights': additional_insights
        }
        
        return metrics
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        # Return minimal default metrics to prevent template errors
        return {
            'last_price': 0.00,
            'last_date': datetime.now().strftime('%b %d, %Y'),
            'forecast_end_price': 0.00,
            'growth_pct': 0.0,
            'growth_class': 'neutral',
            'growth_icon': 'minus',
            'peak_price': 0.00,
            'peak_growth': 0.0,
            'min_price': 0.00,
            'min_decline': 0.0,
            'min_class': 'neutral',
            'min_icon': 'minus',
            'volatility': 0.0,
            'support_level': 0.00,
            'resistance_level': 0.00,
            'trend': 'neutral',
            'trend_class': 'neutral-trend',
            'trend_icon': 'minus',
            'quarterly_forecast': [
                {'period': 'Q1', 'avg_price': 0.00, 'min_price': 0.00, 'max_price': 0.00, 'end_date': (datetime.now() + timedelta(days=90)).strftime('%b %d, %Y')},
                {'period': 'Q2', 'avg_price': 0.00, 'min_price': 0.00, 'max_price': 0.00, 'end_date': (datetime.now() + timedelta(days=180)).strftime('%b %d, %Y')},
                {'period': 'Q3', 'avg_price': 0.00, 'min_price': 0.00, 'max_price': 0.00, 'end_date': (datetime.now() + timedelta(days=270)).strftime('%b %d, %Y')},
                {'period': 'Q4', 'avg_price': 0.00, 'min_price': 0.00, 'max_price': 0.00, 'end_date': (datetime.now() + timedelta(days=360)).strftime('%b %d, %Y')}
            ],
            'insight_text': f"Unable to generate forecast insights for {ticker}. Please try again later.",
            'short_term_class': 'neutral',
            'short_term_icon': 'minus',
            'short_term_text': "Short-term forecast unavailable.",
            'mid_term_class': 'neutral',
            'mid_term_icon': 'minus',
            'mid_term_text': "Mid-term forecast unavailable.",
            'long_term_class': 'neutral',
            'long_term_icon': 'minus',
            'long_term_text': "Long-term forecast unavailable.",
            'seasonal_pattern': "Seasonality analysis not available.",
            'additional_insights': "Additional insights not available."
        }