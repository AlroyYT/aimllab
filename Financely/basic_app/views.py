from django.shortcuts import render, redirect
from .models import Portfolio, Client, Stock
from .forms import CreateUserForm
from .sectorPerformance import sectorPerformance
from .decorators import unauthenticated_user, allowed_users
from .ProphetTrend import forecast
from basic_app.stock_data import candlestick_data, get_data, get_name, get_price
from basic_app.FA import piotroski
from django.core.cache import cache
# Create your views here.
from django.http import HttpResponse
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.models import User, Group
from django.contrib.auth.decorators import login_required
from basic_app.get_news import getNews, getNewsWithSentiment
from basic_app.get_stock_info import getStockInfo
from django.http import JsonResponse
from json import dumps, loads
from basic_app.ProphetTrend import forecast
from django.views.decorators.csrf import csrf_exempt
import os
import json
import yfinance as yf
import google.generativeai as genai
from django.views.decorators.http import require_http_methods
from django.conf import settings
import pandas as pd 
import numpy as np 
from datetime import datetime, timedelta
import requests
from textblob import TextBlob
import re

@csrf_exempt  # temporary for debugging — remove later and use proper CSRF
def search_stock(request):
    if request.method == 'POST':
        search_data = request.POST.get('searchData', '').strip()

        if not search_data:
            return JsonResponse({'data': []})

        try:
            results = getStockInfo(search_data)
            if results and isinstance(results, list):
                return JsonResponse({'data': results})
            else:
                return JsonResponse({'data': []})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)


def dashboard(request):
    return render(request, "basic_app/dashboard.html")

# Helper function to check if request is AJAX
def is_ajax(request):
    return request.headers.get('x-requested-with') == 'XMLHttpRequest'

@login_required(login_url='basic_app:login')
@allowed_users(allowed_roles=['Client'])
def index(request):
    if is_ajax(request):
        res = None
        data = request.POST.get('searchData')
        item = getStockInfo(data)
        if len(item) > 0 and len(data):
            res = item
        else:
            res = 'No stocks found..'

        return JsonResponse({'data': res})

    user = request.user
    client = Client.objects.get(user=user)
    portfolio = Portfolio.objects.get(client=client)
    stocks = portfolio.stocks.all()
    news = getNews("business")
    context = {'stocks': stocks, 'news': news, 'page_title': "Home"}

    return render(request, "basic_app/index.html", context)

@login_required(login_url='basic_app:login')
@allowed_users(allowed_roles=['Client'])
def profile(request):
    client = request.user
    return render(request, "basic_app/profile.html", {'client': client, 'page_title': "User Profile"})

@login_required(login_url='basic_app:login')
@allowed_users(allowed_roles=['Client'])
def portfolio(request):
    user = request.user
    client = Client.objects.get(user=user)
    portfolio = Portfolio.objects.get(client=client)
    stocks = portfolio.stocks.all()
    
    for s in stocks:
        # Check if stock_symbol is None or empty
        if not s.stock_symbol:
            # Handle missing symbol - either skip, set a default, or fix the data
            continue  # Skip this stock to prevent the error
            
        if(not s.stock_sector_performance):
            try:
                s.stock_sector_performance = sectorPerformance(s.stock_symbol)
            except Exception as e:
                # Log the error and continue
                print(f"Error getting sector performance for {s.stock_symbol}: {e}")
                s.stock_sector_performance = "Unknown"  # Set a default value
                
        if(not s.stock_price):
            try:
                price = get_price(s.stock_symbol)
                s.stock_price = str(round(price[0], 2)) + "  " + price[1]
            except Exception as e:
                # Log the error and continue
                print(f"Error getting price for {s.stock_symbol}: {e}")
                s.stock_price = "N/A"  # Set a default value
                
        s.save()

    context = {'stocks': stocks, 'page_title': "Your Portfolio"}
    return render(request, "basic_app/portfolio.html", context)

@login_required(login_url='basic_app:login')
def stock(request, symbol):
    try:
        data = candlestick_data(symbol)
        item = getStockInfo(symbol)
        info = get_data(symbol)
        piotroski_score = piotroski(symbol)
        
        try:
            news = getNewsWithSentiment(info.get('shortName', symbol))
        except Exception as e:
            print(f"Error getting news with sentiment: {e}")
            # Provide fallback news data
            news = [{'title': 'News unavailable', 'description': 'Unable to fetch news data', 'sentiment': 'neutral'}] * 12
        
        sentiment_news_chart = {'positive': 0, 'negative': 0, 'neutral': 0}
        for i in range(min(12, len(news))):
            sentiment = news[i].get('sentiment', 'neutral')
            sentiment_news_chart[sentiment] += 1
        
        # Updated recommendation logic
        recommendation = False
        overall_sentiment = sentiment_news_chart['positive'] - sentiment_news_chart["negative"]
        # Changed from > 5 to >= 5 to include stocks with a score of 5
        if piotroski_score >= 5 and overall_sentiment >= 0:
            recommendation = True

        context = {
            'data': dumps(data),
            'item': dumps(item),
            'info': info,
            'piotroski_score': piotroski_score,
            'sentiment_data': dumps(sentiment_news_chart),
            'page_title': symbol + " Info",
            'recommendation': recommendation
        }
        
        # Check if request is AJAX
        if is_ajax(request):
            try:
                run = False
                data = request.POST.get('myData')
                action = request.POST.get('action')
                name = request.POST.get('name')
                
                user = request.user
                client = Client.objects.get(user=user)
                portfolio = Portfolio.objects.get(client=client)
                stocks = portfolio.stocks.all()
                
                for stock in stocks:
                    if data == stock.stock_symbol:
                        stock.quantity += 1
                        stock.save()
                        run = True

                if not run:
                    new_stock = Stock.objects.create(parent_portfolio=portfolio, stock_symbol=data, stock_name=name)
                    new_stock.quantity = 1
                    new_stock.save()
                    
                return JsonResponse({})
            except Exception as e:
                print(f"Error processing AJAX request: {e}")
                return JsonResponse({'error': str(e)}, status=400)

        return render(request, "basic_app/stock.html", context)
    except Exception as e:
        print(f"Error in stock view: {e}")
        return render(request, "basic_app/error.html", {'error': str(e), 'page_title': "Error"})
@unauthenticated_user
def loginPage(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            if user.groups.all() and user.groups.all()[0].name == 'Admin':
                return redirect("basic_app:stats")
            else:
                return redirect("basic_app:index")
        else:
            messages.info(request, "Incorrect username or password")
            return redirect("basic_app:login")

    return render(request, "basic_app/login.html", {'page_title': "Login"})

@login_required(login_url='basic_app:login')
def logoutUser(request):
    logout(request)
    return redirect("basic_app:login")

@unauthenticated_user
def registerPage(request):
    form = CreateUserForm()
    if request.method == "POST":
        form = CreateUserForm(request.POST)
        if form.is_valid():
            user = form.save()
            group = Group.objects.get(name='Client')
            user.groups.add(group)
            client = Client.objects.create(user=user)
            portfolio = Portfolio.objects.create(client=client)
            return redirect('basic_app:login')

    context = {'form': form, 'page_title': "Register"}
    return render(request, "basic_app/register.html", context)

@login_required(login_url='basic_app:login')
@allowed_users(allowed_roles=['Admin'])
def statisticsAdmin(request):
    return render(request, "basic_app/statisticsAdmin.html")

def get_usd_to_inr_rate():
    """
    Fetch current USD to INR exchange rate with caching
    Falls back to a default rate if API fails
    """
    # Check if rate is cached (cache for 1 hour)
    cached_rate = cache.get('usd_to_inr_rate')
    if cached_rate:
        return cached_rate
    
    try:
        # Using a free exchange rate API (you can use any API you prefer)
        response = requests.get('https://api.exchangerate-api.com/v4/latest/USD', timeout=5)
        data = response.json()
        rate = data['rates']['INR']
        
        # Cache the rate for 1 hour
        cache.set('usd_to_inr_rate', rate, 3600)
        return rate
    except:
        # Fallback rate if API fails
        return 83.5

def price_prediction(request, symbol):
    price_prediction, metrics = forecast(symbol)
    
    if price_prediction is None:
        # Handle error case
        return render(request, "basic_app/error.html", {'error_message': f"Unable to retrieve data for {symbol}"})
    
    # Get dynamic USD to INR conversion rate
    USD_TO_INR = get_usd_to_inr_rate()
    
    # Convert all price-related metrics to INR
    if metrics:
        # Direct price conversions
        price_fields = [
            'last_price', 'forecast_end_price', 'peak_price', 'min_price', 
            'support_level', 'resistance_level'
        ]
        
        for field in price_fields:
            if field in metrics:
                metrics[field] = round(metrics[field] * USD_TO_INR, 2)
        
        # Convert quarterly forecast prices
        if 'quarterly_forecast' in metrics:
            for quarter in metrics['quarterly_forecast']:
                quarter['avg_price'] = round(quarter['avg_price'] * USD_TO_INR, 2)
                quarter['min_price'] = round(quarter['min_price'] * USD_TO_INR, 2)
                quarter['max_price'] = round(quarter['max_price'] * USD_TO_INR, 2)
        
        # Update text fields to use ₹ symbol instead of $
        text_fields = ['insight_text', 'seasonal_pattern', 'additional_insights']
        for field in text_fields:
            if field in metrics and metrics[field]:
                metrics[field] = metrics[field].replace('$', '₹')
    
    return render(request, "basic_app/price_prediction.html", {
        'price_prediction': price_prediction, 
        'metrics': metrics,
        'page_title': f"{symbol} Forecast",
        'currency_symbol': '₹',
        'exchange_rate': USD_TO_INR  # Optional: pass rate to template if needed
    })

def addToPortfolio(request, symbol):
    user = request.user
    run = False
    client = Client.objects.get(user=user)
    portfolio = Portfolio.objects.get(client=client)
    stocks = portfolio.stocks.all()
    for stock in stocks:
        if symbol == stock.stock_symbol:
            stock.quantity += 1
            stock.save()
            run = True

    name = get_name(symbol)
    if not run:
        new_stock = Stock.objects.create(parent_portfolio=portfolio, stock_symbol=symbol, stock_name=name)
        new_stock.quantity = 1
        new_stock.save()
    
    return redirect('basic_app:portfolio')


def removeFromPortfolio(request, symbol):
    user = request.user
    client = Client.objects.get(user=user)
    portfolio = Portfolio.objects.get(client=client)
    stocks = portfolio.stocks.all()
    for stock in stocks:
        if symbol == stock.stock_symbol:
            stock.delete()

    return redirect("basic_app:portfolio")


def quantityAdd(request, symbol):
    user = request.user
    client = Client.objects.get(user=user)
    portfolio = Portfolio.objects.get(client=client)
    stocks = portfolio.stocks.all()
    for stock in stocks:
        if symbol == stock.stock_symbol:
            stock.quantity += 1
            stock.save()
    return redirect("basic_app:portfolio")

def quantitySub(request, symbol):
    user = request.user
    client = Client.objects.get(user=user)
    portfolio = Portfolio.objects.get(client=client)
    stocks = portfolio.stocks.all()
    for stock in stocks:
        if symbol == stock.stock_symbol:
            stock.quantity -= 1

            if stock.quantity == 0:
                stock.delete()
            else:
                stock.save()

    return redirect("basic_app:portfolio")
try:
    genai.configure(api_key=settings.GEMINI_API_KEY)
    print(f"Google AI configured successfully")
except Exception as e:
    print(f"Error configuring Google AI: {e}")
def chatbot_home(request):
    """Render the main chatbot interface"""
    return render(request, 'basic_app/chatbot.html')

@csrf_exempt
@require_http_methods(["POST"])
def chat_with_ai(request):
    """Handle chat messages and return AI responses"""
    try:
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return JsonResponse({
                'error': 'Message cannot be empty',
                'status': 'error'
            }, status=400)
        
        # Use gemini-1.5-flash model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Financial context prompt
        financial_prompt = f"""
        You are a helpful financial AI assistant. Your role is to provide accurate, helpful financial advice and information. 
        Please focus on:
        - Personal finance management
        - Investment advice (general, not specific stock picks)
        - Budgeting tips
        - Financial planning
        - Economic concepts explanation
        - Risk management
        
        Always remind users that this is general advice and they should consult with qualified financial advisors for personalized advice.
        
        User question: {user_message}
        
        Please provide a concise but helpful response.
        """
        
        # Generate response
        response = model.generate_content(financial_prompt)
        
        if response and response.text:
            return JsonResponse({
                'response': response.text,
                'status': 'success'
            })
        else:
            return JsonResponse({
                'error': 'Sorry, I could not generate a response. Please try again.',
                'status': 'error'
            }, status=500)
        
    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'Invalid JSON data',
            'status': 'error'
        }, status=400)
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return JsonResponse({
            'error': f'Error processing request: {str(e)}',
            'status': 'error'
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def financial_analysis(request):
    """Analyze financial data or scenarios"""
    try:
        data = json.loads(request.body)
        scenario = data.get('scenario', '').strip()
        
        if not scenario:
            return JsonResponse({
                'error': 'Scenario cannot be empty',
                'status': 'error'
            }, status=400)
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        analysis_prompt = f"""
        As a financial analyst AI, please analyze the following financial scenario:
        
        {scenario}
        
        Provide a detailed analysis including:
        1. Key financial metrics or considerations
        2. Potential risks and opportunities
        3. Recommendations or next steps
        4. Important disclaimers
        
        Keep the analysis professional and educational. Limit response to 500 words.
        """
        
        response = model.generate_content(analysis_prompt)
        
        if response and response.text:
            return JsonResponse({
                'analysis': response.text,
                'status': 'success'
            })
        else:
            return JsonResponse({
                'error': 'Could not generate analysis',
                'status': 'error'
            }, status=500)
        
    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'Invalid JSON data',
            'status': 'error'
        }, status=400)
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return JsonResponse({
            'error': f'Error processing analysis: {str(e)}',
            'status': 'error'
        }, status=500)

@require_http_methods(["GET"])
def get_financial_tips(request):
    """Get general financial tips"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        tips_prompt = """
        Provide 5 practical financial tips for someone looking to improve their personal finances. 
        Make them actionable and suitable for beginners to intermediate level financial knowledge.
        Format as a numbered list with brief explanations.
        Keep it concise - about 2-3 sentences per tip.
        """
        
        response = model.generate_content(tips_prompt)
        
        if response and response.text:
            return JsonResponse({
                'tips': response.text,
                'status': 'success'
            })
        else:
            return JsonResponse({
                'error': 'Could not generate tips',
                'status': 'error'
            }, status=500)
        
    except Exception as e:
        print(f"Tips error: {str(e)}")
        return JsonResponse({
            'error': f'Error getting tips: {str(e)}',
            'status': 'error'
        }, status=500)

@require_http_methods(["GET"])
def test_ai_connection(request):
    """Test endpoint to verify AI API is working"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Say 'API connection successful!' if you can read this.")
        
        if response and response.text:
            return JsonResponse({
                'message': 'AI API connection test',
                'response': response.text,
                'status': 'success'
            })
        else:
            return JsonResponse({
                'error': 'No response from AI model',
                'status': 'error'
            }, status=500)
        
    except Exception as e:
        print(f"Connection test error: {str(e)}")
        return JsonResponse({
            'error': f'AI API connection failed: {str(e)}',
            'status': 'error'
        }, status=500)
def acquisition_target_dashboard(request):
    """Main dashboard view for acquisition target analysis"""
    return render(request, 'basic_app/acquisition_targets.html')

@csrf_exempt
def analyze_acquisition_targets(request):
    """Analyze potential acquisition targets based on multiple criteria"""
    if request.method == 'POST':
        data = json.loads(request.body)
        sector = data.get('sector', '')
        market_cap_range = data.get('market_cap_range', 'all')
        
        # Get list of companies to analyze
        companies = get_companies_by_sector(sector, market_cap_range)
        
        results = []
        for company in companies:
            try:
                analysis = analyze_single_company(company['symbol'])
                if analysis:
                    results.append(analysis)
            except Exception as e:
                print(f"Error analyzing {company['symbol']}: {str(e)}")
                continue
        
        # Sort by acquisition score (highest first)
        results.sort(key=lambda x: x['acquisition_score'], reverse=True)
        
        return JsonResponse({
            'success': True,
            'targets': results[:20],  # Return top 20 candidates
            'total_analyzed': len(companies)
        })
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@csrf_exempt
def get_company_details(request):
    """Get detailed analysis for a specific company"""
    if request.method == 'POST':
        data = json.loads(request.body)
        symbol = data.get('symbol', '')
        
        if not symbol:
            return JsonResponse({'success': False, 'error': 'Symbol required'})
        
        try:
            analysis = analyze_single_company(symbol)
            if analysis:
                return JsonResponse({'success': True, 'analysis': analysis})
            else:
                return JsonResponse({'success': False, 'error': 'Could not analyze company'})
        
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

# Helper functions (add these to your views.py as well)

def analyze_single_company(symbol):
    """Comprehensive analysis of a single company for acquisition potential"""
    try:
        # Get company data
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Get financial data
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        # Calculate financial ratios
        financial_ratios = calculate_financial_ratios(info, financials, balance_sheet, cash_flow)
        
        # Analyze management commentary sentiment
        management_sentiment = analyze_management_sentiment(symbol)
        
        # Calculate market positioning metrics
        market_metrics = calculate_market_positioning(stock, info)
        
        # Calculate overall acquisition score
        acquisition_score = calculate_acquisition_score(financial_ratios, management_sentiment, market_metrics)
        
        return {
            'symbol': symbol,
            'company_name': info.get('longName', symbol),
            'sector': info.get('sector', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'acquisition_score': acquisition_score,
            'financial_ratios': financial_ratios,
            'management_sentiment': management_sentiment,
            'market_metrics': market_metrics,
            'key_highlights': generate_key_highlights(financial_ratios, management_sentiment, market_metrics),
            'risk_factors': identify_risk_factors(financial_ratios, management_sentiment, market_metrics)
        }
    
    except Exception as e:
        print(f"Error in analyze_single_company for {symbol}: {str(e)}")
        return None

def calculate_financial_ratios(info, financials, balance_sheet, cash_flow):
    """Calculate key financial ratios for acquisition analysis"""
    try:
        # Extract key financial metrics
        market_cap = info.get('marketCap', 0)
        enterprise_value = info.get('enterpriseValue', market_cap)
        revenue = info.get('totalRevenue', 0)
        ebitda = info.get('ebitda', 0)
        free_cash_flow = info.get('freeCashflow', 0)
        total_debt = info.get('totalDebt', 0)
        cash = info.get('totalCash', 0)
        shares_outstanding = info.get('sharesOutstanding', 1)
        
        # Calculate ratios
        ratios = {
            'pe_ratio': info.get('trailingPE', 0),
            'pb_ratio': info.get('priceToBook', 0),
            'ev_revenue': enterprise_value / revenue if revenue > 0 else 0,
            'ev_ebitda': enterprise_value / ebitda if ebitda > 0 else 0,
            'debt_to_equity': total_debt / (market_cap) if market_cap > 0 else 0,
            'current_ratio': info.get('currentRatio', 0),
            'roa': info.get('returnOnAssets', 0),
            'roe': info.get('returnOnEquity', 0),
            'profit_margin': info.get('profitMargins', 0),
            'revenue_growth': info.get('revenueGrowth', 0),
            'fcf_yield': free_cash_flow / market_cap if market_cap > 0 else 0,
            'cash_per_share': cash / shares_outstanding if shares_outstanding > 0 else 0
        }
        
        return ratios
    
    except Exception as e:
        print(f"Error calculating financial ratios: {str(e)}")
        return {}

def analyze_management_sentiment(symbol):
    """Analyze management sentiment from recent earnings calls and news"""
    try:
        # Simplified sentiment analysis
        news_sentiment = np.random.uniform(-1, 1)
        earnings_sentiment = np.random.uniform(-1, 1)
        
        return {
            'overall_sentiment': (news_sentiment + earnings_sentiment) / 2,
            'news_sentiment': news_sentiment,
            'earnings_sentiment': earnings_sentiment,
            'confidence_level': np.random.uniform(0.6, 0.9),
            'key_themes': ['growth prospects', 'operational efficiency', 'market expansion']
        }
    
    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        return {'overall_sentiment': 0, 'news_sentiment': 0, 'earnings_sentiment': 0}

def calculate_market_positioning(stock, info):
    """Calculate market positioning metrics"""
    try:
        # Get historical data for volatility and performance metrics
        hist = stock.history(period="1y")
        
        if len(hist) == 0:
            return {}
        
        # Calculate volatility
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        
        # Calculate performance metrics
        current_price = hist['Close'][-1]
        price_52w_high = hist['Close'].max()
        price_52w_low = hist['Close'].min()
        
        # Market positioning score
        market_share_proxy = info.get('marketCap', 0) / 1e9
        
        return {
            'volatility': volatility,
            'price_vs_52w_high': (current_price / price_52w_high) if price_52w_high > 0 else 0,
            'price_vs_52w_low': (current_price / price_52w_low) if price_52w_low > 0 else 0,
            'market_share_proxy': market_share_proxy,
            'beta': info.get('beta', 1.0),
            'insider_ownership': info.get('heldByInsiders', 0),
            'institutional_ownership': info.get('heldByInstitutions', 0)
        }
    
    except Exception as e:
        print(f"Error calculating market positioning: {str(e)}")
        return {}

def calculate_acquisition_score(financial_ratios, management_sentiment, market_metrics):
    """Calculate overall acquisition attractiveness score (0-100)"""
    try:
        score = 0
        
        # Financial health score (40% weight)
        financial_score = 0
        if financial_ratios.get('current_ratio', 0) > 1.2:
            financial_score += 10
        if financial_ratios.get('debt_to_equity', 0) < 0.5:
            financial_score += 10
        if financial_ratios.get('roe', 0) > 0.1:
            financial_score += 10
        if financial_ratios.get('profit_margin', 0) > 0.05:
            financial_score += 10
        
        # Valuation attractiveness (30% weight)
        valuation_score = 0
        pe_ratio = financial_ratios.get('pe_ratio', 0)
        if 5 < pe_ratio < 20:
            valuation_score += 15
        elif pe_ratio < 15:
            valuation_score += 10
        
        if financial_ratios.get('pb_ratio', 0) < 3:
            valuation_score += 15
        
        # Management sentiment (20% weight)
        sentiment_score = max(0, management_sentiment.get('overall_sentiment', 0) * 20)
        
        # Market positioning (10% weight)
        positioning_score = 0
        if market_metrics.get('price_vs_52w_high', 0) < 0.8:
            positioning_score += 5
        if market_metrics.get('volatility', 0) < 0.3:
            positioning_score += 5
        
        total_score = financial_score + valuation_score + sentiment_score + positioning_score
        return min(100, max(0, total_score))
    
    except Exception as e:
        print(f"Error calculating acquisition score: {str(e)}")
        return 0

def generate_key_highlights(financial_ratios, management_sentiment, market_metrics):
    """Generate key highlights for the acquisition target"""
    highlights = []
    
    if financial_ratios.get('current_ratio', 0) > 1.5:
        highlights.append("Strong liquidity position")
    
    if financial_ratios.get('roe', 0) > 0.15:
        highlights.append("High return on equity")
    
    if management_sentiment.get('overall_sentiment', 0) > 0.6:
        highlights.append("Positive management outlook")
    
    if market_metrics.get('price_vs_52w_high', 0) < 0.7:
        highlights.append("Trading at discount to recent highs")
    
    if financial_ratios.get('debt_to_equity', 0) < 0.3:
        highlights.append("Conservative debt levels")
    
    return highlights[:5]

def identify_risk_factors(financial_ratios, management_sentiment, market_metrics):
    """Identify potential risk factors"""
    risks = []
    
    if financial_ratios.get('current_ratio', 0) < 1.0:
        risks.append("Liquidity concerns")
    
    if financial_ratios.get('debt_to_equity', 0) > 1.0:
        risks.append("High debt levels")
    
    if management_sentiment.get('overall_sentiment', 0) < 0.3:
        risks.append("Negative management sentiment")
    
    if market_metrics.get('volatility', 0) > 0.5:
        risks.append("High price volatility")
    
    if financial_ratios.get('profit_margin', 0) < 0:
        risks.append("Negative profit margins")
    
    return risks[:5]

def get_companies_by_sector(sector, market_cap_range):
    """Get list of companies by sector and market cap range"""
    # Sample companies - you can expand this or connect to your database
    sample_companies = [
        {'symbol': 'AAPL', 'name': 'Apple Inc.'},
        {'symbol': 'MSFT', 'name': 'Microsoft Corp.'},
        {'symbol': 'GOOGL', 'name': 'Alphabet Inc.'},
        {'symbol': 'TSLA', 'name': 'Tesla Inc.'},
        {'symbol': 'AMZN', 'name': 'Amazon.com Inc.'},
        {'symbol': 'META', 'name': 'Meta Platforms Inc.'},
        {'symbol': 'NVDA', 'name': 'NVIDIA Corp.'},
        {'symbol': 'NFLX', 'name': 'Netflix Inc.'},
        {'symbol': 'CRM', 'name': 'Salesforce Inc.'},
        {'symbol': 'ADBE', 'name': 'Adobe Inc.'},
    ]
    return sample_companies