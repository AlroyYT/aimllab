from django.shortcuts import render, redirect
from .models import Portfolio, Client, Stock
from .forms import CreateUserForm
from .sectorPerformance import sectorPerformance
from .decorators import unauthenticated_user, allowed_users
from .ProphetTrend import forecast
from basic_app.stock_data import candlestick_data, get_data, get_name, get_price
from basic_app.FA import piotroski
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
import google.generativeai as genai
from django.views.decorators.http import require_http_methods
from django.conf import settings


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

def price_prediction(request, symbol):
    price_prediction, metrics = forecast(symbol)
    
    if price_prediction is None:
        # Handle error case
        return render(request, "basic_app/error.html", {'error_message': f"Unable to retrieve data for {symbol}"})
    
    return render(request, "basic_app/price_prediction.html", {
        'price_prediction': price_prediction, 
        'metrics': metrics,
        'page_title': f"{symbol} Forecast"
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