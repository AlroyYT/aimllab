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