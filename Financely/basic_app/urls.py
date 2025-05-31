from django.urls import path
from basic_app import views

app_name = 'basic_app'

urlpatterns = [
    # Existing URLs
    path('', views.dashboard, name="dashboard"),
    path('home/', views.index, name='index'),
    path('profile/', views.profile, name='profile'),
    path('portfolio/', views.portfolio, name='portfolio'),
    path('home/<str:symbol>/', views.stock, name="stock"),
    path('home/<str:symbol>/price_prediction/', views.price_prediction, name="prediction"),
    path('login/', views.loginPage, name="login"),
    path('logout/', views.logoutUser, name="logout"),
    path('register/', views.registerPage, name="register"),
    path('stats/', views.statisticsAdmin, name="stats"),
    path('home/<str:symbol>/add/', views.addToPortfolio, name="addToPortfolio"),
    path('portfolio/<str:symbol>/remove/', views.removeFromPortfolio, name="removeFromPortfolio"),
    path('portfolio/<str:symbol>/quantityAdd/', views.quantityAdd, name="quantityAdd"),
    path('portfolio/<str:symbol>/quantitySub/', views.quantitySub, name="quantitySub"),
    path('search/', views.search_stock, name='search_stock'),
    
    # Chatbot URLs
    path('chatbot/', views.chatbot_home, name='chatbot_home'),
    
    # API endpoints
    path('api/chat/', views.chat_with_ai, name='chat_with_ai'),
    path('api/analysis/', views.financial_analysis, name='financial_analysis'),
    path('api/tips/', views.get_financial_tips, name='financial_tips'),
    path('api/test/', views.test_ai_connection, name='test_ai_connection'),
]