# from django.urls import path
# from . import views
#
# from django.contrib import admin
# from django.urls import path, include
#
# urlpatterns = [
#     path('predict/<str:reservoir_name>/<str:date>/', views.reservoir_detail, name='reservoir_detail'),
#     path('admin/', admin.site.urls),
#     path('', include('dashboard.urls')),  # This includes the URLs from your app
# ]

from django.contrib import admin
from django.urls import path
from . import views
from .views import reservoir_predictions, reservoir_detail, reservoir_data_api

urlpatterns = [
    path('', views.home, name='home'),  # Default home page
    path('predict/<str:stn_id>/', reservoir_detail, name='reservoir_detail'),
    path('admin/', admin.site.urls),
    path('reservoir_predictions/', reservoir_predictions, name='reservoir_predictions'),
    path('api/reservoir_data/<int:reservoir_id>/', reservoir_data_api, name='reservoir_data_api'),
]
