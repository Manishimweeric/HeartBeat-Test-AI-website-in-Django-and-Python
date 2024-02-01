from django.urls import path
from . import views
urlpatterns = [
    path('', views.heart_predictor, name='heart_predict'),
]
