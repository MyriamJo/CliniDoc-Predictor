from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.upload_view, name='predict'),
    path('', views.upload_view, name='predict'),
]