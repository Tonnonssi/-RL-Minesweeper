# game/urls.py
from django.urls import path
from .views import update_board

urlpatterns = [
    path('update_board/', update_board, name='update_board')
]
