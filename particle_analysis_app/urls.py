from django.urls import path
from . import views

urlpatterns = [
    path('', views.particle_analysis, name='particle_analysis'),
    path('map_contributions', views.map_contributions, name='map_contributions'),
]