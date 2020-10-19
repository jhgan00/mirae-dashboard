# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path
from app import views


urlpatterns = [
    # Matches any html file - to be used for gentella
    # Avoid using your .html in your resources.
    # Or create a separate django app.
    # re_path(r'^.*\.html', views.pages, name='pages'),

    # The home page
    path("", views.HomeView.as_view(), name="home"),
    path('settings', views.SettingView.as_view(), name='settings'),
    path('profile', views.profile, name="profile"),

    path("create", views.InsuranceClaimCV.as_view(), name="create"),

    path("tables", views.InsuranceClaimLV.as_view(), name="tables"),
    path("details", views.InsuranceClaimRedirection.as_view()),
    path("details/<int:pk>", views.InsuranceClaimDV.as_view(), name="details")
]
