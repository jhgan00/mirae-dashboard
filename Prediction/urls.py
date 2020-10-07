from django.urls import path
import Prediction.views as views

urlpatterns = [
    path("predict/", views.InsuranceClaimPredict.as_view(), name="predict")
]