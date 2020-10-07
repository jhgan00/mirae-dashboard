# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator
from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required
from django.template import loader
from django.http import HttpResponse, JsonResponse
from django.views.generic import ListView, DetailView, View
from rest_framework.response import Response
from rest_framework.views import APIView
from Prediction.serializers import InsuranceClaimSerializer
from Prediction.views import InsuranceClaimPredict
from app.models import InsuranceClaim
from app.plots import plot_class_prob, plot_local_exp


@login_required(login_url="/login/")
def index(request):
    context = {}
    context['segment'] = 'index'
    html_template = loader.get_template('index.html')
    return HttpResponse(html_template.render(context, request))


class InsuranceClaimCV(APIView):
    def post(self, request, format=None):
        InsuranceClaimPredict().post(request)
        return Response(status=200)


class InsuranceClaimLV(ListView):
    model = InsuranceClaim
    template_name = "tables.html"
    context_object_name = "claims"
    paginate_by = 10

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        queryset = InsuranceClaim.objects.all()
        for label in ["자동지급", "심사", "조사", None]:
            cnt = queryset.filter(target=label).count()
            label_str = f"n_{label}"
            context[label_str] = cnt
        return context


class InsuranceClaimRedirection(View):
    def get(self, request):
        latest = InsuranceClaim.objects.latest("ID")
        dst = f"/details/{latest.ID}"
        return redirect(dst)


class InsuranceClaimDV(DetailView):
    model = InsuranceClaim
    template_name = "details.html"

    @csrf_exempt
    def post(self, request, **kwargs):
        obj = super().get_object()
        paginator = Paginator(obj.get_fields(), 5)
        page_n = self.request.POST.get("page_n", None)
        results = {key: val for key, val in paginator.page(page_n).object_list}
        return JsonResponse({"results": results})

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        claim_obj = context['object']
        lime_obj = claim_obj.claim.all()
        data = InsuranceClaimSerializer(claim_obj).data
        paginator = Paginator(list(data.items()), 5)

        context["page_n"] = 1
        context["paginator"] = paginator
        context["first_page"] = paginator.page(1).object_list
        context["page_range"] = paginator.page_range

        labels = ["자동지급", "심사", "조사"]
        prob = [data[label] for label in labels]
        label = labels[prob.index(max(prob))]
        context["label"] = label
        context["prob"] = max(prob)
        # class prob plot
        context["class_prob_plot"] = plot_class_prob(labels, prob)
        # local explanation plot
        context["local_exp_plot"] = plot_local_exp(lime_obj.values())
        return context
