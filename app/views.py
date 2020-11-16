# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator
from django.shortcuts import redirect, render
from django.contrib.auth.decorators import login_required
from django.template import loader
from django.http import HttpResponse, JsonResponse, HttpRequest
from django.views.generic import ListView, DetailView, View, TemplateView

from rest_framework.response import Response
from rest_framework.views import APIView
from Prediction.serializers import InsuranceClaimSerializer
from Prediction.views import InsuranceClaimPredict
from app.models import InsuranceClaim, PredictionPerformance
from app.plots import plot_class_prob, plot_force, plot_performance, plot_classification
from app.forms import InsuranceClaimForm, InsuranceClaimUpdateForm
import os
import numpy as np
import pandas as pd

from django.contrib.auth.mixins import LoginRequiredMixin

mapper = {'nur_hosp_yn': '요양병원여부',
 'ac_ctr_diff': '청구일계약일기간',
 'hsp_avg_optt_bilg_isamt_s': '병원별평균통원청구보험금',
 'hsp_avg_surop_bilg_isamt_s': '병원별평균수술청구보험금',
 'ar_rclss_cd': '발생지역',
 'fds_cust_yn': '보험사기이력고객여부',
 'hspz_dys_s': '입원청구건수',
 'inamt_nvcd': '가입금액구간',
 'hsp_avg_diag_bilg_isamt_s': '병원별평균진단청구보험금',
 'blrs_cd': '치료행위',
 'dsas_ltwt_gcd': '질병경중등급',
 'dsas_avg_diag_bilg_isamt_s': '질병별평균진단청구보험금',
 'dsas_acd_rst_dcd': '질병구분',
 'kcd_gcd': 'KCD등급',
 'hsp_avg_hspz_bilg_isamt_s': '병원별평균입원청구보험금',
 'optt_blcnt_s': '통원횟수',
 'mtad_cntr_yn': '중도부가계약여부',
 'heltp_pf_ntyn': '건강인우대계약여부',
 'prm_nvcd': '보험료구간',
 'surop_blcnt_s': '수술청구건수',
 'mdct_inu_rclss_dcd': '의료기관구분',
 'dsas_avg_optt_bilg_isamt_s': '질병별평균통원청구보험금',
 'isrd_age_dcd': '고객연령',
 'hspz_blcnt_s': '입원청구건수',
 'dsas_avg_surop_bilg_isamt_s': '질병별평균수술청구보험금',
 'urlb_fc_yn': '부실판매자계약여부',
 'dsas_avg_hspz_bilg_isamt_s': '질병별평균입원청구보험금',
 'smrtg_5y_passed_yn': '보담보5년경과여부',
 'ac_rst_diff': '청구일부활일기간',
 'bilg_isamt_s': '청구보험금',
 'optt_nbtm_s': '통원횟수'}




@login_required(login_url="/login/")
def profile(request):
    context = {}
    context['segment'] = 'profile'
    html_template = loader.get_template('profile.html')
    return HttpResponse(html_template.render(context, request))


class HomeView(LoginRequiredMixin,ListView):
    login_url = '/login/'
    template_name = "index.html"
    model = InsuranceClaim

    def get_context_data(self, **kwargs):
        context = super().get_context_data()
        claims = self.model.objects.filter(base_ym=os.environ["base_ym"])
        prev_claims = self.model.objects.filter(base_ym=int(os.environ["base_ym"])-1).count()
        n_claims = claims.count()
        inc_claims = round((n_claims/prev_claims - 1) * 100, 2)

        n_automation = self.model.objects \
            .filter(base_ym=os.environ["base_ym"])\
            .filter(sampling_method="automation").count()
        automation = round((n_automation / n_claims) * 100, 2)

        if inc_claims < 1: sign_claim = "fa-arrow-down"
        else: sign_claim = "fa-arrow-up"

        performance = pd.DataFrame(PredictionPerformance.objects.values())
        performance_plot = plot_performance(performance)
        prev_performance = performance.query(f"base_ym=={int(os.environ['base_ym'])-1}").performance.values[0]
        performance = performance.query(f"base_ym=={os.environ['base_ym']}").performance.values[0]
        inc_performance = round((performance/prev_performance - 1) * 100, 2)
        if inc_performance < 1: sign_performance = "fa-arrow-down"
        else: sign_performance = "fa-arrow-up"

        classification = pd.DataFrame(claims.values()).target.value_counts().rename("cnt").reset_index()
        classification_plot = plot_classification(classification)


        context["performance_plot"] = performance_plot
        context["threshold"] = os.environ["THRESHOLD"]

        context["total_claim"] = f'{n_claims:,}'
        context["inc_claim"] = abs(inc_claims)
        context["sign_claim"] = sign_claim
        context["automation"] = automation

        context["automation"] = automation
        # context["inc_automation"] = inc_automation
        # context["sign_automaion"] = sign_automaion

        context["performance"] = performance
        context["inc_performance"] = np.abs(inc_performance)
        context["sign_performance"] = sign_performance

        context["classification_plot"] = classification_plot

        return context


class TestView(LoginRequiredMixin,TemplateView):
    login_url = '/login/'
    template_name = "test.html"

    def get(self, request, *args, **kwargs):
        form = InsuranceClaimForm()
        data = dict(form=form)
        return render(request, self.template_name, data)

    @csrf_exempt
    def post(self, request):
        if "create" in request.POST:
            form = InsuranceClaimForm(request.POST)
            assert form.is_valid()
            data = form.cleaned_data

            request = HttpRequest()
            request.method = "POST"
            request.data = form.cleaned_data
            InsuranceClaimPredict().post(request)
            # 만약 폼이 적절하면 Prediction API로 보내서 결과 받아오기

            return redirect(f'/details/{data["ID"]}')


class InsuranceClaimCV(LoginRequiredMixin, APIView):
    login_url = '/login/'
    def post(self, request):
        InsuranceClaimPredict().post(request)
        return Response(status=200)


class InsuranceClaimLV(LoginRequiredMixin, ListView):
    login_url = '/login/'
    model = InsuranceClaim
    template_name = "tables.html"
    context_object_name = "claims"
    paginate_by = 5

    def get_queryset(self):
        return InsuranceClaim.objects.filter(target__isnull=True)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        queryset = InsuranceClaim.objects.all()
        for label in ["자동지급", "심사", "조사"]:
            prv_cnt = queryset.filter(target=label).filter(base_ym=int(os.environ["base_ym"])-1).count()
            cnt = queryset.filter(target=label).filter(base_ym=os.environ["base_ym"]).count()

            if prv_cnt > cnt: sign = "fa-arrow-down"
            else: sign = "fa-arrow-up"

            try: inc = cnt/prv_cnt - 1
            except ZeroDivisionError: inc = 0

            context[f"cnt_{label}"] = f'{cnt:,}'
            context[f"inc_{label}"] = np.abs(round(inc * 100, 2))
            context[f"sign_{label}"] = sign

        unclassified = queryset.filter(target__isnull=True).count()
        context["cnt_None"] = unclassified

        context["unclassified_list"] = queryset.filter(target__isnull=True)
        return context


class InsuranceClaimRedirection(LoginRequiredMixin, View):
    login_url = '/login/'
    def get(self, request):
        latest = InsuranceClaim.objects.latest("base_ym")
        dst = f"/details/{latest.ID}"
        return redirect(dst)


class InsuranceClaimDV(LoginRequiredMixin, DetailView):
    login_url = '/login/'
    redirect_field_name = 'details'

    model = InsuranceClaim
    template_name = "details.html"

    @csrf_exempt
    def post(self, request, **kwargs):

        if 'update' in request.POST:
            claim = self.get_object()
            ID = claim.ID

            form = InsuranceClaimUpdateForm(request.POST, instance=claim)
            assert form.is_valid()
            claim = form.save(commit=False)
            claim.save(update_fields=list(form.fields))

            return redirect(f"/details/{ID}")

        else:
            print(request.POST)

            obj = super().get_object()
            paginator = Paginator(obj.get_fields(), 5)
            page_n = self.request.POST.get("page_n", None)
            results = {key: val for key, val in paginator.page(page_n).object_list}
            return JsonResponse({"results": results})

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # claim data
        claim_obj = context['object']
        claim = InsuranceClaimSerializer(claim_obj).data
        data = pd.Series(claim)
        data.index = data.index.to_series().replace(mapper)
        claim = data.to_dict()

        # lime data
        labels = ["자동지급", "심사", "조사"]
        context['labels'] = labels

        # init datatable pagination
        paginator = Paginator(list(claim.items()), 5)
        context["page_n"] = 1
        context["paginator"] = paginator
        context["first_page"] = paginator.page(1).object_list
        context["page_range"] = paginator.page_range

        #
        prob = [claim[label] for label in labels]
        label = labels[prob.index(max(prob))]
        context["label"] = label
        context["prob"] = max(prob)

        # plots
        context["class_prob_plot"] = plot_class_prob(labels, prob)
        context["force_plot"] = plot_force(data)
        context['form'] = InsuranceClaimUpdateForm()


        return context