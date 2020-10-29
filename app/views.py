# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator
from django.shortcuts import redirect, render
from django.contrib.auth.decorators import login_required
from django.template import loader
from django.http import HttpResponse, JsonResponse
from django.views.generic import ListView, DetailView, View, TemplateView
from rest_framework.response import Response
from rest_framework.views import APIView
from Prediction.serializers import InsuranceClaimSerializer
from Prediction.views import InsuranceClaimPredict
from Prediction.apps import PredictionConfig
from app.models import InsuranceClaim, PredictionPerformance
from app.plots import plot_class_prob, plot_force, plot_threshold, plot_performance, plot_feature_importance
from app.forms import CostForm
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

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


class HomeView(ListView):
    template_name = "index.html"
    model = InsuranceClaim

    def get_context_data(self, **kwargs):
        context = super().get_context_data()
        prev_claims = self.model.objects.filter(base_ym=int(os.environ["base_ym"])-1).count()
        n_claims = self.model.objects.filter(base_ym=os.environ["base_ym"]).count()
        inc_claims = round((n_claims/prev_claims - 1) * 100, 2)

        n_automation = self.model.objects \
            .filter(base_ym=os.environ["base_ym"])\
            .filter(sampling_method="automation").count()
        automation = round((n_automation / n_claims) * 100, 2)

        if inc_claims < 1: sign_claim = "fa-arrow-down"
        else: sign_claim = "fa-arrow-up"

        performance = pd.DataFrame(PredictionPerformance.objects.values())
        performance_plot = plot_performance(performance)
        prev_performance = performance.query(f"base_ym_id=={int(os.environ['base_ym'])-1}").performance.values[0]
        performance = performance.query(f"base_ym_id=={os.environ['base_ym']}").performance.values[0]
        inc_performance = round((performance/prev_performance - 1) * 100, 2)
        if inc_performance < 1: sign_performance = "fa-arrow-down"
        else: sign_performance = "fa-arrow-up"

        normal = PredictionConfig.classifier_normal
        na = PredictionConfig.classifier_na

        fi = pd.concat([
            pd.DataFrame(
                list(zip(normal.feature_name_, normal.feature_importances_)),
                columns=["normal_feature", "normal_importance"]
            ).sort_values("normal_importance", ascending=False).head(5).assign(rank=np.arange(1,6)).reset_index(drop=True),
            pd.DataFrame(
                list(zip(na.feature_name_, na.feature_importances_)),
                columns=["na_feature", "na_importance"]
            ).sort_values("na_importance", ascending=False).head(5).assign(rank=np.arange(1,6)).reset_index(drop=True)
        ], axis=1).to_dict("records")

        context["feature_importances"] = fi

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

        context["importance_plot"] = plot_feature_importance()

        return context


class SettingView(TemplateView):
    template_name = "settings.html"


    def update_threshold(self, cost01, cost02, cost10, cost12, cost21, cost20, manual):

        def eval_threshold(result, thr, error_cost, manual):
            prediction = result.query(f"prob > {thr}")
            confusion = confusion_matrix(prediction.target.values, prediction.prediction.values)

            n_manual = result.query(f"prob <= {thr}").shape[0]

            error_cost = (error_cost * confusion).sum()
            clf_cost = (manual * n_manual)
            total_cost = (error_cost + clf_cost) / 9
            automation = 1 - (n_manual / result.shape[0])
            improvement = 1 - total_cost / base_cost

            return error_cost, clf_cost, total_cost, automation, improvement

        result = pd.DataFrame(InsuranceClaim.objects.filter(base_ym__lte=201910).values())
        result = result.assign(prob=result[["자동지급", "심사", "조사"]].max(axis=1))

        base_cost = (manual * result.shape[0]) / 9

        thrs = np.arange(0.5, 1, 0.01)
        error_cost = np.array(
            [[0, cost01, cost02],
             [cost10, 0, cost12],
             [cost20, cost21, 0]]
        )
        df = pd.DataFrame([
            eval_threshold(result, thr, error_cost, manual) for thr in thrs],
            columns=["error_cost", "clf_cost", "total_cost", "automation", "improvement"]
        ).assign(threshold=thrs, base_cost=base_cost)

        best_idx = df.total_cost.idxmin()

        os.environ["THRESHOLD"] = str(int(df.threshold.values[best_idx] * 100))
        os.environ["ERROR_COST"] = str(round(df.error_cost.values[best_idx]))
        os.environ["CLF_COST"] = str(round(df.clf_cost.values[best_idx] , 2))
        os.environ["TOTAL_COST"] = str(round(df.total_cost.values[best_idx], 2))
        os.environ["AUTOMATION"] = str(round(df.automation.values[best_idx] * 100, 2))
        os.environ["IMPROVEMENT"] = str(round(df.improvement.values[best_idx] * 100, 2))
        os.environ["BASE_COST"] = str(base_cost)

        return df

    def get(self, request, *args, **kwargs):

        cost01 = float(os.environ['COST01'])
        cost02 = float(os.environ['COST02'])
        cost10 = float(os.environ['COST10'])
        cost12 = float(os.environ['COST12'])
        cost20 = float(os.environ['COST20'])
        cost21 = float(os.environ['COST21'])
        manual = float(os.environ['MANUAL'])

        filename = f"{cost01}_{cost02}_{cost10}_{cost12}_{cost20}_{cost21}_{manual}.html"
        fpath = f"app/includes/{filename}"

        if os.path.isfile(fpath):

            with open(fpath, "r") as html:
                threshold_plot = html.read()

        else:
            df = self.update_threshold(cost01, cost02, cost10, cost12, cost21, cost20, manual)
            threshold_plot = plot_threshold(df, fpath)

        form = CostForm(
            initial=dict(cost01=cost01, cost02=cost02, cost10=cost10, cost12=cost12, cost20=cost20, cost21=cost21, manual=manual)
        )
        data = dict(
            form=form,
            threshold_plot=threshold_plot,
            threshold=os.environ["THRESHOLD"],
            improvement=os.environ["IMPROVEMENT"],
            total_cost=os.environ["TOTAL_COST"],
            error_cost=os.environ["ERROR_COST"],
            clf_cost=os.environ["CLF_COST"],
            automation=os.environ["AUTOMATION"]
        )
        return render(request, self.template_name, data)

    @csrf_exempt
    def post(self, request):

        if "cost" in request.POST:

            form = CostForm(request.POST)
            assert form.is_valid()

            data = form.cleaned_data
            cost01 = data['cost01']
            cost02 = data['cost02']
            cost10 = data['cost10']
            cost12 = data['cost12']
            cost20 = data['cost20']
            cost21 = data['cost21']
            manual = data['manual']

            filename = f"{cost01}_{cost02}_{cost10}_{cost12}_{cost20}_{cost21}_{manual}.html"
            fpath = f"app/includes/{filename}"

            if os.path.isfile(fpath):

                with open(fpath, "r") as html:
                    threshold_plot = html.read()

            else:

                df = self.update_threshold(cost01, cost02, cost10, cost12, cost21, cost20, manual)
                threshold_plot = plot_threshold(df, fpath)

            data = dict(
                form=form,
                threshold_plot=threshold_plot,
                threshold=os.environ["THRESHOLD"],
                improvement=os.environ["IMPROVEMENT"],
                total_cost=os.environ["TOTAL_COST"],
                error_cost=os.environ["ERROR_COST"],
                clf_cost=os.environ["CLF_COST"],
                automation=os.environ["AUTOMATION"]
            )

            return render(request, self.template_name, data)


class InsuranceClaimCV(APIView):
    def post(self, request):
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
            prv_cnt = queryset.filter(target=label).filter(base_ym=int(os.environ["base_ym"])-1).count()
            cnt = queryset.filter(target=label).filter(base_ym=os.environ["base_ym"]).count()

            if prv_cnt > cnt: sign = "fa-arrow-down"
            else: sign = "fa-arrow-up"

            try: inc = cnt/prv_cnt - 1
            except ZeroDivisionError: inc = 0

            context[f"cnt_{label}"] = f'{cnt:,}'
            context[f"inc_{label}"] = np.abs(round(inc * 100, 2))
            context[f"sign_{label}"] = sign


        context["unclassified_list"] = queryset.filter(target__isnull=True)
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

        # claim data
        claim_obj = context['object']
        claim = InsuranceClaimSerializer(claim_obj).data
        data = pd.Series(claim)
        data.index = data.index.to_series().replace(mapper)
        claim = data.to_dict()

        # lime data
        labels = ["자동지급", "심사", "조사"]

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
        return context
