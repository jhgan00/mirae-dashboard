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
from app.models import InsuranceClaim
from app.plots import plot_class_prob, plot_local_exp, plot_threshold
from app.forms import CostForm
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


@login_required(login_url="/login/")
def profile(request):
    context = {}
    context['segment'] = 'profile'
    html_template = loader.get_template('profile.html')
    return HttpResponse(html_template.render(context, request))


class HomeView(TemplateView):
    template_name = "index.html"


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

            if prv_cnt > cnt:
                sign = "fa-arrow-down"

            else:
                sign = "fa-arrow-up"

            try:
                inc = cnt/prv_cnt - 1

            except ZeroDivisionError:
                inc = 0

            if cnt > 1000:
                cnt = str(cnt * 0.001).replace(".", ",")

            context[f"cnt_{label}"] = cnt
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

        # lime data
        lime_obj = claim_obj.lime.all()
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
        context["local_exp_plot"] = plot_local_exp(lime_obj.values())
        return context
