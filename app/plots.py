from plotly.offline import plot
import plotly.express as px
import numpy as np
import pandas as pd
from app.models import InsuranceClaim
from sklearn.metrics import confusion_matrix
import os

def plot_class_prob(labels, prob):
    data = pd.DataFrame(dict(labels=labels, prob=prob))
    fig = px.bar(
        data,
        x="labels",
        y="prob",
        color="labels",
        color_discrete_map={"자동지급": "#5cb85c", "조사": "#d9534f", "심사": "#f0ad4e"},
        category_orders = dict(labels=["자동지급", '심사', "조사"]),
        text=[f"{p}%" for p in prob],
        title="<b>모델의 예측</b>",
        template="plotly_white"
    )
    fig.update_traces(textfont=dict(size=12))
    fig.update_yaxes(title="확률", range=[0, 100])
    fig.update_xaxes(title="분류")
    fig.update_layout(showlegend=False, title_font_size=20)
    div = plot(fig, output_type="div", auto_open=False, config=dict(displayModeBar=False))
    return div


def plot_local_exp(data):
    data = pd.DataFrame(data).sort_values("rank", ascending=True)
    axis_lim = data.local_exp.abs().max() * 1.2
    fig = px.bar(
        data,
        x="local_exp",
        y="feature",
        color=(data.local_exp >= 0).map({True: "Supports", False: "Contradicts"}),
        orientation="h",
        title="<b>LIME 분석</b>",
        category_orders=dict(feature=[x for x in data.feature]),
        template="plotly_white"
    )
    # Lime 방향 표시
    fig.update_traces(textfont=dict(size=14), textposition="outside")
    fig.update_yaxes(title=None, range=[-0.5, 3], showticklabels=True)
    fig.update_xaxes(title="Local Explanation",  range=[-axis_lim, axis_lim])
    fig.update_layout(title_font_size=20, legend=dict(xanchor="right", x=0, yanchor="bottom", y=1, title=None, orientation="h"))
    div = plot(fig, output_type="div", auto_open=False, config=dict(displayModeBar=False))
    return div


def plot_threshold(cost01=3., cost02=3., cost10=3., cost12=3., cost20=3., cost21=3., manual=1.):

    def cost(result, thr, error_cost, manual):
        prediction = result.query(f"prob > {thr}")
        confusion = confusion_matrix(prediction.target.values, prediction.prediction.values)
        n_manual = result.query(f"prob <= {thr}").shape[0]
        total_cost = (error_cost * confusion).sum() + (manual * n_manual)
        mean_cost = total_cost / 9
        return mean_cost

    filename = f"{cost01}_{cost02}_{cost10}_{cost12}_{cost20}_{cost21}_{manual}.html"
    fpath = f"app/includes/{filename}"

    if os.path.isfile(fpath):
        with open(fpath, "r") as html:
            div = html.read()
        return div

    else:

        result = pd.DataFrame(InsuranceClaim.objects.all().values())
        result = result.assign(prob=result[["자동지급", "심사", "조사"]].max(axis=1))
        thrs = np.arange(0.5, 1, 0.01)
        error_cost = np.array(
            [[0, cost01, cost02],
             [cost10, 0, cost12],
             [cost20, cost21, 0]]
        )
        mean_costs = [cost(result, thr, error_cost, manual) for thr in thrs]
        df = pd.DataFrame(dict(threshold=thrs, cost=mean_costs))
        best_idx = df.cost.idxmin()
        best_threshold = round(df.threshold.values[best_idx], 2)
        automation = round((result.prob > best_threshold).sum() / result.shape[0] * 100, 2)

        fig = px.line(
            df,
            x="threshold",
            y="cost",
            template="plotly_dark",
            line_shape='spline',
            title=f"<b>Threshold: {best_threshold} 자동화 비율: {automation}</b>"
        )

        fig.update_layout(
            title_font_size=20,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig.update_traces(line=dict(width=5))
        fig.update_xaxes(title="THRESHOLD", range = [0.45, 1.05], showgrid=False)
        fig.update_yaxes(title="COST", showgrid=True)
        div = plot(fig, output_type="div", auto_open=False, config=dict(displayModeBar=False))

        with open(fpath, "w") as html:
            html.write(div)

        return div