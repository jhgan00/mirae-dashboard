from plotly.offline import plot
import plotly.express as px
import pandas as pd
import os
import io
import urllib
import base64
import matplotlib.pyplot as plt
from lightgbm import plot_importance as _plot_importance
from Prediction.apps import PredictionConfig
import shap

# plt.style.use("seaborn-whitegrid")
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams['axes.unicode_minus'] = False



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
        template="plotly_white"
    )
    fig.update_traces(textfont=dict(size=12))
    fig.update_yaxes(title="확률", range=[0, 100])
    fig.update_xaxes(title=None)
    fig.update_layout(showlegend=False, title_font_size=20, autosize=False, height=340)
    div = plot(fig, output_type="div", auto_open=False, config=dict(displayModeBar=False))



    return div


def plot_force(data):
    """
    하나의 데이터에 대해서 shape val 뽑고 모델의 prediction에 대응하는 분석 보여주기
    :param data:
    :return:
    """
    dropcols = ["ID","자동지급", "심사", "조사", "conf", "pred", "target", "sampling_method", "base_ym"]

    labels = dict(자동지급=0, 조사=1, 심사=2)
    pred = labels[data["pred"]]
    X = data.drop(dropcols)

    if X.보험료구간 == 99:
        print("if")
        X = X.drop(["가입금액구간", "보험료구간"])
        explainer = PredictionConfig.explainer_na

    else:
        print("else")
        explainer = PredictionConfig.explainer_normal



    shap_values = explainer.shap_values(X.values.reshape((1,-1)))
    fplot = shap.force_plot(
        explainer.expected_value[pred],
        shap_values[pred],
        X.values.reshape((1,-1)),
        feature_names = X.index,
        matplotlib=True, show=False
    )

    buffer = io.BytesIO()
    fplot.savefig(buffer, bbox_inches="tight", format="png")
    buffer.seek(0)
    string = base64.b64encode(buffer.read())
    uri = urllib.parse.quote(string)
    return uri


def plot_threshold(df, fpath):

    df = df.assign(base_cost=float(os.environ["BASE_COST"]), automation="AUTOMATION", base="BASE")

    fig = px.line(
        df,
        x="threshold",
        y="total_cost",
        template="plotly_dark",
        line_shape='spline',
        color="automation"
    )

    fig.update_layout(
        title_font_size=20,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend_title=None
    )

    fig2 = px.line(
        df,
        x="threshold",
        y="base_cost",
        color="base"
    )
    fig2.update_traces(line=dict(color="#FF0000", dash="dash"))

    fig.add_trace(fig2.data[0])
    fig.update_traces(line=dict(width=5))
    fig.update_xaxes(title="THRESHOLD", range = [0.45, 1.05], showgrid=False)
    fig.update_yaxes(title="COST", showgrid=True)
    div = plot(fig, output_type="div", auto_open=False, config=dict(displayModeBar=False))

    with open(fpath, "w") as html:
        html.write(div)

    return div


def plot_performance(performance):

    performance = performance.assign(
        month = pd.to_datetime(performance.base_ym_id, format="%Y%m")
    )

    fig = px.line(
        performance,
        x="month",
        y="performance",
        template="plotly_dark",
    )

    fig.data[0].update(mode='markers+lines')

    fig.update_layout(
        title_font_size=20,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend_title=None,
        xaxis_tickformat = '%b<br>%Y'
    )

    fig.update_traces(line=dict(width=5), marker=dict(size=15))
    fig.update_xaxes(title="MONTH", showgrid=False)
    fig.update_yaxes(title="F1 SCORE", range = [0.6, 0.8], showgrid=True)
    div = plot(fig, output_type="div", auto_open=False, config=dict(displayModeBar=False))

    return div


def plot_feature_importance():
    normal = PredictionConfig.classifier_normal
    na = PredictionConfig.classifier_na

    fig = plt.figure(figsize=(10, 20))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    _plot_importance(normal, ax=ax1, max_num_features=5)
    _plot_importance(na, ax=ax2, max_num_features=5)

    plt.tight_layout()
    fig = plt.gcf()
    buffer = io.BytesIO()
    fig.savefig(buffer,format="png")
    buffer.seek(0)
    string = base64.b64encode(buffer.read())
    uri = urllib.parse.quote(string)
    return uri