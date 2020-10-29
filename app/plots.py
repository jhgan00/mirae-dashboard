from plotly.offline import plot
import plotly.express as px
import pandas as pd
import os
import io
import urllib
import base64
import matplotlib.pyplot as plt
from Prediction.apps import PredictionConfig
import shap

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


def plot_classification(classification):
    """
    받아야되는거? 이번 달 데이터
    :return:
    """

    fig = px.pie(
        classification,
        values = "cnt",
        names = "index",
        color = "index",
        color_discrete_map={"자동지급": "#5cb85c", "조사": "#d9534f", "심사": "#f0ad4e"},
        template = "plotly_dark"
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend_title=None,
    )

    fig.update_traces(
        textfont=dict(size=17, color="white"),
        textinfo='label+percent+value'
    )

    fig.update_xaxes(title=None)
    div = plot(fig, output_type="div", auto_open=False, config=dict(displayModeBar=False))
    return div


def plot_force(data):

    dropcols = ["ID","자동지급", "심사", "조사", "conf", "pred", "target", "sampling_method", "base_ym"]

    labels = dict(자동지급=0, 심사=1, 조사=2)
    pred = labels[data["pred"]]
    X = data.drop(dropcols)

    if X.보험료구간 == 99:
        X = X.drop(["가입금액구간", "보험료구간"])
        explainer = PredictionConfig.explainer_na

    else:
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
    # performance: 데이터프레임이 그대로 넘어온 상태
    performance = performance.assign(
        month = pd.to_datetime(performance.base_ym_id, format="%Y%m")
    )
    performance = performance.set_index("month").drop(["id","base_ym_id"], axis=1).stack().reset_index()
    performance.columns = ["month", "performance", "value"]

    fig = px.line(
        performance,
        x="month",
        y="value",
        color="performance",
        template="plotly_dark"
    )

    fig.data[0].update(mode='markers+lines')
    fig.data[1].update(mode='markers+lines')

    fig.update_layout(
        title_font_size=20,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend_title=None,
        xaxis_tickformat = '%b<br>%Y'
    )

    fig.update_traces(line=dict(width=5), marker=dict(size=15))
    fig.update_xaxes(title="MONTH", showgrid=False)
    fig.update_yaxes(title="", range = [0.6, 0.9], showgrid=True)
    div = plot(fig, output_type="div", auto_open=False, config=dict(displayModeBar=False))

    return div