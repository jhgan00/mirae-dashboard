import plotly
from plotly.offline import plot
import plotly.express as px
import pandas as pd


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
    div = plot(fig, output_type="div", auto_open=False)
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
    div = plot(fig, output_type="div", auto_open=False)
    return div