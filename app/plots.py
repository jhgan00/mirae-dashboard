from plotly.offline import plot
import plotly.express as px
import pandas as pd
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