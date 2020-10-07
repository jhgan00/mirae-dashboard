from plotly.offline import plot
from plotly.graph_objs import Scatter,Layout,Bar
import pandas as pd


def plot_class_prob(labels, prob):
    div = plot({
        "data": [Bar(x=labels, y=prob)],
        "layout": Layout(title="<b>Class Probability</b>")
    }, output_type="div", auto_open=False
    )
    return div


def plot_local_exp(data):
    data = pd.DataFrame(data).sort_values("rank", ascending=False)
    div = plot({
        "data": [Bar(x=data.local_exp, y=data.discretized, orientation="h")],
        "layout": Layout(title="<b>Local Explanation</b>")
    }, output_type="div", auto_open=False)
    return div