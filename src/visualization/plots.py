import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def build_signal_figure(signal: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=signal["time"],
            y=signal["amplitude"],
            mode="lines",
            line={"color": "#1f77b4", "width": 1.4},
            name="Вибросигнал",
        )
    )
    fig.update_layout(
        title="Исходный вибросигнал",
        xaxis_title="Время, усл. ед.",
        yaxis_title="Амплитуда",
        height=420,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
    )
    return fig


def build_attribution_heatmap(heatmap: np.ndarray) -> go.Figure:
    fig = px.imshow(
        heatmap,
        color_continuous_scale="jet",
        aspect="auto",
        labels={"x": "Окно сигнала", "y": "Частотный канал", "color": "Вклад"},
        title="Карта важности модели",
    )
    fig.update_layout(
        height=420,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        coloraxis_colorbar={"title": "Вклад"},
    )
    return fig


def build_rul_figure(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["step"],
            y=history["true_rul"],
            mode="lines",
            name="True RUL",
            line={"color": "#2ca02c", "width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=history["step"],
            y=history["pred_rul"],
            mode="lines+markers",
            name="Pred RUL",
            line={"color": "#d62728", "width": 2},
            marker={"size": 4},
        )
    )
    fig.update_layout(
        title="Симуляция деградации подшипника",
        xaxis_title="Шаг наблюдения",
        yaxis_title="Нормированный RUL",
        yaxis={"range": [-0.05, 1.05]},
        height=500,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        legend={"orientation": "h", "y": 1.08, "x": 0.0},
    )
    return fig


def build_gauge_figure(rul_value: int) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=rul_value,
            number={"suffix": "%", "font": {"size": 54}},
            title={"text": "Остаточный ресурс буксового узла", "font": {"size": 24}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#2f3e46"},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#d9dee3",
                "steps": [
                    {"range": [0, 20], "color": "#ff4b4b"},
                    {"range": [20, 50], "color": "#f6c343"},
                    {"range": [50, 100], "color": "#2fb344"},
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 4},
                    "thickness": 0.75,
                    "value": rul_value,
                },
            },
        )
    )
    fig.update_layout(
        height=430,
        margin={"l": 30, "r": 30, "t": 70, "b": 20},
    )
    return fig
