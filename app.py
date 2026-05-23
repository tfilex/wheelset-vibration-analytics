import os
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.demo.mock_data import (
    DASHBOARD_RUL,
    SIGNAL_TYPES,
    TEST_BEARINGS,
    build_checks_history,
)
from src.prediction.demo_inference import (
    ModelCatalogMode,
    classify_signal,
    discover_classification_models,
    discover_rul_models,
    get_checkpoint_fingerprint,
    load_cnn_lstm_rul_model,
    load_resnet18_classifier,
    predict_rul_series,
)
from src.visualization.plots import (
    build_attribution_heatmap,
    build_gauge_figure,
    build_rul_figure,
    build_signal_figure,
)
from src.utils.health_index import (
    compute_hi,
    compute_slope,
    format_rul_display,
    get_status,
    rul_to_km,
)
from src.utils.thresholds import speed_threshold


st.set_page_config(layout="wide", page_title="Система диагностики")

MODEL_MODE_LABELS: dict[ModelCatalogMode, str] = {
    "demo": "Prod",
    "experimental": "Test",
}
MODEL_MODE_VALUES: tuple[ModelCatalogMode, ...] = ("demo", "experimental")


def get_env_model_mode() -> ModelCatalogMode:
    raw_mode = os.getenv("MODEL_CATALOG_MODE", "demo").strip().lower()
    if raw_mode in MODEL_MODE_VALUES:
        return raw_mode
    return "demo"


def is_model_mode_locked() -> bool:
    raw_value = os.getenv("MODEL_CATALOG_LOCKED", "").strip().lower()
    return raw_value in {"1", "true", "yes", "on"}


def render_cwru_tab(model_mode: ModelCatalogMode) -> None:
    st.header("Классификация дефектов (CWRU)")

    model_options = discover_classification_models(model_mode)
    if not model_options:
        st.error("Не найдены совместимые checkpoint для классификации CWRU.")
        return

    model_labels = {option["path"]: option["label"] for option in model_options}
    selected_model = st.selectbox(
        "Выберите модель классификации",
        options=list(model_labels.keys()),
        format_func=lambda path: model_labels[path],
    )
    selected_model_fingerprint = get_checkpoint_fingerprint(selected_model)

    try:
        model = load_resnet18_classifier(selected_model, selected_model_fingerprint)
    except Exception as exc:
        st.error("Не удалось загрузить выбранную модель классификации.")
        st.code(str(exc))
        return

    st.caption(
        f"Реальный инференс модели {model.model_name} "
        f"из checkpoint {model.checkpoint_path}."
    )

    signal_type = st.selectbox(
        "Выберите тестовый сигнал",
        options=SIGNAL_TYPES,
        index=0,
    )

    if st.button("Выполнить диагностику", type="primary"):
        try:
            with st.spinner("Выполняется реальный инференс ResNet-18..."):
                result = classify_signal(
                    signal_type,
                    selected_model,
                    selected_model_fingerprint,
                )
        except Exception as exc:
            st.error("Диагностика не выполнена: ошибка инференса.")
            st.code(str(exc))
            return

        if result["is_normal"]:
            st.success(result["message"])
        else:
            st.error(result["message"])

        st.caption(
            f"Checkpoint: {result['checkpoint_path']} | "
            f"класс: {result['predicted_class_id']} ({result['predicted_class_name']}) | "
            f"уверенность: {result['confidence']:.1%} | "
            f"файл: {result['source_file']}"
        )

        left_col, right_col = st.columns(2)
        with left_col:
            st.plotly_chart(
                build_signal_figure(result["signal"]),
                width="stretch",
            )
        with right_col:
            st.plotly_chart(
                build_attribution_heatmap(result["attribution"]),
                width="stretch",
            )


def render_rul_tab(model_mode: ModelCatalogMode) -> None:
    st.header("Прогноз ресурса RUL (XJTU-SY)")

    model_options = discover_rul_models(model_mode)
    if not model_options:
        st.error("Не найдены совместимые checkpoint для прогноза RUL.")
        return

    model_labels = {option["path"]: option["label"] for option in model_options}
    selected_model = st.selectbox(
        "Выберите модель прогноза RUL",
        options=list(model_labels.keys()),
        format_func=lambda path: model_labels[path],
    )
    selected_model_fingerprint = get_checkpoint_fingerprint(selected_model)

    try:
        model = load_cnn_lstm_rul_model(selected_model, selected_model_fingerprint)
    except Exception as exc:
        st.error("Не удалось загрузить выбранную модель прогноза RUL.")
        st.code(str(exc))
        return

    st.caption(
        f"Реальный инференс гибридной модели CNN + {model.temporal_type.upper()} "
        f"из checkpoint {model.checkpoint_path}."
    )

    bearing = st.selectbox(
        "Выберите подшипник из тестовой выборки",
        options=TEST_BEARINGS,
        index=0,
    )

    if st.button("Запустить симуляцию деградации", type="primary"):
        st.info(
            f"Запущен реальный инференс CNN+{model.temporal_type.upper()} "
            f"для {bearing}."
        )
        placeholder = st.empty()

        try:
            with st.spinner("Считаю RUL по реальным CSV-окнам XJTU-SY..."):
                rul_history, metadata = predict_rul_series(
                    bearing,
                    checkpoint_path=selected_model,
                    checkpoint_fingerprint=selected_model_fingerprint,
                )
        except Exception as exc:
            st.error("Симуляция не выполнена: ошибка инференса RUL.")
            st.code(str(exc))
            return

        st.caption(
            f"Checkpoint: {metadata['checkpoint_path']} | "
            f"модель: {metadata['model_name']} | "
            f"seq_length: {metadata['seq_length']} | "
            f"window: {metadata['window_size']} | "
            f"pipeline: {metadata['data_pipeline']} | "
            f"опорных инференсов: {metadata['anchor_points']} | "
            f"данные: {metadata['bearing_dir']}"
        )

        rows: list[dict[str, float]] = []
        for observation in rul_history.to_dict("records"):
            rows.append(observation)
            placeholder.plotly_chart(
                build_rul_figure(pd.DataFrame(rows)),
                width="stretch",
            )
            time.sleep(0.05)

        st.success("Симуляция завершена.")
        st.subheader("Health Index и диагностический статус")

        hi_series = compute_hi(rul_history["pred_rul"].to_numpy(), window=10)
        hi_now = float(hi_series[-1])
        slope = compute_slope(hi_series, n=20)
        rul_km, sigma_km = rul_to_km(hi_now, slope)
        level, name, color = get_status(hi_now)
        rul_display = format_rul_display(rul_km, sigma_km)

        st.markdown(
            f"<b>Текущий статус:</b> <span style='color:{color}'>{level}. {name}</span>",
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Health Index",
            f"{hi_now:.3f}",
            delta=f"{slope * 100:.2f}% / окно",
            delta_color="inverse",
        )
        col2.metric("Остаточный ресурс", rul_display)
        col3.metric("Уровень", f"{level}. {name}")

        status_data = {
            "Уровень": [
                "1 - Норма",
                "2 - Удовлетворительно",
                "3 - Требует контроля",
                "4 - Аварийное",
            ],
            "Условие HI": [
                "HI > 0.85",
                "0.60 < HI <= 0.85",
                "0.35 < HI <= 0.60",
                "HI <= 0.35",
            ],
            "Действие": [
                "Плановое ТО",
                "Повышенное наблюдение",
                "Внеплановый осмотр",
                "Остановка и замена",
            ],
        }
        df_status = pd.DataFrame(status_data)

        def highlight_current(row):
            if row["Уровень"].startswith(str(level)):
                return ["background-color: #fff3cd; font-weight: bold"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_status.style.apply(highlight_current, axis=1),
            width="stretch",
            hide_index=True,
        )

        fig_hi = go.Figure()
        fig_hi.add_trace(
            go.Scatter(
                x=rul_history["step"],
                y=hi_series,
                name="Health Index",
                line={"color": "#1f77b4", "width": 2.5},
            )
        )
        fig_hi.add_hline(y=0.85, line_dash="dash", line_color="orange")
        fig_hi.add_hline(y=0.60, line_dash="dash", line_color="red")
        fig_hi.add_hline(y=0.35, line_dash="dash", line_color="darkred")
        fig_hi.update_layout(
            title="Динамика Health Index",
            xaxis_title="Шаг наблюдения",
            yaxis_title="HI",
            yaxis={"range": [-0.05, 1.05]},
            height=420,
            margin={"l": 20, "r": 20, "t": 55, "b": 20},
        )
        st.plotly_chart(fig_hi, width="stretch")

        threshold_data = pd.DataFrame(
            {
                "Скорость": ["< 40 км/ч", "40-80 км/ч", "> 80 км/ч"],
                "Порог RMS": [
                    f"{speed_threshold(30):.1f} g",
                    f"{speed_threshold(60):.1f} g",
                    f"{speed_threshold(100):.1f} g",
                ],
            }
        )
        st.markdown("#### Скоростные пороги виброускорения")
        st.dataframe(threshold_data, width="stretch", hide_index=True)


def render_dashboard_tab() -> None:
    st.header("Бортовой модуль (Дашборд)")
    st.caption("Имитация интерфейса машиниста поезда.")

    status_col, defect_col, rul_col = st.columns(3)
    status_col.metric("Статус системы", "WARNING")
    defect_col.metric("Обнаруженный дефект", "Внешнее кольцо")
    rul_col.metric("Остаточный ресурс", f"{DASHBOARD_RUL}%")

    st.plotly_chart(build_gauge_figure(DASHBOARD_RUL), width="stretch")

    st.subheader("История последних проверок")
    st.dataframe(build_checks_history(), width="stretch", hide_index=True)


def main() -> None:
    st.sidebar.title("Навигация")
    default_model_mode = get_env_model_mode()
    if is_model_mode_locked():
        model_mode = default_model_mode
        st.sidebar.caption(f"Режим моделей: {MODEL_MODE_LABELS[model_mode]}")
    else:
        model_mode = st.sidebar.radio(
            "Режим моделей",
            options=list(MODEL_MODE_VALUES),
            index=MODEL_MODE_VALUES.index(default_model_mode),
            format_func=lambda mode: MODEL_MODE_LABELS[mode],
        )
    page = st.sidebar.radio(
        "Раздел",
        (
            "Классификация дефектов (CWRU)",
            "Прогноз ресурса RUL (XJTU-SY)",
            "Бортовой модуль (Дашборд)",
        ),
    )

    if page == "Классификация дефектов (CWRU)":
        render_cwru_tab(model_mode)
    elif page == "Прогноз ресурса RUL (XJTU-SY)":
        render_rul_tab(model_mode)
    else:
        render_dashboard_tab()


if __name__ == "__main__":
    main()
