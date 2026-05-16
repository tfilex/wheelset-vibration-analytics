import time

import pandas as pd
import streamlit as st

from src.demo.mock_data import (
    DASHBOARD_RUL,
    SIGNAL_TYPES,
    TEST_BEARINGS,
    build_checks_history,
)
from src.prediction.demo_inference import (
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


st.set_page_config(layout="wide", page_title="Система диагностики")


def render_cwru_tab() -> None:
    st.header("Классификация дефектов (CWRU)")

    model_options = discover_classification_models()
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


def render_rul_tab() -> None:
    st.header("Прогноз ресурса RUL (XJTU-SY)")

    model_options = discover_rul_models()
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
    page = st.sidebar.radio(
        "Раздел",
        (
            "Классификация дефектов (CWRU)",
            "Прогноз ресурса RUL (XJTU-SY)",
            "Бортовой модуль (Дашборд)",
        ),
    )

    if page == "Классификация дефектов (CWRU)":
        render_cwru_tab()
    elif page == "Прогноз ресурса RUL (XJTU-SY)":
        render_rul_tab()
    else:
        render_dashboard_tab()


if __name__ == "__main__":
    main()
