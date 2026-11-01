"""Sample solution for part 4. Run with: streamlit run wine_app/app.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, TypedDict, cast

import joblib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import streamlit as st
from sklearn import metrics
from sklearn.pipeline import Pipeline


APP_DIR: Final[Path] = Path(__file__).resolve().parent

MODEL_FILENAME: Final[str] = "model.joblib"
CONFIG_FILENAME: Final[str] = "config.json"
TEST_FILENAME: Final[str] = "test.csv"
EXAMPLE_CSV_FILENAME: Final[str] = "wine_example.csv"
PREDICTIONS_CSV_FILENAME: Final[str] = "wine_predictions.csv"

MODEL_PATH: Final[Path] = APP_DIR / MODEL_FILENAME
CONFIG_PATH: Final[Path] = APP_DIR / CONFIG_FILENAME
TEST_PATH: Final[Path] = APP_DIR / TEST_FILENAME

type FloatArray = NDArray[np.float64]

FEATURE_LABELS: Final[dict[str, str]] = {
    "fixed_acidity": "Фиксированная кислотность",
    "volatile_acidity": "Летучая кислотность",
    "citric_acid": "Лимонная кислота",
    "residual_sugar": "Остаточный сахар",
    "chlorides": "Хлориды",
    "free_sulfur_dioxide": "Свободный диоксид серы",
    "total_sulfur_dioxide": "Общий диоксид серы",
    "density": "Плотность",
    "pH": "pH",
    "sulphates": "Сульфаты",
    "alcohol": "Алкоголь",
    "is_red": "Красное вино",
}


class AppConfig(TypedDict):
    threshold: float


def load_config(path: Path) -> AppConfig:
    config = json.loads(path.read_text(encoding="utf-8"))
    threshold = float(config["threshold"])
    if not 0 <= threshold <= 1:
        raise ValueError(f"Порог в {CONFIG_FILENAME} должен быть в диапазоне [0, 1].")
    return {"threshold": threshold}


@st.cache_resource
def load_model(path: Path) -> Pipeline:
    return cast(Pipeline, joblib.load(path))


@st.cache_data
def load_test_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_features(data: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    missing = [name for name in feature_names if name not in data.columns]
    if missing:
        raise ValueError("В CSV не хватает столбцов: " + ", ".join(missing))
    if data.empty:
        raise ValueError("В CSV нет строк с данными.")
    
    try:
        features = data.loc[:, feature_names].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Все признаки должны быть числами. Проверьте значения в CSV.") from exc
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("В признаках есть пропуски или бесконечные значения. Исправьте CSV.")
    if not features["is_red"].isin([0, 1]).all():
        raise ValueError("Столбец is_red должен содержать только 0 (белое) или 1 (красное).")
    return features


def positive_proba(model: Pipeline, features: pd.DataFrame) -> FloatArray:
    # Locate class 1 explicitly instead of relying on the probability column order.
    positive_column = list(model.classes_).index(1)
    return np.asarray(model.predict_proba(features)[:, positive_column], dtype=float)


@st.cache_data
def test_proba(_model: Pipeline, features: pd.DataFrame) -> FloatArray:
    return positive_proba(_model, features)


def predict_batch(
    model: Pipeline, data: pd.DataFrame, feature_names: list[str], threshold: float
) -> pd.DataFrame:
    features = validate_features(data, feature_names)

    result = data.copy()
    result["probability_good"] = positive_proba(model, features)
    result["prediction"] = (result["probability_good"] >= threshold).astype(int)
    result["decision"] = np.where(result["prediction"] == 1, "В подборку", "Не в подборку")
    return result


def show_single_wine(model: Pipeline, test_features: pd.DataFrame, threshold: float) -> None:
    st.subheader("Оценка одного вина")
    st.caption("Диапазоны и начальные значения - минимум, максимум и медиана test.csv.")

    values: dict[str, float] = {}
    columns = st.columns(3)
    for index, name in enumerate(test_features.columns):
        if name == "is_red":
            continue

        low = float(test_features[name].min())
        high = float(test_features[name].max())
        median = float(test_features[name].median())
        with columns[index % 3]:
            values[name] = st.number_input(
                FEATURE_LABELS.get(name, name),
                min_value=low,
                max_value=high,
                value=median,
                step=0.00001 if name == "density" else 0.001,
                format="%.5f" if name == "density" else "%.3f",
                help=f"{name}; диапазон в тестовых данных: {low:g}–{high:g}",
                key=f"wine_{name}",
            )
            
    values["is_red"] = int(st.toggle("Красное вино", value=bool(test_features["is_red"].median())))
    wine = pd.DataFrame([values], columns=test_features.columns)

    probability = float(positive_proba(model, wine)[0])
    st.metric("Вероятность хорошего вина", f"{probability:.1%}")

    if probability >= threshold:
        st.success(f"Включаем в подборку: вероятность {probability:.4f} ≥ порог {threshold:.4f}.")
    else:
        st.info(f"Убираем из подборки: вероятность {probability:.4f} < порог {threshold:.4f}.")


def show_metrics(y_true: pd.Series[int], probabilities: FloatArray, threshold: float) -> None:
    st.subheader("Модельные метрики")

    predicted = (probabilities >= threshold).astype(int)
    precision = metrics.precision_score(y_true, predicted, zero_division=0)
    recall = metrics.recall_score(y_true, predicted, zero_division=0)
    f1 = metrics.f1_score(y_true, predicted, zero_division=0)
    for column, label, value in zip(
        st.columns(3), ["Precision", "Recall", "F1"], [precision, recall, f1]
    ):
        column.metric(label, f"{value:.3f}")

    cm = metrics.confusion_matrix(y_true, predicted, labels=[0, 1])
    st.write(f"В подборке: **{predicted.sum()} из {len(predicted)}** вин при пороге **{threshold:.4f}**.")
    st.write("Матрица ошибок: строки - истинный класс, столбцы - решение модели.")
    st.dataframe(
        pd.DataFrame(cm, index=["Истина: 0", "Истина: 1"], columns=["Прогноз: 0", "Прогноз: 1"])
    )
    if predicted.sum() == 0:
        st.caption("Модель не выбрала ни одного вина. ")

    fpr, tpr, _ = metrics.roc_curve(y_true, probabilities)
    pr_precision, pr_recall, _ = metrics.precision_recall_curve(y_true, probabilities)
    roc_auc = metrics.roc_auc_score(y_true, probabilities)
    average_precision = metrics.average_precision_score(y_true, probabilities)
    tn, fp, fn, tp = cm.ravel()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4, label="Случайное ранжирование")
    axes[0].scatter(fp / (fp + tn), tp / (tp + fn), color="crimson", zorder=3, label=f"Порог {threshold:.4f}")
    axes[0].set(xlabel="FPR", ylabel="TPR (Recall)", title="ROC-кривая")

    axes[1].step(pr_recall, pr_precision, where="post", label=f"PR-AUC (AP) = {average_precision:.3f}")
    axes[1].axhline(
        float(y_true.mean()),
        color="black",
        linestyle="--",
        alpha=0.4,
        label="Доля хороших вин"
    )

    curve_precision = precision if predicted.sum() else 1.0
    axes[1].scatter(recall, curve_precision, color="crimson", zorder=3,
                    label=f"Порог {threshold:.4f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="PR-кривая")

    for ax in axes:
        ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if predicted.sum() == 0:
        st.caption("На PR-кривой показана условная крайняя точка (recall=0, precision=1).")
    st.info(
        "Понижение порога расширяет подборку: хороших вин пропускается меньше, "
        "но в неё могут попасть и посредственные. Повышение порога делает отбор строже, "
        "потенциально уменьшая её размер."
    )


def show_batch(model: Pipeline, test_features: pd.DataFrame, threshold: float) -> None:
    st.subheader("Пакетное предсказание")
    st.write(f"Загрузите CSV с 12 признаками без target. Текущий порог: **{threshold:.4f}**.")

    st.download_button(
        "Скачать пример CSV", test_features.head(5).to_csv(index=False).encode("utf-8"),
        file_name=EXAMPLE_CSV_FILENAME, mime="text/csv", key="example_csv",
    )
    uploaded = st.file_uploader("CSV с признаками вин (разделитель - запятая)", type=["csv"])
    if uploaded is None:
        return
    
    try:
        data = pd.read_csv(uploaded)
        result = predict_batch(model, data, test_features.columns.tolist(), threshold)
    except (ValueError, TypeError, UnicodeError, pd.errors.ParserError) as exc:
        st.error(f"Не удалось обработать CSV. {exc}")
        return
    
    extra = [name for name in data.columns if name not in test_features.columns]
    if extra:
        st.caption("Дополнительные столбцы сохранены, но не используются моделью: " + ", ".join(extra))
    
    st.dataframe(result, hide_index=True)
    st.download_button(
        "Скачать предсказания", result.to_csv(index=False).encode("utf-8-sig"),
        file_name=PREDICTIONS_CSV_FILENAME, mime="text/csv", key="predictions_csv",
    )
    st.caption("probability_good - вероятность класса 1; prediction - 0/1; decision - решение.")


def show_coefficients(model: Pipeline, feature_names: list[str]) -> None:
    st.subheader("Что выучила модель")

    classifier = model.steps[-1][1]
    coefficients = pd.Series(classifier.coef_[0], index=feature_names).sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        [FEATURE_LABELS.get(str(name), str(name)) for name in coefficients.index], coefficients,
        color=["#31855b" if weight >= 0 else "#bb4b5a" for weight in coefficients],
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Коэффициент после StandardScaler", title="Влияние признаков на логарифм шансов класса 1")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    positive = coefficients[coefficients > 0].nlargest(3)
    negative = coefficients[coefficients < 0].nsmallest(3)
    def describe(series: pd.Series[float]) -> str:
        return ", ".join(
            f"{FEATURE_LABELS.get(str(name), str(name))} ({value:+.3f})"
            for name, value in series.items()
        )

    st.write("Сильнее всего за хорошее вино: " + describe(positive) + ".")
    st.write("Сильнее всего против: " + describe(negative) + ".")
    st.caption(
        "При прочих равных положительный коэффициент повышает вероятность класса 1, "
        "отрицательный - понижает. Коэффициент относится к изменению признака на одно "
        "стандартное отклонение обучающей выборки."
    )


def show_sidebar(default_threshold: float) -> float:
    with st.sidebar:
        st.header("Порог отбора")
        threshold = st.slider(
            "Минимальная вероятность хорошего вина", min_value=0.0, max_value=1.0,
            value=default_threshold, step=0.001, format="%.4f", key="threshold",
        )
        st.caption(f"Порог из {CONFIG_FILENAME}: {default_threshold:.6f}. Один порог для всех вкладок.")
        st.caption("Вино попадает в подборку, если вероятность ≥ порог.")
    return threshold


def main() -> None:
    st.set_page_config(page_title="Винная подборка", page_icon="🍷", layout="wide")
    st.title("Помощник сомелье")

    try:
        model = load_model(MODEL_PATH)
        test = load_test_data(TEST_PATH)
        config = load_config(CONFIG_PATH)
        
        feature_names = list(model.feature_names_in_)
        test_features = validate_features(test, feature_names)
        
        y_true = test["target"]
        if set(y_true.unique()) != {0, 1}:
            raise ValueError("test.csv должен содержать оба класса target: 0 и 1.")
        probabilities = test_proba(model, test_features)
    except (OSError, ValueError, KeyError, AttributeError, ImportError) as exc:
        st.error(f"Не удалось загрузить файлы модели: {exc}")
        st.info("Сохраните model.joblib, config.json и test.csv из ноутбука в папку wine_app.")
        st.stop()

    threshold = show_sidebar(config["threshold"])

    single, model_metrics, batch, coefficients = st.tabs(
        ["Оценка одного вина", "Модельные метрики", "Пакетное предсказание", "Что выучила модель"]
    )
    with single:
        show_single_wine(model, test_features, threshold)
    with model_metrics:
        show_metrics(y_true, probabilities, threshold)
    with batch:
        show_batch(model, test_features, threshold)
    with coefficients:
        show_coefficients(model, feature_names)


if __name__ == "__main__":
    main()
