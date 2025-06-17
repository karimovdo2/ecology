# app.py
# Streamlit‑приложение: экологические факторы → заболеваемость
# Запуск:  streamlit run app.py

import streamlit as st
from pathlib import Path
import io, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.inspection import PartialDependenceDisplay
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor
import shap

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

st.set_page_config(page_title="Эко‑здоровье: XGBoost‑анализ", layout="wide")

# ───────────────── UI ────────────────────────────────────────────────────────
st.title("Анализ влияния экологических факторов на заболеваемость")

uploaded = st.file_uploader("Загрузите объединённую таблицу (.xlsx или .csv)", type=["xlsx", "csv"])

if uploaded:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    # базовые колонки
    year_col   = st.selectbox("Колонка года", [c for c in df.columns if df[c].dtype!=object])
    region_col = st.selectbox("Колонка района", df.select_dtypes("object").columns)
    target_col = st.selectbox("Целевая переменная (заболеваемость)",
                              [c for c in df.columns if c not in [year_col, region_col]])

    # параметры
    min_nonmiss = st.slider("Минимальная доля заполненных значений для признака",
                            0.1, 0.9, 0.3, 0.05)
    test_years  = st.slider("Сколько последних лет оставить в тесте", 1, 3, 1)
    max_depth   = st.slider("max_depth (XGB)", 3, 10, 6, 1)
    n_estim     = st.slider("n_estimators (XGB)", 300, 2000, 1200, 100)

    if st.button("Запустить анализ"):
        with st.spinner("Обработка данных и обучение модели…"):
            # определяем экосписок
            eco_cols = [c for c in df.columns if c.startswith(("Air_", "Water_", "Soil_"))]

            # добавляем лаг‑1, если нет
            for c in eco_cols:
                lag = f"{c}_lag1"
                if lag not in df.columns:
                    df[lag] = df.groupby(region_col)[c].shift(1)

            num_cols  = eco_cols + [f"{c}_lag1" for c in eco_cols]
            good_cols = [c for c in num_cols if df[c].notna().mean() >= min_nonmiss]

            # train/test
            train_mask = df[year_col] < df[year_col].max() - test_years
            X_train = df.loc[train_mask, good_cols + [region_col]]
            y_train = df.loc[train_mask, target_col]
            X_test  = df.loc[~train_mask, good_cols + [region_col]]
            y_test  = df.loc[~train_mask, target_col]

            # пайплайн
            cat_idx = [X_train.columns.get_loc(region_col)]
            preproc = ColumnTransformer(
                [("cat", OneHotEncoder(handle_unknown="ignore"), cat_idx)],
                remainder="passthrough",
                sparse_threshold=0.3
            )
            xgb = XGBRegressor(
                n_estimators=n_estim, max_depth=max_depth,
                learning_rate=0.045, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=1.0,
                random_state=42, missing=np.nan, n_jobs=-1
            )
            pipe = Pipeline([("prep", preproc), ("xgb", xgb)]).fit(X_train, y_train)

            # метрики
            y_pred = pipe.predict(X_test)
            R2  = r2_score(y_test, y_pred)
            RMSE = mean_squared_error(y_test, y_pred, squared=False)
            MAE  = mean_absolute_error(y_test, y_pred)

            tscv = TimeSeriesSplit(n_splits=5)
            cv_r2 = cross_val_score(pipe, X_train, y_train, cv=tscv, scoring="r2")
            st.success(f"**Hold‑out R² = {R2:.3f}  |  CV R² = {cv_r2.mean():.3f} ± {cv_r2.std():.3f}**")
            st.caption(f"RMSE = {RMSE:,.1f}   |   MAE = {MAE:,.1f}")

            # имена признаков
            feat_names = pipe.named_steps["prep"].get_feature_names_out()
            X_train_enc = pipe.named_steps["prep"].transform(X_train)

            # gain‑импортанс
            gain = pipe.named_steps["xgb"].get_booster().get_score(importance_type="gain")
            gain_s = pd.Series(gain).reindex(feat_names, fill_value=0).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(6,8))
            sns.barplot(y=gain_s.head(20).index, x=gain_s.head(20).values, ax=ax, palette="viridis")
            ax.set_title("XGB gain importance"); ax.set_xlabel("gain"); ax.set_ylabel("")
            st.pyplot(fig)

            # SHAP
            explainer = shap.TreeExplainer(pipe.named_steps["xgb"])
            shap_values = explainer.shap_values(X_train_enc)
            shap_abs = np.abs(shap_values).mean(axis=0)
            shap_df  = pd.DataFrame({"feature": feat_names, "shap": shap_abs})\
                        .sort_values("shap", ascending=False)
            # SHAP bar
            fig, ax = plt.subplots(figsize=(6,8))
            sns.barplot(y=shap_df.head(20)["feature"], x=shap_df.head(20)["shap"], ax=ax, palette="magma")
            ax.set_title("Global SHAP importance"); ax.set_xlabel("mean |SHAP|"); ax.set_ylabel("")
            st.pyplot(fig)

            # SHAP beeswarm
            st.subheader("SHAP beeswarm")
            shap_fig = shap.summary_plot(shap_values,
                                         pd.DataFrame(X_train_enc, columns=feat_names),
                                         show=False, max_display=25, plot_size=(8,6))
            st.pyplot(bbox_inches="tight", clear_figure=True)

            # SHAP dependence for top‑3
            st.subheader("SHAP dependence (топ‑3)")
            for feat in shap_df.head(3)["feature"]:
                shap.dependence_plot(
                    feat, shap_values, pd.DataFrame(X_train_enc, columns=feat_names),
                    interaction_index=None, show=False, alpha=0.4
                )
                st.pyplot(bbox_inches="tight", clear_figure=True)

            # Partial dependence (числовые)
            num_feats = [f for f in shap_df["feature"] if f.startswith("remainder__")]
            top_raw   = [f.replace("remainder__", "") for f in num_feats[:3]]
            st.subheader("Partial dependence (топ‑3 числовых)")
            for raw in top_raw:
                fig, ax = plt.subplots(figsize=(5,3))
                PartialDependenceDisplay.from_estimator(
                    pipe, X_train, [raw], ax=ax, grid_resolution=50
                )
                st.pyplot(fig)

            # scatter прогноз‑факт
            fig, ax = plt.subplots(figsize=(4,4))
            sns.scatterplot(x=y_test, y=y_pred, ax=ax)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--k")
            ax.set_xlabel("Факт"); ax.set_ylabel("Прогноз"); ax.set_title("Test: прогноз vs факт")
            st.pyplot(fig)

            # residuals
            fig, ax = plt.subplots(figsize=(5,3))
            sns.histplot(y_test - y_pred, bins=30, kde=True, ax=ax)
            ax.set_title("Гистограмма остатков"); ax.set_xlabel("Residual")
            st.pyplot(fig)

            # годовые тренды
            st.subheader("Годовые тренды (топ‑6 регионов)")
            top_regions = df.groupby(region_col)[target_col].mean()\
                             .sort_values(ascending=False).head(6).index
            for reg in top_regions:
                sub = df[df[region_col]==reg].sort_values(year_col)
                sub_enc = pipe.named_steps["prep"].transform(sub[good_cols + [region_col]])
                sub["pred"] = pipe.named_steps["xgb"].predict(sub_enc)
                fig, ax = plt.subplots(figsize=(5,2.5))
                ax.plot(sub[year_col], sub[target_col], "-o", label="Факт")
                ax.plot(sub[year_col], sub["pred"], "-s", label="Прогноз")
                ax.set_title(reg); ax.set_xlabel("Год"); ax.set_ylabel("")
                ax.legend(); st.pyplot(fig)

            # таблица факторов
            stats_rows = []
            for feat in shap_df.head(25)["feature"]:
                raw_name = feat.replace("remainder__", "").replace("cat__", "")
                pearson = df[[raw_name, target_col]].corr().iloc[0,1] if raw_name in df else np.nan
                idx = list(feat_names).index(feat)
                stats_rows.append(dict(
                    feature=feat,
                    pearson_corr=pearson,
                    shap_mean=float(shap_values[:, idx].mean()),
                    shap_sd=float(shap_values[:, idx].std()),
                    nonmiss_ratio=float(df[raw_name].notna().mean()) if raw_name in df else np.nan
                ))
            stats_df = pd.DataFrame(stats_rows)
            st.dataframe(stats_df.head(25))

            # download
            csv = stats_df.to_csv(index=False).encode()
            st.download_button("Скачать таблицу факторов (CSV)", csv, "predictor_stats.csv", "text/csv")
else:
    st.info("⬆️  Загрузите файл, чтобы начать.")
