# app.py
# Streamlit-приложение: XGBoost (+ One-Hot «Район») с пошаговым выводом

import streamlit as st
import warnings, io, json
from pathlib import Path
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
plt.rcParams["figure.dpi"] = 110
st.set_page_config(page_title="Эко-здоровье (XGB-SHAP)", layout="wide")

# ───────────────────────── UI ──────────────────────────────────────────────
st.title("Экологические факторы → Заболеваемость   (XGBoost + SHAP)")

upl = st.file_uploader("Загрузите сводную таблицу (.xlsx или .csv)", type=("xlsx", "csv"))
if not upl:
    st.info("⬆️  Сначала загрузите файл")
    st.stop()

# чтение
if upl.name.lower().endswith(".csv"):
    df = pd.read_csv(upl)
else:
    df = pd.read_excel(upl, engine="openpyxl")

# выбор колонок
year_col   = st.selectbox("Колонка года", [c for c in df.columns if df[c].dtype != "object"])
region_col = st.selectbox("Колонка района", df.select_dtypes("object").columns)
target_col = st.selectbox("Целевая переменная", [c for c in df.columns if c not in [year_col, region_col]])

# гиперпараметры
min_nonmiss = st.slider("Мин. доля заполненности признака", 0.1, 0.9, 0.3, 0.05)
test_years  = st.slider("Сколько последних лет в тесте", 1, 3, 1)
max_depth   = st.slider("max_depth (XGB)", 3, 10, 6)
n_estim     = st.slider("n_estimators (XGB)", 300, 2000, 1200, 100)

start = st.button("Запустить анализ")
if not start:
    st.stop()

# ───────────────────── Подготовка и обучение (кешируем) ─────────────────────
@st.cache_resource(show_spinner=False)
def train_model(df, year_col, region_col, target_col,
                min_nonmiss, test_years, max_depth, n_estim):

    eco_cols = [c for c in df.columns if c.startswith(("Air_", "Water_", "Soil_"))]
    for c in eco_cols:
        lag = f"{c}_lag1"
        if lag not in df.columns:
            df[lag] = df.groupby(region_col)[c].shift(1)

    num_cols  = eco_cols + [f"{c}_lag1" for c in eco_cols]
    good_cols = [c for c in num_cols if df[c].notna().mean() >= min_nonmiss]

    train_mask = df[year_col] < df[year_col].max() - test_years
    X_train = df.loc[train_mask, good_cols + [region_col]]
    y_train = df.loc[train_mask, target_col]
    X_test  = df.loc[~train_mask, good_cols + [region_col]]
    y_test  = df.loc[~train_mask, target_col]

    pipe = Pipeline([
        ("prep", ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"),
              [X_train.columns.get_loc(region_col)])],
            remainder="passthrough", sparse_threshold=0.3)),
        ("xgb", XGBRegressor(
            n_estimators=n_estim, max_depth=max_depth,
            learning_rate=0.045, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0,
            random_state=42, n_jobs=-1, missing=np.nan))
    ]).fit(X_train, y_train)

    return dict(df=df, good_cols=good_cols,
                pipe=pipe, X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test)

with st.spinner("⏳ Обучаем XGBoost…"):
    data = train_model(df.copy(), year_col, region_col, target_col,
                       min_nonmiss, test_years, max_depth, n_estim)
st.success("✅ Модель обучена")

pipe     = data["pipe"]
X_train  = data["X_train"]; y_train = data["y_train"]
X_test   = data["X_test"];  y_test  = data["y_test"]
good_cols= data["good_cols"]

# ───────────────────── Метрики (показываем сразу) ───────────────────────────
y_pred = pipe.predict(X_test)
R2  = r2_score(y_test, y_pred)
RMSE = mean_squared_error(y_test, y_pred, squared=False)
MAE  = mean_absolute_error(y_test, y_pred)

tscv = TimeSeriesSplit(n_splits=5)
cv_r2 = cross_val_score(pipe, X_train, y_train, cv=tscv, scoring="r2")
st.markdown(f"### 🎯 Hold-out R² = **{R2:.3f}** &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"CV R² = **{cv_r2.mean():.3f} ± {cv_r2.std():.3f}**")
st.caption(f"RMSE = {RMSE:,.0f}   MAE = {MAE:,.0f}")

# ───────────────────── Feature gain barplot ─────────────────────────────────
feat_names  = pipe.named_steps["prep"].get_feature_names_out()
gain_scores = pipe.named_steps["xgb"].get_booster().get_score(importance_type="gain")
gain_ser    = pd.Series(gain_scores).reindex(feat_names, fill_value=0).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(6,8))
sns.barplot(y=gain_ser.head(20).index, x=gain_ser.head(20).values,
            ax=ax, palette="viridis")
ax.set_title("XGB gain importance"); ax.set_xlabel("gain"); ax.set_ylabel("")
st.pyplot(fig)

# ───────────────────── SHAP (прогресс-бар) ──────────────────────────────────
with st.spinner("Вычисляем SHAP (подвыборка 400 строк)…"):
    X_train_enc = pipe.named_steps["prep"].transform(X_train)
    sub_idx = np.random.choice(X_train_enc.shape[0],
                               size=min(400, X_train_enc.shape[0]),
                               replace=False)
    explainer   = shap.TreeExplainer(pipe.named_steps["xgb"])
    shap_values = explainer.shap_values(X_train_enc[sub_idx])

shap_abs = np.abs(shap_values).mean(axis=0)
shap_df  = pd.DataFrame({"feature": feat_names, "shap": shap_abs})\
             .sort_values("shap", ascending=False)

# SHAP-bar
fig, ax = plt.subplots(figsize=(6,8))
sns.barplot(y=shap_df.head(20)["feature"], x=shap_df.head(20)["shap"],
            ax=ax, palette="magma")
ax.set_title("Global SHAP importance")
st.pyplot(fig)

# SHAP-beeswarm
st.subheader("SHAP beeswarm")
shap.summary_plot(shap_values,
                  pd.DataFrame(X_train_enc[sub_idx], columns=feat_names),
                  show=False, max_display=25, plot_size=(8,6))
st.pyplot(bbox_inches="tight", clear_figure=True)

# SHAP dependence (топ-3)
st.subheader("SHAP dependence (топ-3)")
for feat in shap_df.head(3)["feature"]:
    shap.dependence_plot(feat, shap_values,
                         pd.DataFrame(X_train_enc[sub_idx], columns=feat_names),
                         interaction_index=None, show=False, alpha=0.4)
    st.pyplot(bbox_inches="tight", clear_figure=True)

# ───────────────────── Partial dependence (числовые) ────────────────────────
num_feats = [f for f in shap_df["feature"] if f.startswith("remainder__")]
top_raw   = [f.replace("remainder__", "") for f in num_feats[:3]]
st.subheader("Partial dependence (топ-3 числовых)")
for raw in top_raw:
    fig, ax = plt.subplots(figsize=(5,3))
    PartialDependenceDisplay.from_estimator(pipe, X_train, [raw],
                                            ax=ax, grid_resolution=50)
    st.pyplot(fig)

# ───────────────────── Scatter и residuals ───────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    fig, ax = plt.subplots(figsize=(4,4))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--k")
    ax.set_xlabel("Факт"); ax.set_ylabel("Прогноз"); ax.set_title("Test: прогноз vs факт")
    st.pyplot(fig)
with c2:
    fig, ax = plt.subplots(figsize=(4,3))
    sns.histplot(y_test - y_pred, bins=30, kde=True, ax=ax)
    ax.set_title("Гистограмма остатков"); ax.set_xlabel("Residual")
    st.pyplot(fig)

# ───────────────────── Тренды по регионам ───────────────────────────────────
st.subheader("Годовые тренды (топ-6 регионов)")
top_regions = df.groupby(region_col)[target_col].mean()\
                 .sort_values(ascending=False).head(6).index
for reg in top_regions:
    sub = df[df[region_col]==reg].sort_values(year_col)
    sub_enc = pipe.named_steps["prep"].transform(sub[good_cols + [region_col]])
    sub["pred"] = pipe.named_steps["xgb"].predict(sub_enc)
    fig, ax = plt.subplots(figsize=(6,2.5))
    ax.plot(sub[year_col], sub[target_col], "-o", label="Факт")
    ax.plot(sub[year_col], sub["pred"], "-s", label="Прогноз")
    ax.set_title(reg); ax.set_xlabel("Год"); ax.set_ylabel("")
    ax.legend(); st.pyplot(fig)

# ───────────────────── Таблица факторов + download ──────────────────────────
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
st.dataframe(stats_df)

csv = stats_df.to_csv(index=False).encode()
st.download_button("⬇️ Скачать таблицу факторов (CSV)",
                   csv, "predictor_stats.csv", "text/csv")

