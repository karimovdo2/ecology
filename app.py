# app.py
# Streamlit‑приложение: XGBoost + SHAP, пошаговый вывод
# ------------------------------------------------------
import streamlit as st, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.inspection import PartialDependenceDisplay
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import shap, os

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
st.set_page_config(page_title="Eco‑Health XGB", layout="wide")

# ───────────────────────── UI ────────────────────────────────
st.title("Экология → заболеваемость :  XGBoost + SHAP")

upl = st.file_uploader("Загрузите .xlsx или .csv", type=("xlsx", "csv"))
if not upl:
    st.stop()

df = pd.read_csv(upl) if upl.name.lower().endswith(".csv") else pd.read_excel(upl, engine="openpyxl")
year_col = st.selectbox("Колонка года", sorted([c for c in df.columns if df[c].dtype != "object"]))
region_col = st.selectbox("Колонка района", sorted(df.select_dtypes("object").columns))
target_col = st.selectbox("Целевая переменная", sorted([c for c in df.columns if c not in [year_col, region_col]]))

# Параметры для фильтрации
min_nonmiss = st.slider("Мин. заполненность признака", 0.0, 1.0, 0.3, 0.05)
min_row_fill = st.slider("Мин. заполненность строки (для каждого района)", 0.0, 1.0, 0.3, 0.01)
test_years = st.slider("Лет в тесте", 1, 3, 1)
max_depth = st.slider("max_depth", 2, 10, 3)
n_estim = st.slider("n_estimators", 100, 300, 100, 50)

if not st.button("🚀 Запустить анализ"):
    st.stop()

# ───────────────────── Подготовка и кэш ──────────────────────
@st.cache_resource(show_spinner=False)
def train_cache(df, year_col, region_col, target_col,
                min_nonmiss, min_row_fill, test_years, max_depth, n_estim):
    # фильтрация по заполненности строк (минимум 30%)
    df = df[df.notna().mean(axis=1) >= min_row_fill]
    
    eco_cols = [c for c in df.columns if c.startswith(("Air_", "Water_", "Soil_"))]
    for c in eco_cols:
        lag = f"{c}_lag1"
        if lag not in df.columns:
            df[lag] = df.groupby(region_col)[c].shift(1)

    num_cols  = eco_cols + [f"{c}_lag1" for c in eco_cols]
    good_cols = [c for c in num_cols if df[c].notna().mean() >= min_nonmiss]

    # вывод статистики по районам
    num_districts = len(df[region_col].unique())
    # Показать количество строк данных
    num_rows = len(df)
    st.write(f"Количество строк данных: {num_rows}")

    # Показать количество выбранных признаков и районов
    st.write(f"Выбрано {len(good_cols)} признаков и {len(df[region_col].unique())} районов.")


    train_mask = df[year_col] < df[year_col].max() - test_years
    X_train = df.loc[train_mask, good_cols + [region_col]]
    y_train = df.loc[train_mask, target_col]
    X_test  = df.loc[~train_mask, good_cols + [region_col]]
    y_test  = df.loc[~train_mask, target_col]

    pipe = Pipeline([  # создаём пайплайн с XGBoost
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

    return df, good_cols, pipe, X_train, y_train, X_test, y_test

with st.spinner("⏳ Обучаем XGBoost…"):
    df, good_cols, pipe, X_tr, y_tr, X_te, y_te = train_cache(
        df.copy(), year_col, region_col, target_col,
        min_nonmiss, min_row_fill, test_years, max_depth, n_estim)
st.success("✅ Модель обучена")

# ───────────────────── Метрики ───────────────────────────────

y_pred = pipe.predict(X_te)

# Вычисление R², RMSE и MAE
R2 = r2_score(y_te, y_pred)
st.markdown(f"### 🎯 Hold‑out R² **{R2:.3f}**")
RMSE = np.sqrt(mean_squared_error(y_te, y_pred))
MAE = mean_absolute_error(y_te, y_pred)

# Вывод метрик сразу после их расчета

st.caption(f"RMSE **{RMSE:,.0f}**   |   MAE **{MAE:,.0f}**")



# ───────────────────── SHAP ──────────────────────────────────
with st.spinner("SHAP (подвыборка 400 строк)…"):
    # Трансформируем данные с помощью pipeline
    X_tr_enc = pipe.named_steps["prep"].transform(X_tr)

    # Выбираем первые 400 строк, чтобы избежать проблем с несоответствием данных
    sub = np.arange(min(400, X_tr_enc.shape[0]))

    # Проверяем, что размерность данных после трансформации совпадает с количеством признаков
    if X_tr_enc.shape[1] != len(feat_names):
        st.error(f"Количество признаков в данных после трансформации не совпадает с количеством признаков в модели: {X_tr_enc.shape[1]} != {len(feat_names)}")
        st.stop()  # Останавливаем выполнение, если данные не совпадают
    
    # Инициализируем SHAP Explainer
    explainer = shap.TreeExplainer(pipe.named_steps["xgb"])
    
    # Вычисляем SHAP значения
    shap_values = explainer.shap_values(X_tr_enc[sub])

# Получаем абсолютные значения SHAP для оценки важности признаков
shap_abs = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({"feature": feat_names, "shap": shap_abs}).sort_values("shap", ascending=False)

# Bar plot для SHAP важности
fig, ax = plt.subplots(figsize=(6, 8))
sns.barplot(y=shap_df.head(20)["feature"], x=shap_df.head(20)["shap"], palette="magma", ax=ax)
ax.set_title("Global SHAP importance")
ax.set_xlabel("|SHAP|")
ax.set_ylabel("")
st.pyplot(fig)

# SHAP beeswarm plot
st.subheader("SHAP beeswarm")
shap.summary_plot(shap_values, pd.DataFrame(X_tr_enc[sub], columns=feat_names),
                  show=False, max_display=25, plot_size=(8, 6))
st.pyplot(plt.gcf())  # вывод текущего рисунка
plt.clf()

# SHAP dependence plots для топ-3 признаков
st.subheader("SHAP dependence (топ‑3)")
for feat in shap_df.head(3)["feature"]:
    shap.dependence_plot(feat, shap_values,
                         pd.DataFrame(X_tr_enc[sub], columns=feat_names),
                         show=False, interaction_index=None, alpha=0.4)
    st.pyplot(plt.gcf())
    plt.clf()






# ───────────────────── Scatter & residuals ───────────────────
c1, c2 = st.columns(2)
with c1:
    fig, ax = plt.subplots(figsize=(4,4))
    sns.scatterplot(x=y_te, y=y_pred, ax=ax)
    ax.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], "--k")
    ax.set_xlabel("Факт"); ax.set_ylabel("Прогноз"); ax.set_title("Test: прогноз vs факт")
    st.pyplot(fig)
with c2:
    fig, ax = plt.subplots(figsize=(4,3))
    sns.histplot(y_te - y_pred, bins=30, kde=True, ax=ax)
    ax.set_title("Гистограмма остатков"); ax.set_xlabel("Residual")
    st.pyplot(fig)

# ───────────────────── Тренды по регионам ────────────────────
st.subheader("Годовые тренды (топ‑6 регионов)")
top_regions = df.groupby(region_col)[target_col].mean().sort_values(ascending=False).head(6).index
for reg in top_regions:
    sub = df[df[region_col]==reg].sort_values(year_col)
    sub_enc = pipe.named_steps["prep"].transform(sub[good_cols + [region_col]])
    sub["pred"] = pipe.named_steps["xgb"].predict(sub_enc)
    fig, ax = plt.subplots(figsize=(5,2.5))
    ax.plot(sub[year_col], sub[target_col], "-o", label="Факт")
    ax.plot(sub[year_col], sub["pred"], "-s", label="Прогноз")
    ax.set_title(reg); ax.set_xlabel("Год"); ax.set_ylabel("")
    ax.legend(); st.pyplot(fig)

# ───────────────────── Таблица факторов ──────────────────────
stats_rows = []
for feat in shap_df.head(25)["feature"]:
    raw = feat.replace("remainder__", "").replace("cat__", "")
    pearson = df[[raw, target_col]].corr().iloc[0,1] if raw in df else np.nan
    idx = list(feat_names).index(feat)
    stats_rows.append(dict(
        feature=feat,
        pearson_corr=pearson,
        shap_mean=float(shap_values[:, idx].mean()),
        shap_sd=float(shap_values[:, idx].std()),
        nonmiss_ratio=float(df[raw].notna().mean()) if raw in df else np.nan
    ))
stats_df = pd.DataFrame(stats_rows)
st.dataframe(stats_df)

csv = stats_df.to_csv(index=False).encode()
st.download_button("⬇️ Скачать CSV", csv, "predictor_stats.csv", "text/csv")
