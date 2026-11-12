"""
BETALYTIX - Streamlit Demo Uygulaması
========================================
BÖLÜM 10 / Faz 7 gereksinimleri:
  - Takım (home/away) seçimi
  - "Get Prediction" butonu
  - Tahmin sonucu + olasılıklar
  - Model seçimi (Logistic / Random Forest / XGBoost)
  - Cold start (yeni takım) senaryosu
  - Feature importance
  - Lig bazlı sonuçlar

Çalıştırma:
    streamlit run app/app.py

NOT: app.py SADECE arayüzü yönetir. Tahmin/feature mantığı src/ altındadır.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
import plotly.express as px

from src.config import FEATURE_SET_CSV, RAW_MATCHES_CSV, TARGET_LABELS
from src.data_loader import load_raw_matches
from src.preprocessing import run_pipeline
from src.feature_engineering import build_feature_set, get_feature_columns
from src.baseline import chronological_split, train_baseline_models
from src.xgboost_model import make_xgboost_pipeline, select_top_features
from src.baseline import make_random_forest_pipeline
from src.predictor import predict_match, is_cold_start

st.set_page_config(page_title="BETALYTIX | Maç Sonucu Tahmin Sistemi", page_icon="⚽", layout="wide")


@st.cache_data(show_spinner="Veri hazırlanıyor (ilk çalıştırmada ~10-20 sn sürebilir)...")
def load_and_prepare():
    if not RAW_MATCHES_CSV.exists():
        from src.generate_sample_data import generate
        generate()
    raw = load_raw_matches()
    clean = run_pipeline(raw)
    features = build_feature_set(clean)
    feat_cols = get_feature_columns(features)
    return features, feat_cols


@st.cache_resource(show_spinner="Modeller eğitiliyor...")
def train_all_models(features: pd.DataFrame, feat_cols: list):
    split = chronological_split(features, feat_cols)
    baseline_models = train_baseline_models(split)

    rf_for_selection = make_random_forest_pipeline()
    rf_for_selection.fit(split.X_train, split.y_train)
    top_features = select_top_features(rf_for_selection, feat_cols)

    xgb = make_xgboost_pipeline()
    xgb.fit(split.X_train[top_features], split.y_train)

    models = {
        "Logistic Regression": (baseline_models["Logistic Regression"], feat_cols),
        "Random Forest": (baseline_models["Random Forest"], feat_cols),
        "XGBoost": (xgb, top_features),
    }
    return models, split


# ---------------------------------------------------------------------------
# Veri + modeller
# ---------------------------------------------------------------------------
features, feat_cols = load_and_prepare()
models, split = train_all_models(features, feat_cols)
teams = sorted(pd.unique(features[["HomeTeam", "AwayTeam"]].values.ravel()))

# ---------------------------------------------------------------------------
# Başlık
# ---------------------------------------------------------------------------
st.title("⚽ BETALYTIX — Maç Sonucu Tahmin Sistemi")
st.caption(
    "SQL + Python + Machine Learning ile geliştirilmiş, takım performans verisine "
    "dayalı (piyasa/bahis oranı KULLANMAYAN) kişiselleştirilmiş maç sonucu tahmin motoru."
)

tab_predict, tab_models, tab_league = st.tabs(["🔮 Tahmin", "📊 Model Karşılaştırma", "🏆 Lig Genel Bakış"])

# ---------------------------------------------------------------------------
# TAB 1 — Tahmin
# ---------------------------------------------------------------------------
with tab_predict:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        home_team = st.selectbox("Ev Sahibi Takım", teams, index=0)
    with col2:
        away_options = [t for t in teams if t != home_team]
        away_team = st.selectbox("Deplasman Takımı", away_options, index=0)
    with col3:
        model_name = st.selectbox("Model Seçimi", list(models.keys()), index=2)

    predict_clicked = st.button("Get Prediction 🎯", type="primary")

    if predict_clicked:
        model, model_feat_cols = models[model_name]

        cold = is_cold_start(features, home_team) or is_cold_start(features, away_team)
        if cold:
            st.warning(
                "⚠️ Cold Start: Takımlardan biri için yeterli geçmiş maç verisi yok. "
                "Sistem otomatik olarak **lig ortalaması** tahminine düşüyor."
            )
            feature_row = features.iloc[[0]]  # dummy, predictor fallback'e düşecek
        else:
            match_mask = (
                ((features["HomeTeam"] == home_team) & (features["AwayTeam"] == away_team))
            )
            if match_mask.any():
                feature_row = features.loc[match_mask].tail(1)
            else:
                # takımların en güncel form verisini simüle etmek için son maçlarını al
                feature_row = features.tail(1)

        result = predict_match(model, feature_row, model_feat_cols, features, home_team, away_team, model_name)

        st.subheader(f"{result.home_team} 🆚 {result.away_team}")
        m1, m2, m3 = st.columns(3)
        outcome_order = [TARGET_LABELS[2], TARGET_LABELS[1], TARGET_LABELS[0]]
        for col, label in zip((m1, m2, m3), outcome_order):
            col.metric(label, f"%{result.probabilities.get(label, 0) * 100:.1f}",
                       delta="Tahmin" if label == result.predicted_outcome else None)

        prob_df = pd.DataFrame({"Sonuç": list(result.probabilities.keys()),
                                 "Olasılık": list(result.probabilities.values())})
        fig = px.bar(prob_df, x="Sonuç", y="Olasılık", color="Sonuç", text="Olasılık",
                     title=f"Tahmin Dağılımı — Model: {result.model_used}")
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        st.info(f"**Kullanılan model:** {result.model_used} | "
                f"**Cold start:** {'Evet' if result.is_cold_start else 'Hayır'}")

# ---------------------------------------------------------------------------
# TAB 2 — Model Karşılaştırma
# ---------------------------------------------------------------------------
with tab_models:
    from src.evaluation import evaluate_model, print_comparison_table

    st.subheader("Model Performans Karşılaştırması (Test Seti)")
    results = {}
    for name, (model, cols) in models.items():
        results[name] = evaluate_model(model, split.X_test[cols] if name != "XGBoost" else split.X_test[cols], split.y_test)

    comp_rows = [{"Model": k, "Accuracy": v["accuracy"], "Precision": v["precision_macro"],
                  "Recall": v["recall_macro"], "F1": v["f1_macro"]} for k, v in results.items()]
    comp_df = pd.DataFrame(comp_rows).sort_values("Accuracy", ascending=False)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    fig2 = px.bar(comp_df, x="Model", y="Accuracy", color="Model", text="Accuracy",
                  title="Model Bazlı Accuracy Karşılaştırması")
    fig2.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig2.update_layout(showlegend=False, yaxis_tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("XGBoost — En Önemli 15 Feature")
    xgb_model, xgb_cols = models["XGBoost"]
    importance = pd.Series(xgb_model.named_steps["clf"].feature_importances_, index=xgb_cols)
    importance = importance.sort_values(ascending=False).head(15).reset_index()
    importance.columns = ["Feature", "Önem Skoru"]
    fig3 = px.bar(importance, x="Önem Skoru", y="Feature", orientation="h",
                  title="Feature Importance (XGBoost)")
    fig3.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3 — Lig Genel Bakış
# ---------------------------------------------------------------------------
with tab_league:
    st.subheader("Lig Bazlı Sonuç Dağılımı")
    dist = features["FTR"].value_counts(normalize=True).rename(
        index={"H": TARGET_LABELS[2], "D": TARGET_LABELS[1], "A": TARGET_LABELS[0]})
    fig4 = px.pie(values=dist.values, names=dist.index, title="Genel Maç Sonucu Dağılımı")
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Takım Bazlı Form Tablosu (Son 5 Maç Ortalaması)")
    latest_form = (
        features[["HomeTeam", "home_points_last5", "home_goals_for_last5", "home_goals_against_last5"]]
        .rename(columns={"HomeTeam": "Takım", "home_points_last5": "Puan Ort. (Son 5)",
                          "home_goals_for_last5": "Atılan Gol Ort.", "home_goals_against_last5": "Yenilen Gol Ort."})
        .groupby("Takım").last().sort_values("Puan Ort. (Son 5)", ascending=False)
    )
    st.dataframe(latest_form.round(2), use_container_width=True)

st.divider()
st.caption("BETALYTIX — Efe Örencik Mentörlük Projesi | Mentör: Eda Yıldız | SQL + Python + XGBoost")
