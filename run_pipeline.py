"""
BETALYTIX - Uçtan Uca Pipeline Çalıştırıcı
=============================================
Faz 1 -> Faz 5'i tek komutla çalıştırır ve konsola özet rapor basar.

Kullanım:
    python run_pipeline.py
"""

import pandas as pd

from src.config import RAW_MATCHES_CSV, FEATURE_SET_CSV, CLEAN_MATCHES_CSV
from src.data_loader import load_raw_matches, data_summary, load_to_sqlite
from src.preprocessing import run_pipeline as clean_pipeline
from src.feature_engineering import build_feature_set, get_feature_columns
from src.baseline import chronological_split, train_baseline_models, make_random_forest_pipeline
from src.xgboost_model import make_xgboost_pipeline, select_top_features, check_overfitting
from src.evaluation import evaluate_model, print_comparison_table, analyze_which_class_fails


def main():
    print("=" * 70)
    print("BETALYTIX PIPELINE")
    print("=" * 70)

    # --- Faz 1: Veri yükleme + özet ---
    if not RAW_MATCHES_CSV.exists():
        print("\n[Faz 1] Örnek veri seti bulunamadı, üretiliyor...")
        from src.generate_sample_data import generate
        generate()

    raw = load_raw_matches()
    print(f"\n[Faz 1] Ham veri yüklendi: {len(raw)} maç")
    summary = data_summary(raw)
    print(f"  Takım sayısı: {summary['takim_sayisi']}")
    print(f"  Tarih aralığı: {summary['tarih_araligi']}")
    print(f"  Sonuç dağılımı: {summary['sonuc_dagilimi']}")
    load_to_sqlite(raw)
    print("  SQLite veritabanına yazıldı -> data/processed/betalytix.db")

    # --- Faz 2: Temizleme ---
    clean = clean_pipeline(raw)
    CLEAN_MATCHES_CSV.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_MATCHES_CSV, index=False)
    print(f"\n[Faz 2] Veri temizlendi: {len(clean)} satır kaldı")

    # --- Faz 3: Feature Engineering ---
    features = build_feature_set(clean)
    feat_cols = get_feature_columns(features)
    features.to_csv(FEATURE_SET_CSV, index=False)
    print(f"\n[Faz 3] {len(feat_cols)} feature üretildi, {len(features)} satır kullanılabilir")

    # --- Faz 4: Baseline modeller ---
    split = chronological_split(features, feat_cols)
    baseline_models = train_baseline_models(split)
    results = {}
    for name, model in baseline_models.items():
        results[name] = evaluate_model(model, split.X_test, split.y_test)
    print("\n[Faz 4] Baseline Model Sonuçları:")
    print_comparison_table(results)
    print("  " + analyze_which_class_fails(results["Random Forest"]["confusion_matrix"]))

    # --- Faz 5: XGBoost ---
    rf_for_selection = baseline_models["Random Forest"]
    top_features = select_top_features(rf_for_selection, feat_cols)
    xgb = make_xgboost_pipeline()
    xgb.fit(split.X_train[top_features], split.y_train)
    xgb_result = evaluate_model(xgb, split.X_test[top_features], split.y_test)
    results["XGBoost (Top-20)"] = xgb_result

    print("\n[Faz 5] XGBoost (Top-20 feature) Sonucu:")
    print_comparison_table({"XGBoost (Top-20)": xgb_result})
    overfit = check_overfitting(xgb, split.X_train[top_features], split.y_train,
                                 split.X_test[top_features], split.y_test)
    print(f"  Overfitting kontrolü: {overfit}")

    print("\n" + "=" * 70)
    print("TÜM MODELLERİN KARŞILAŞTIRMASI")
    print("=" * 70)
    print_comparison_table(results)

    print("\n[OK] Pipeline tamamlandı. Demo için: streamlit run app/app.py")


if __name__ == "__main__":
    main()
