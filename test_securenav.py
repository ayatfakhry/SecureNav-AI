"""
SecureNav AI — src package
"""
from src.gnss_simulator      import GNSSSimulator, GNSSEpoch, epochs_to_dataframe
from src.spoofing_simulator  import SpoofingSimulator
from src.jamming_simulator   import JammingSimulator
from src.drift_simulator     import DriftSimulator
from src.feature_extraction  import extract_features, FEATURE_COLUMNS, TARGET_COLUMN, CLASS_NAMES
from src.model_training      import (
    build_random_forest, build_svm, build_mlp,
    prepare_data, cross_validate_model, train_model,
    save_model, load_model, get_feature_importances,
)
from src.anomaly_detection   import (
    AnomalyDetector, PositionDriftAnalyser,
    fit_anomaly_detector_from_df, ensemble_anomaly_score,
)
from src.evaluation          import (
    compute_metrics, compute_roc, summarise_cv,
    compare_models, save_report,
)
from src.alert_system        import (
    AlertGenerator, Alert, process_batch,
    alert_summary, print_alert_summary,
    SEVERITY_INFO, SEVERITY_WARNING, SEVERITY_CRITICAL,
)
from src.visualization       import (
    plot_signal_overview, plot_confusion_matrix,
    plot_feature_importance, plot_roc_curves,
    plot_pca_scatter, plot_position_trajectory,
    plot_model_comparison, plot_anomaly_scores,
    save_dashboard,
)

__all__ = [
    "GNSSSimulator", "GNSSEpoch", "epochs_to_dataframe",
    "SpoofingSimulator", "JammingSimulator", "DriftSimulator",
    "extract_features", "FEATURE_COLUMNS", "TARGET_COLUMN", "CLASS_NAMES",
    "build_random_forest", "build_svm", "build_mlp",
    "prepare_data", "cross_validate_model", "train_model",
    "save_model", "load_model", "get_feature_importances",
    "AnomalyDetector", "PositionDriftAnalyser",
    "fit_anomaly_detector_from_df", "ensemble_anomaly_score",
    "compute_metrics", "compute_roc", "summarise_cv",
    "compare_models", "save_report",
    "AlertGenerator", "Alert", "process_batch",
    "alert_summary", "print_alert_summary",
    "SEVERITY_INFO", "SEVERITY_WARNING", "SEVERITY_CRITICAL",
    "plot_signal_overview", "plot_confusion_matrix",
    "plot_feature_importance", "plot_roc_curves",
    "plot_pca_scatter", "plot_position_trajectory",
    "plot_model_comparison", "plot_anomaly_scores",
    "save_dashboard",
]
