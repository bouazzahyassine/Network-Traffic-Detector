
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import csv
import io

from pathlib import Path

# =====================================================
# 1. CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Network Traffic Anomaly Detection",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models/random_forest_final_pipeline.pkl"
FIGURES_DIR = BASE_DIR / "results/figures"

# =====================================================
# 2. CUSTOM STYLE
# =====================================================

st.markdown("""
<style>
.main {
    background-color: #f5f8fc;
}
.block-container {
    padding-top: 2rem;
}
[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}
h1, h2, h3 {
    color: #12345a;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 3. MODEL LOADING
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("🛡️ Network Traffic Anomaly Detection")

st.markdown(
    "### Intelligent Network Intrusion Detection System"
)

st.caption(
    "Machine Learning | Random Forest | CIC-IDS2017"
)

if not MODEL_PATH.exists():
    st.error("Final model not found.")
    st.stop()

try:
    model = load_model()

    if not hasattr(model, "feature_names_in_"):
        st.error("The model does not contain feature names.")
        st.stop()

    expected_features = list(model.feature_names_in_)
    classes = list(model.classes_)

    if len(expected_features) != 78:
        st.error(
            f"Unexpected model configuration: "
            f"{len(expected_features)} features instead of 78."
        )
        st.stop()

except Exception as error:
    st.error(f"Model loading error: {error}")
    st.stop()

# =====================================================
# 4. SIDEBAR
# =====================================================

st.sidebar.title("🛡️ Navigation")

page = st.sidebar.radio(
    "Select a section",
    [
        "Dashboard",
        "Traffic Analysis",
        "Model Evaluation",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.write("**Model:** Random Forest")
st.sidebar.write("**Dataset:** CIC-IDS2017")
st.sidebar.write(f"**Features:** {len(expected_features)}")
st.sidebar.write(f"**Classes:** {len(classes)}")

# =====================================================
# 5. DASHBOARD
# =====================================================

if page == "Dashboard":

    st.header("Project Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", "99.9989%")
    c2.metric("Macro Precision", "99.9996%")
    c3.metric("Macro Recall", "99.9717%")
    c4.metric("Macro F1-score", "99.9857%")

    st.info(
        "These metrics come from the held-out test "
        "evaluation containing 89,182 observations."
    )

    st.subheader("Classification Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("BENIGN", "Normal traffic")
    c2.metric("FTP-Patator", "Attack traffic")
    c3.metric("SSH-Patator", "Attack traffic")

    st.subheader("Scientific Visualizations")

    figures = [
        ("01_class_distribution.png", "Class Distribution"),
        ("02_confusion_matrix.png", "Confusion Matrix"),
        ("03_class_performance.png", "Performance by Class"),
        ("04_global_metrics.png", "Global Evaluation Metrics"),
        (
            "05_destination_port_comparison.png",
            "Destination Port Comparison"
        )
    ]

    for filename, caption in figures:

        image_path = FIGURES_DIR / filename

        if image_path.exists():

            st.markdown(f"**{caption}**")

            st.image(
                str(image_path),
                use_container_width=True
            )

        else:
            st.warning(f"Figure not found: {filename}")

# =====================================================
# 6. TRAFFIC ANALYSIS
# =====================================================

elif page == "Traffic Analysis":

    st.header("Network Traffic Analysis")

    st.write(
        "Upload a CSV containing the 78 original numerical "
        "network traffic features."
    )

    st.info(
        "The Label column is optional and will not be used "
        "for prediction. Additional columns are ignored."
    )

    st.warning(
        "Missing and invalid numerical values are handled "
        "by the preprocessing pipeline. Predictions on "
        "heavily incomplete data may be unreliable."
    )

    uploaded_file = st.file_uploader(
        "Upload network traffic CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            # -----------------------------------------
            # 6.1 READ CSV CONTENT
            # -----------------------------------------

            raw_bytes = uploaded_file.getvalue()

            if len(raw_bytes) == 0:
                st.error("The uploaded file is empty.")
                st.stop()

            try:
                csv_text = raw_bytes.decode("utf-8-sig")

            except UnicodeDecodeError:
                st.error(
                    "Invalid file encoding. "
                    "Please upload a CSV encoded in UTF-8."
                )
                st.stop()

            # -----------------------------------------
            # 6.2 CHECK ORIGINAL DUPLICATE HEADERS
            # -----------------------------------------

            try:
                header_reader = csv.reader(
                    io.StringIO(csv_text)
                )

                original_headers = next(header_reader)

            except StopIteration:
                st.error("The CSV does not contain a header.")
                st.stop()

            except csv.Error as error:
                st.error(f"CSV header error: {error}")
                st.stop()

            duplicated_headers = sorted({
                name
                for name in original_headers
                if original_headers.count(name) > 1
            })

            if duplicated_headers:

                st.error(
                    "Duplicate column names detected. "
                    "Please correct the CSV before prediction."
                )

                st.write(duplicated_headers)
                st.stop()

            # -----------------------------------------
            # 6.3 READ DATAFRAME
            # -----------------------------------------

            df = pd.read_csv(
                io.StringIO(csv_text),
                low_memory=False
            )

            st.subheader("Dataset Preview")

            st.write(f"Rows: {len(df):,}")
            st.write(f"Columns: {len(df.columns)}")

            if df.empty:

                st.warning(
                    "The uploaded CSV contains headers "
                    "but no observations."
                )

                st.stop()

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            # -----------------------------------------
            # 6.4 VALIDATE REQUIRED FEATURES
            # -----------------------------------------

            missing_features = [
                feature
                for feature in expected_features
                if feature not in df.columns
            ]

            if missing_features:

                st.error(
                    f"{len(missing_features)} required "
                    "features are missing."
                )

                st.write(missing_features)

                st.stop()

            st.success(
                f"All {len(expected_features)} required "
                "features are available."
            )

            # -----------------------------------------
            # 6.5 EXTRACT EXPECTED FEATURES
            # -----------------------------------------

            X_original = df[expected_features].copy()

            X = X_original.copy()

            # Convert all features to numerical values
            X = X.apply(
                lambda column: pd.to_numeric(
                    column,
                    errors="coerce"
                )
            )

            # -----------------------------------------
            # 6.6 VALIDATE DATA QUALITY
            # -----------------------------------------

            # Original missing values
            missing_values = int(
                X_original.isna().sum().sum()
            )

            # Non-numeric values converted to NaN
            non_numeric_values = int(
                (
                    X_original.notna() & X.isna()
                ).sum().sum()
            )

            # Infinite values
            infinite_values = int(
                np.isinf(
                    X.to_numpy(dtype=np.float64)
                ).sum()
            )

            # Replace positive and negative infinity
            X = X.replace(
                [np.inf, -np.inf],
                np.nan
            )

            # Total invalid values
            invalid_values = (
                missing_values
                + non_numeric_values
                + infinite_values
            )

            # Features entirely missing
            empty_features = [
                column
                for column in expected_features
                if X[column].isna().all()
            ]

            # Rows with all features missing
            empty_rows = int(
                X.isna().all(axis=1).sum()
            )

            # -----------------------------------------
            # 6.7 DISPLAY VALIDATION RESULTS
            # -----------------------------------------

            st.subheader("Data Quality Validation")

            q1, q2, q3, q4 = st.columns(4)

            q1.metric(
                "Missing Values",
                f"{missing_values:,}"
            )

            q2.metric(
                "Non-numeric Values",
                f"{non_numeric_values:,}"
            )

            q3.metric(
                "Infinite Values",
                f"{infinite_values:,}"
            )

            q4.metric(
                "Total Invalid Values",
                f"{invalid_values:,}"
            )

            if invalid_values > 0:

                st.warning(
                    "Some invalid values were detected. "
                    "The trained SimpleImputer will handle "
                    "the resulting missing values."
                )

            else:

                st.success(
                    "No missing, non-numeric or infinite "
                    "values were detected."
                )

            if empty_features:

                st.warning(
                    f"{len(empty_features)} features contain "
                    "only missing values."
                )

                st.write(empty_features)

            if empty_rows:

                st.warning(
                    f"{empty_rows} rows contain no valid "
                    "feature values."
                )

            # -----------------------------------------
            # 6.8 PREDICTION
            # -----------------------------------------

            if st.button(
                "Analyze Network Traffic",
                type="primary",
                use_container_width=True
            ):

                with st.spinner(
                    "Analyzing network traffic..."
                ):

                    try:

                        predictions = model.predict(X)

                        # ---------------------------------
                        # 6.9 CREATE RESULTS
                        # ---------------------------------

                        results = df.copy()

                        results["Prediction"] = predictions

                        # ---------------------------------
                        # 6.10 SUMMARY
                        # ---------------------------------

                        st.success(
                            "Network traffic analysis completed!"
                        )

                        st.subheader("Analysis Summary")

                        benign_count = int(
                            np.sum(predictions == "BENIGN")
                        )

                        attack_count = int(
                            np.sum(predictions != "BENIGN")
                        )

                        a, b, c = st.columns(3)

                        a.metric(
                            "Total Flows",
                            f"{len(predictions):,}"
                        )

                        b.metric(
                            "BENIGN",
                            f"{benign_count:,}"
                        )

                        c.metric(
                            "Attack Flows",
                            f"{attack_count:,}"
                        )

                        # ---------------------------------
                        # 6.11 DISTRIBUTION
                        # ---------------------------------

                        st.subheader(
                            "Prediction Distribution"
                        )

                        counts = (
                            pd.Series(predictions)
                            .value_counts()
                            .reindex(classes, fill_value=0)
                        )

                        st.bar_chart(counts)

                        # ---------------------------------
                        # 6.12 RESULTS TABLE
                        # ---------------------------------

                        st.subheader(
                            "Classification Results"
                        )

                        st.dataframe(
                            results,
                            use_container_width=True
                        )

                        # ---------------------------------
                        # 6.13 ATTACK TYPES
                        # ---------------------------------

                        st.subheader(
                            "Detected Attack Types"
                        )

                        for label in classes:

                            count = int(
                                np.sum(predictions == label)
                            )

                            if label == "BENIGN":

                                st.success(
                                    f"{label}: {count:,} flows"
                                )

                            else:

                                st.warning(
                                    f"{label}: {count:,} flows"
                                )

                        # ---------------------------------
                        # 6.14 EXPORT RESULTS
                        # ---------------------------------

                        csv_data = results.to_csv(
                            index=False
                        ).encode("utf-8-sig")

                        st.download_button(
                            label="Download Prediction Results",
                            data=csv_data,
                            file_name=(
                                "network_traffic_predictions.csv"
                            ),
                            mime="text/csv",
                            use_container_width=True
                        )

                    except Exception as error:

                        st.error(
                            f"Prediction error: {error}"
                        )

        except pd.errors.EmptyDataError:

            st.error(
                "The uploaded CSV does not contain valid data."
            )

        except pd.errors.ParserError as error:

            st.error(
                f"Invalid CSV structure: {error}"
            )

        except Exception as error:

            st.error(
                f"CSV processing error: {error}"
            )

# =====================================================
# 7. MODEL EVALUATION
# =====================================================

elif page == "Model Evaluation":

    st.header("Final Model Evaluation")

    st.write(
        "Held-out evaluation on the CIC-IDS2017 test dataset."
    )

    c1, c2 = st.columns(2)

    c1.metric("Test Samples", "89,182")
    c2.metric("Incorrect Predictions", "1")

    st.subheader("Confusion Matrix")

    matrix_path = FIGURES_DIR / "02_confusion_matrix.png"

    if matrix_path.exists():

        st.image(
            str(matrix_path),
            use_container_width=True
        )

    else:

        st.warning("Confusion matrix not found.")

    st.subheader("Class Performance")

    performance_path = (
        FIGURES_DIR / "03_class_performance.png"
    )

    if performance_path.exists():

        st.image(
            str(performance_path),
            use_container_width=True
        )

    else:

        st.warning("Class performance figure not found.")

    st.subheader("Destination Port Sensitivity")

    port_path = (
        FIGURES_DIR / "05_destination_port_comparison.png"
    )

    if port_path.exists():

        st.image(
            str(port_path),
            use_container_width=True
        )

    else:

        st.warning("Destination Port comparison not found.")

    st.caption(
        "These results are specific to the CIC-IDS2017 "
        "evaluation dataset. They do not guarantee equivalent "
        "performance on real-world network traffic."
    )

# =====================================================
# 8. ABOUT PROJECT
# =====================================================

elif page == "About Project":

    st.header("About This Project")

    st.markdown("""
    **Project:** Network Traffic Anomaly Detection Using Machine Learning

    **Domain:** Cybersecurity

    **Dataset:** CIC-IDS2017 – Tuesday Working Hours

    **Algorithm:** Random Forest Classifier

    **Framework:** Streamlit

    **Programming Language:** Python

    ### Supported Classes

    - BENIGN
    - FTP-Patator
    - SSH-Patator

    ### Model Pipeline

    1. SimpleImputer (median strategy)
    2. VarianceThreshold
    3. RandomForestClassifier

    ### Main Features

    - CSV network traffic analysis
    - Machine Learning classification
    - Attack traffic identification
    - Data quality validation
    - Scientific performance visualization
    - Prediction results export

    ### Current Limitations

    - Only three traffic classes are supported.
    - PCAP processing is not implemented.
    - Live network monitoring is not implemented.
    - Evaluation is based on CIC-IDS2017 data.
    """)

    st.success(
        "Academic cybersecurity project based on CIC-IDS2017."
    )

# =====================================================
# 9. FOOTER
# =====================================================

st.divider()

st.caption(
    "Network Traffic Anomaly Detection | "
    "Random Forest | CIC-IDS2017 | Academic Project"
)
