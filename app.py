import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

# ================================
# Page Config
# ================================
st.set_page_config(page_title="Terrorism ML Predictor", layout="wide")

st.title("🧠 Terrorism Outcome Prediction ML App")
st.markdown("Upload terrorism dataset, train models, and view results interactively.")

# ================================
# Sidebar - Upload Dataset
# ================================
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# ================================
# Main App Logic
# ================================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding="ISO-8859-1", low_memory=False)
    st.success(f"✅ Dataset loaded! Shape: {df.shape}")
    st.write("### Preview of Data")
    st.dataframe(df.head())

    # ================================
    # Preprocessing
    # ================================
    target_col = "success"
    features = ["nkill", "nwound", "suicide", "attacktype1", "targtype1", "weaptype1"]

    if target_col not in df.columns:
        st.error(f"❌ Target column '{target_col}' not found in dataset!")
    else:
        y = df[target_col]
        X = df[features].fillna(0)

        # scale numeric
        scaler = StandardScaler()
        X[["nkill", "nwound"]] = scaler.fit_transform(X[["nkill", "nwound"]])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        st.write("✅ Data Preprocessing Done")

        # ================================
        # Model Training
        # ================================
        models = {
            "Logistic Regression": LogisticRegression(max_iter=500, n_jobs=-1, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"),
            "XGBoost": XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss"
            )
        }

        st.sidebar.header("Select Model")
        model_choice = st.sidebar.selectbox("Choose a model", list(models.keys()))

        if st.sidebar.button("🚀 Train Model"):
            model = models[model_choice]
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

            # Metrics
            auc = roc_auc_score(y_test, probs)
            st.subheader(f"📊 {model_choice} Results")
            st.write(f"**ROC AUC:** {auc:.3f}")

            st.text("Classification Report")
            st.text(classification_report(y_test, preds, zero_division=0))

            # Confusion Matrix
            cm = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"{model_choice} - Confusion Matrix")
            st.pyplot(fig)

            # ROC Curve
            fig2, ax2 = plt.subplots()
            RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax2)
            st.pyplot(fig2)

            # Feature Importance (for tree models)
            if model_choice in ["Random Forest", "XGBoost"]:
                st.subheader("🔑 Feature Importance")
                importances = model.feature_importances_
                feat_imp = pd.DataFrame({"Feature": X.columns, "Importance": importances})
                feat_imp = feat_imp.sort_values(by="Importance", ascending=False)

                fig3, ax3 = plt.subplots()
                sns.barplot(x="Importance", y="Feature", data=feat_imp, ax=ax3, palette="viridis")
                ax3.set_title(f"{model_choice} - Feature Importance")
                st.pyplot(fig3)

else:
    st.info("📂 Please upload a terrorism dataset CSV file to continue.")
