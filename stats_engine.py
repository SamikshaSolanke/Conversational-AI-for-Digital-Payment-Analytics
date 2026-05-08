import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, r2_score, precision_score, recall_score
import os
import threading

dataset_path = "upi_transactions_2024.csv"

def _load_data():
    if os.path.exists(dataset_path):
        return pd.read_csv(dataset_path)
    return pd.DataFrame()


# =============================================================================
# What-If Engine — Multi-parameter, multi-outcome prediction
# =============================================================================

class WhatIfEngine:
    """
    Trains GradientBoosting models on the UPI dataset and provides:
    - Scenario simulation (baseline vs. modified parameter predictions)
    - Sensitivity analysis (tornado chart data)
    - Feature importance extraction
    """

    # Adjustable categorical parameters (user can change via dropdowns)
    CATEGORICAL_PARAMS = [
        "network_type", "device_type", "merchant_category",
        "sender_age_group", "sender_state", "sender_bank", "transaction type"
    ]

    # Adjustable numerical parameters (user can change via sliders)
    NUMERICAL_PARAMS = ["hour_of_day", "amount (INR)", "is_weekend"]

    # All feature columns used by models
    FEATURE_COLS = CATEGORICAL_PARAMS + NUMERICAL_PARAMS

    # Target definitions
    TARGETS = {
        "success_rate": {"col": "transaction_status", "type": "classification"},
        "fraud_risk": {"col": "fraud_flag", "type": "classification"},
        "expected_amount": {"col": "amount (INR)", "type": "regression"},
    }

    def __init__(self):
        self._models = {}          # {target_name: trained_model}
        self._encoders = {}        # {col_name: LabelEncoder}
        self._feature_names = []   # Column names after encoding
        self._is_trained = False
        self._lock = threading.Lock()
        self._options = {}         # {param_name: [unique_values]}
        self._baseline = {}        # {param_name: baseline_value (mode/mean)}
        self._accuracy = {}        # {target_name: accuracy/r2}
        self._df_clean = None

    def _ensure_trained(self):
        """Lazy initialization — train models on first request."""
        if self._is_trained:
            return
        with self._lock:
            if self._is_trained:
                return
            self._train()

    def _train(self):
        df = _load_data()
        if df.empty:
            raise ValueError("Dataset not found or is empty.")

        # Only keep rows that have all required columns
        required = self.FEATURE_COLS + ["transaction_status", "fraud_flag"]
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        df_clean = df.dropna(subset=required).copy()

        # Subsample for speed (max 30k rows for training)
        if len(df_clean) > 30000:
            df_clean = df_clean.sample(n=30000, random_state=42)

        # --- Compute options and baseline ---
        for col in self.CATEGORICAL_PARAMS:
            if col in df_clean.columns:
                unique_vals = sorted(df_clean[col].dropna().unique().tolist())
                self._options[col] = unique_vals
                self._baseline[col] = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else unique_vals[0]

        for col in self.NUMERICAL_PARAMS:
            if col in df_clean.columns:
                if col == "is_weekend":
                    self._options[col] = [0, 1]
                    self._baseline[col] = int(df_clean[col].mode().iloc[0])
                elif col == "hour_of_day":
                    self._options[col] = {"min": 0, "max": 23, "step": 1}
                    self._baseline[col] = int(round(df_clean[col].mean()))
                elif col == "amount (INR)":
                    self._options[col] = {
                        "min": int(df_clean[col].min()),
                        "max": int(min(df_clean[col].max(), 50000)),  # Cap UI slider
                        "step": 100
                    }
                    self._baseline[col] = int(round(df_clean[col].mean()))

        # --- Encode categoricals ---
        encoded_df = df_clean.copy()
        for col in self.CATEGORICAL_PARAMS:
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
            self._encoders[col] = le

        X = encoded_df[self.FEATURE_COLS].values
        self._feature_names = list(self.FEATURE_COLS)
        self._df_clean = df_clean

        # --- Train models ---
        # 1. Success Rate (classification: SUCCESS=1, else=0)
        y_success = (df_clean["transaction_status"].str.upper() == "SUCCESS").astype(int).values
        clf_success = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, subsample=0.8
        )
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_success, test_size=0.2, random_state=42)
        clf_success.fit(X_tr, y_tr)
        
        y_pred_success = clf_success.predict(X_te)
        cv_success = cross_val_score(clf_success, X, y_success, cv=5, scoring='accuracy')
        
        self._accuracy["success_rate"] = {
            "accuracy": round(accuracy_score(y_te, y_pred_success) * 100, 2),
            "cv_accuracy": round(cv_success.mean() * 100, 2),
            "precision": round(precision_score(y_te, y_pred_success, zero_division=0) * 100, 2),
            "recall": round(recall_score(y_te, y_pred_success, zero_division=0) * 100, 2)
        }
        self._models["success_rate"] = clf_success

        # 2. Fraud Risk (classification: fraud_flag=1 vs 0)
        y_fraud = df_clean["fraud_flag"].astype(int).values
        clf_fraud = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, subsample=0.8
        )
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_fraud, test_size=0.2, random_state=42)
        clf_fraud.fit(X_tr, y_tr)
        
        y_pred_fraud = clf_fraud.predict(X_te)
        cv_fraud = cross_val_score(clf_fraud, X, y_fraud, cv=5, scoring='accuracy')
        
        self._accuracy["fraud_risk"] = {
            "accuracy": round(accuracy_score(y_te, y_pred_fraud) * 100, 2),
            "cv_accuracy": round(cv_fraud.mean() * 100, 2),
            "precision": round(precision_score(y_te, y_pred_fraud, zero_division=0) * 100, 2),
            "recall": round(recall_score(y_te, y_pred_fraud, zero_division=0) * 100, 2)
        }
        self._models["fraud_risk"] = clf_fraud

        # 3. Expected Amount (regression)
        # For amount prediction, use only categorical + hour + is_weekend as features
        amount_features = [c for c in self.FEATURE_COLS if c != "amount (INR)"]
        amount_feature_indices = [self.FEATURE_COLS.index(c) for c in amount_features]
        X_amount = X[:, amount_feature_indices]
        y_amount = df_clean["amount (INR)"].values
        reg_amount = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, subsample=0.8
        )
        X_tr, X_te, y_tr, y_te = train_test_split(X_amount, y_amount, test_size=0.2, random_state=42)
        reg_amount.fit(X_tr, y_tr)
        self._accuracy["expected_amount"] = round(r2_score(y_te, reg_amount.predict(X_te)), 4)
        self._models["expected_amount"] = reg_amount
        self._amount_feature_indices = amount_feature_indices

        self._is_trained = True

    def _encode_scenario(self, params: dict) -> np.ndarray:
        """Encode a scenario dict into a feature vector matching model input."""
        row = []
        for col in self.FEATURE_COLS:
            val = params.get(col, self._baseline.get(col))
            if col in self._encoders:
                le = self._encoders[col]
                val_str = str(val)
                if val_str in le.classes_:
                    row.append(le.transform([val_str])[0])
                else:
                    # Unknown category — use baseline
                    row.append(le.transform([str(self._baseline[col])])[0])
            else:
                row.append(float(val))
        return np.array(row).reshape(1, -1)

    def get_options(self) -> dict:
        """Return parameter names, types, possible values, and baseline values."""
        self._ensure_trained()
        params = []
        for col in self.CATEGORICAL_PARAMS:
            if col in self._options:
                params.append({
                    "name": col,
                    "type": "categorical",
                    "values": self._options[col],
                    "baseline": self._baseline[col],
                })
        for col in self.NUMERICAL_PARAMS:
            if col in self._options:
                params.append({
                    "name": col,
                    "type": "numerical",
                    "range": self._options[col] if isinstance(self._options[col], dict) else None,
                    "values": self._options[col] if isinstance(self._options[col], list) else None,
                    "baseline": self._baseline[col],
                })
        return {"parameters": params}

    def simulate(self, overrides: dict) -> dict:
        """
        Run simulation with parameter overrides.
        Returns baseline vs. simulated predictions for all 3 targets.
        """
        self._ensure_trained()

        baseline_vec = self._encode_scenario({})  # All baseline values
        sim_vec = self._encode_scenario(overrides)  # With user overrides

        results = {}

        # Success Rate
        clf_success = self._models["success_rate"]
        base_prob = float(clf_success.predict_proba(baseline_vec)[0, 1]) * 100
        sim_prob = float(clf_success.predict_proba(sim_vec)[0, 1]) * 100
        results["success_rate"] = {
            "baseline": round(base_prob, 2),
            "simulated": round(sim_prob, 2),
            "delta": round(sim_prob - base_prob, 2),
            "unit": "%",
            "model_metrics": self._accuracy.get("success_rate"),
        }

        # Fraud Risk
        clf_fraud = self._models["fraud_risk"]
        base_fraud = float(clf_fraud.predict_proba(baseline_vec)[0, 1]) * 100
        sim_fraud = float(clf_fraud.predict_proba(sim_vec)[0, 1]) * 100
        results["fraud_risk"] = {
            "baseline": round(base_fraud, 2),
            "simulated": round(sim_fraud, 2),
            "delta": round(sim_fraud - base_fraud, 2),
            "unit": "%",
            "model_metrics": self._accuracy.get("fraud_risk"),
        }

        # Expected Amount (uses subset of features, excludes amount itself)
        reg_amount = self._models["expected_amount"]
        base_amount_vec = baseline_vec[:, self._amount_feature_indices]
        sim_amount_vec = sim_vec[:, self._amount_feature_indices]
        base_amount = float(reg_amount.predict(base_amount_vec)[0])
        sim_amount = float(reg_amount.predict(sim_amount_vec)[0])
        results["expected_amount"] = {
            "baseline": round(base_amount, 2),
            "simulated": round(sim_amount, 2),
            "delta": round(sim_amount - base_amount, 2),
            "unit": "INR",
            "model_r2": self._accuracy.get("expected_amount"),
        }

        # Feature importances
        importances = {}
        for target_name, model in self._models.items():
            if target_name == "expected_amount":
                feat_names = [self.FEATURE_COLS[i] for i in self._amount_feature_indices]
            else:
                feat_names = self._feature_names
            imp = model.feature_importances_
            importances[target_name] = [
                {"feature": feat_names[i], "importance": round(float(imp[i]), 4)}
                for i in range(len(imp))
            ]
            importances[target_name].sort(key=lambda x: x["importance"], reverse=True)

        results["feature_importances"] = importances
        results["overrides_applied"] = overrides

        return results

    def sensitivity(self, target: str = "success_rate") -> dict:
        """
        For each adjustable parameter, sweep through all possible values
        and compute the range of impact on the target metric.
        Returns data suitable for a tornado chart.
        """
        self._ensure_trained()

        if target not in self._models:
            return {"error": f"Unknown target '{target}'. Choose from: {list(self._models.keys())}"}

        model = self._models[target]
        is_classification = target in ("success_rate", "fraud_risk")

        baseline_vec = self._encode_scenario({})

        # For amount model, use subset features
        if target == "expected_amount":
            baseline_pred_vec = baseline_vec[:, self._amount_feature_indices]
        else:
            baseline_pred_vec = baseline_vec

        if is_classification:
            baseline_pred = float(model.predict_proba(baseline_pred_vec)[0, 1]) * 100
        else:
            baseline_pred = float(model.predict(baseline_pred_vec)[0])

        sensitivity_data = []

        # Sweep categorical parameters
        for col in self.CATEGORICAL_PARAMS:
            if col not in self._options:
                continue
            values = self._options[col]
            preds = []
            for val in values:
                vec = self._encode_scenario({col: val})
                if target == "expected_amount":
                    pred_vec = vec[:, self._amount_feature_indices]
                else:
                    pred_vec = vec
                if is_classification:
                    p = float(model.predict_proba(pred_vec)[0, 1]) * 100
                else:
                    p = float(model.predict(pred_vec)[0])
                preds.append(p)

            if preds:
                sensitivity_data.append({
                    "parameter": col,
                    "min_value": round(min(preds), 2),
                    "max_value": round(max(preds), 2),
                    "range": round(max(preds) - min(preds), 2),
                    "baseline": round(baseline_pred, 2),
                })

        # Sweep numerical parameters
        for col in self.NUMERICAL_PARAMS:
            if col not in self._options:
                continue
            opt = self._options[col]
            if isinstance(opt, dict):
                sweep_vals = np.linspace(opt["min"], opt["max"], min(10, opt["max"] - opt["min"] + 1))
            elif isinstance(opt, list):
                sweep_vals = opt
            else:
                continue

            preds = []
            for val in sweep_vals:
                vec = self._encode_scenario({col: val})
                if target == "expected_amount":
                    pred_vec = vec[:, self._amount_feature_indices]
                else:
                    pred_vec = vec
                if is_classification:
                    p = float(model.predict_proba(pred_vec)[0, 1]) * 100
                else:
                    p = float(model.predict(pred_vec)[0])
                preds.append(p)

            if preds:
                sensitivity_data.append({
                    "parameter": col,
                    "min_value": round(min(preds), 2),
                    "max_value": round(max(preds), 2),
                    "range": round(max(preds) - min(preds), 2),
                    "baseline": round(baseline_pred, 2),
                })

        # Sort by range descending (most impactful first)
        sensitivity_data.sort(key=lambda x: x["range"], reverse=True)

        return {
            "target": target,
            "baseline_value": round(baseline_pred, 2),
            "sensitivity": sensitivity_data,
        }

    def recommend(self, target: str = "success_rate", goal: str = "maximize") -> dict:
        """
        Find the best parameter combination to optimize a target metric.
        Uses random search over 500 combinations.
        Returns top 3 best + worst 1 scenario.
        """
        self._ensure_trained()

        if target not in self._models:
            return {"error": f"Unknown target '{target}'."}

        model = self._models[target]
        is_classification = target in ("success_rate", "fraud_risk")
        maximize = goal == "maximize"

        import random
        rng = random.Random(42)

        # Generate 500 random parameter combinations
        candidates = []
        for _ in range(500):
            combo = {}
            for col in self.CATEGORICAL_PARAMS:
                if col in self._options:
                    combo[col] = rng.choice(self._options[col])
            for col in self.NUMERICAL_PARAMS:
                if col in self._options:
                    opt = self._options[col]
                    if isinstance(opt, list):
                        combo[col] = rng.choice(opt)
                    elif isinstance(opt, dict):
                        combo[col] = rng.randrange(opt["min"], opt["max"] + 1, opt.get("step", 1))
            # Predict
            vec = self._encode_scenario(combo)
            if target == "expected_amount":
                pred_vec = vec[:, self._amount_feature_indices]
            else:
                pred_vec = vec
            if is_classification:
                pred = float(model.predict_proba(pred_vec)[0, 1]) * 100
            else:
                pred = float(model.predict(pred_vec)[0])

            candidates.append({"overrides": combo, "predicted": round(pred, 2)})

        # Sort
        candidates.sort(key=lambda x: x["predicted"], reverse=maximize)

        # Baseline prediction
        baseline_vec = self._encode_scenario({})
        if target == "expected_amount":
            bv = baseline_vec[:, self._amount_feature_indices]
        else:
            bv = baseline_vec
        if is_classification:
            baseline_pred = round(float(model.predict_proba(bv)[0, 1]) * 100, 2)
        else:
            baseline_pred = round(float(model.predict(bv)[0]), 2)

        # Top 3 best
        best = []
        for c in candidates[:3]:
            delta = round(c["predicted"] - baseline_pred, 2)
            # Build human-readable summary
            key_changes = []
            for k, v in c["overrides"].items():
                bl = self._baseline.get(k)
                if str(v) != str(bl):
                    key_changes.append(f"{k}={v}")
            summary = ", ".join(key_changes[:5]) if key_changes else "baseline values"
            best.append({
                "overrides": c["overrides"],
                "predicted": c["predicted"],
                "delta": delta,
                "summary": summary,
            })

        # Worst 1
        worst_c = candidates[-1]
        worst_delta = round(worst_c["predicted"] - baseline_pred, 2)
        worst_changes = []
        for k, v in worst_c["overrides"].items():
            bl = self._baseline.get(k)
            if str(v) != str(bl):
                worst_changes.append(f"{k}={v}")

        worst = {
            "overrides": worst_c["overrides"],
            "predicted": worst_c["predicted"],
            "delta": worst_delta,
            "summary": ", ".join(worst_changes[:5]) if worst_changes else "baseline values",
        }

        # Generate text explanation
        top = best[0]
        unit = "%" if is_classification else "INR"
        explanation = (
            f"To {goal} {target.replace('_', ' ')}, the optimal configuration is: "
            f"{top['summary']}. Predicted value: {top['predicted']}{unit} "
            f"({'↑' if top['delta'] > 0 else '↓'}{abs(top['delta'])}{unit} vs baseline)."
        )

        return {
            "target": target,
            "goal": goal,
            "baseline": baseline_pred,
            "best": best,
            "worst": worst,
            "explanation": explanation,
        }


# Singleton instance
_engine = WhatIfEngine()

def get_whatif_engine() -> WhatIfEngine:
    return _engine


# =============================================================================
# Legacy functions (backward compatible)
# =============================================================================

def calculate_what_if(target_network: str) -> dict:
    """
    Performs a counterfactual simulation using Logistic Regression.
    Predicts probability of 'SUCCESS' based on network_type, amount (INR), and device_type.
    """
    df = _load_data()
    if df.empty:
        return {"error": "Dataset not found or is empty."}
        
    # We only care about predicting based on network_type, amount (INR), device_type
    required_cols = ["network_type", "amount (INR)", "device_type", "transaction_status"]
    if not all(col in df.columns for col in required_cols):
        return {"error": "Missing required columns in dataset."}
        
    # Clean data: drop rows with missing values in required columns
    clean_df = df.dropna(subset=required_cols).copy()
    if clean_df.empty:
        return {"error": "No valid data available after dropping NaNs."}
        
    # Create target variable: 1 if SUCCESS, 0 otherwise
    clean_df["target"] = (clean_df["transaction_status"].str.upper() == "SUCCESS").astype(int)
    
    # Identify users on 3G or WiFi for the baseline
    is_3g_wifi = clean_df["network_type"].isin(["3G", "WiFi"])
    subset_3g_wifi = clean_df[is_3g_wifi].copy()
    
    if subset_3g_wifi.empty:
        return {"error": "No data found for users on 3G or WiFi to establish baseline."}
        
    # Calculate baseline success rate for 3G/WiFi users
    baseline_success_rate = subset_3g_wifi["target"].mean() * 100
    
    # --- Modeling ---
    X_cat = pd.get_dummies(clean_df[["network_type", "device_type"]], drop_first=True, dtype=float)
    X_num = clean_df[["amount (INR)"]]
    
    X = pd.concat([X_num, X_cat], axis=1)
    X = sm.add_constant(X)
    y = clean_df["target"]
    
    try:
        model = sm.Logit(y, X).fit(disp=0)
    except Exception as e:
        return {"error": f"Model fitting failed: {str(e)}"}
        
    # --- Simulation (What-If) ---
    sim_df = subset_3g_wifi[["amount (INR)", "network_type", "device_type"]].copy()
    sim_df["network_type"] = target_network
    
    sim_cat = pd.get_dummies(sim_df[["network_type", "device_type"]], drop_first=True, dtype=float)
    sim_num = sim_df[["amount (INR)"]]
    
    sim_X = pd.concat([sim_num, sim_cat], axis=1)
    sim_X = sim_X.reindex(columns=X.columns.drop('const', errors='ignore'), fill_value=0.0)
    sim_X = sm.add_constant(sim_X, has_constant='add')
    
    try:
        predicted_probs = model.predict(sim_X)
        simulated_success_rate = predicted_probs.mean() * 100
    except Exception as e:
         return {"error": f"Prediction failed: {str(e)}"}
         
    return {
        "baseline_network": "3G/WiFi",
        "target_network": target_network,
        "baseline_success_rate_percent": round(baseline_success_rate, 2),
        "simulated_success_rate_percent": round(simulated_success_rate, 2),
        "absolute_improvement_percent": round(simulated_success_rate - baseline_success_rate, 2)
    }

def get_user_clusters() -> dict:
    """
    Groups transactions based on amount (INR) and hour_of_day into 3 clusters using K-Means.
    """
    df = _load_data()
    if df.empty:
        return {"error": "Dataset not found or is empty."}
        
    required_cols = ["amount (INR)", "hour_of_day"]
    if not all(col in df.columns for col in required_cols):
         return {"error": "Missing required columns in dataset."}
         
    clean_df = df.dropna(subset=required_cols).copy()
    if clean_df.empty:
         return {"error": "No valid data available after dropping NaNs."}
         
    X = clean_df[required_cols]
    
    try:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clean_df["cluster"] = kmeans.fit_predict(X)
    except Exception as e:
        return {"error": f"Clustering failed: {str(e)}"}
        
    summary = clean_df.groupby("cluster").agg(
        transaction_count=("amount (INR)", "count"),
        avg_amount_inr=("amount (INR)", "mean"),
        avg_hour_of_day=("hour_of_day", "mean")
    ).round(2).reset_index()
    
    clusters_list = []
    for _, row in summary.iterrows():
        clusters_list.append({
            "cluster_id": int(row["cluster"]),
            "transaction_count": int(row["transaction_count"]),
            "avg_amount_inr": float(row["avg_amount_inr"]),
            "avg_hour_of_day": float(row["avg_hour_of_day"])
        })
        
    return {"clusters": clusters_list}
