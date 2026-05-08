import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso, PoissonRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

dataset_path = "upi_transactions_2024.csv"

def _load_data():
    if os.path.exists(dataset_path):
        return pd.read_csv(dataset_path)
    return pd.DataFrame()

def _apply_filters(df, filters):
    if filters:
        for k, v in filters.items():
            if k in df.columns:
                if isinstance(v, list):
                    df = df[df[k].isin(v)]
                else:
                    df = df[df[k] == v]
    return df

def _prepare_data(df, feature_cols, target_col=None):
    cols = feature_cols.copy()
    if target_col:
        cols.append(target_col)
    
    # Check if cols exist
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return None, None, f"Missing columns in dataset: {missing}"
        
    df_clean = df.dropna(subset=cols).copy()
    if df_clean.empty:
        return None, None, "Dataset empty after dropping missing values."
        
    X = pd.get_dummies(df_clean[feature_cols], drop_first=True, dtype=float)
    y = df_clean[target_col] if target_col else None
    
    # Subsample if dataset is too large for fast visual/ML analysis (max 5000 rows to avoid stalling and big payloads)
    if len(X) > 5000:
        if y is not None:
            X, _, y, _ = train_test_split(X, y, train_size=5000, random_state=42)
        else:
            X, _ = train_test_split(X, train_size=5000, random_state=42)
            
    return X, y, None

# ... Existing functions ...
def aggregate_by_category(category_col: str, metric: str = 'count', chart_hint: str = 'bar', filters: dict = None) -> dict:
    df = _load_data()
    if df.empty or category_col not in df.columns:
        return {"error": f"Invalid category column '{category_col}' or empty dataset."}
    
    df = _apply_filters(df, filters)
    df = df.dropna(subset=[category_col])
    
    if metric == 'count':
        agg_df = df[category_col].value_counts().reset_index()
        agg_df.columns = ['name', 'value']
    elif metric == 'sum_amount':
        if 'amount (INR)' not in df.columns: return {"error": "Amount missing."}
        agg_df = df.groupby(category_col)['amount (INR)'].sum().reset_index()
        agg_df.columns = ['name', 'value']
    elif metric == 'failed_count':
        if 'transaction_status' not in df.columns: return {"error": "Status missing."}
        agg_df = df[df['transaction_status'].str.upper() == 'FAILED'][category_col].value_counts().reset_index()
        agg_df.columns = ['name', 'value']
    else: return {"error": f"Unknown metric '{metric}'"}

    agg_df = agg_df.sort_values(by='value', ascending=False).head(10)
    return {"type": "visualization", "chart_type": chart_hint, "data": agg_df.to_dict(orient='records'), "title": f"{metric} by {category_col}"}

def time_series_trend(time_col: str = 'hour_of_day', metric: str = 'count', chart_hint: str = 'line', filters: dict = None) -> dict:
    df = _load_data()
    if df.empty or time_col not in df.columns: return {"error": "Invalid time column."}
    df = _apply_filters(df, filters)
    df = df.dropna(subset=[time_col])
    if metric == 'count':
        agg_df = df.groupby(time_col).size().reset_index(name='value')
        agg_df.rename(columns={time_col: 'name'}, inplace=True)
    elif metric == 'sum_amount':
        agg_df = df.groupby(time_col)['amount (INR)'].sum().reset_index(name='value')
        agg_df.rename(columns={time_col: 'name'}, inplace=True)
    else: return {"error": "Unknown metric."}
    
    if time_col == 'hour_of_day': agg_df = agg_df.sort_values(by='name')
    return {"type": "visualization", "chart_type": chart_hint, "data": agg_df.to_dict(orient='records'), "title": f"Trend over {time_col}"}

def correlation_heatmap(columns: list = None) -> dict:
    df = _load_data()
    if df.empty: return {"error": "Empty dataset"}
    num_df = df.select_dtypes(include=['number'])
    if columns: num_df = num_df[[c for c in columns if c in num_df.columns]]
    corr = num_df.corr()
    data = [{"x": c, "y": r, "value": round(float(corr.iloc[i, j]),2)} for i, r in enumerate(corr.index) for j, c in enumerate(corr.columns)]
    return {"type": "visualization", "chart_type": "heatmap", "data": data, "title": "Correlation Matrix"}

def categorical_heatmap(x_col: str, y_col: str, metric: str = 'count', filters: dict = None) -> dict:
    df = _load_data()
    if df.empty or x_col not in df.columns or y_col not in df.columns: return {"error": "Invalid Columns."}
    df = _apply_filters(df, filters)
    if metric == 'count': agg = df.groupby([x_col, y_col]).size().reset_index(name='value')
    else: agg = df.groupby([x_col, y_col])['amount (INR)'].sum().reset_index(name='value')
    data = [{"x": str(r[x_col]), "y": str(r[y_col]), "value": float(r['value'])} for _, r in agg.iterrows()]
    return {"type": "visualization", "chart_type": "heatmap", "data": data, "title": f"Heatmap {x_col} vs {y_col}"}

# --- New ML Algorithms ---

def k_means_clustering(feature_cols: list, n_clusters: int = 3) -> dict:
    df = _load_data()
    X, _, err = _prepare_data(df, feature_cols)
    if err: return {"error": err}
    if len(X.columns) < 1: return {"error": "Need at least 1 numeric feature."}
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Prep viz data
    data = []
    x_col = X.columns[0]
    y_col = X.columns[1] if len(X.columns) > 1 else x_col
    for idx, row in X.iterrows():
        data.append({"x": float(row[x_col]), "y": float(row[y_col]), "cluster": f"Cluster {clusters[list(X.index).index(idx)]}"})
        
    return {
        "type": "visualization", "chart_type": "scatter", "data": data, 
        "title": f"K-Means Clustering ({n_clusters} clusters)",
        "numerical_results": {"inertia": float(kmeans.inertia_)}
    }

def multivariate_linear_regression(target_col: str, feature_cols: list) -> dict:
    df = _load_data()
    X, y, err = _prepare_data(df, feature_cols, target_col)
    if err: return {"error": err}
    
    model = LinearRegression().fit(X, y)
    preds = model.predict(X)
    
    data = [{"x": float(act), "y": float(pred)} for act, pred in zip(y, preds)]
    return {
        "type": "visualization", "chart_type": "scatter_with_line", "data": data,
        "title": f"Multivariate LinReg: {target_col}",
        "numerical_results": {"r2": float(r2_score(y, preds)), "mse": float(mean_squared_error(y, preds)), "coef": {c: float(v) for c,v in zip(X.columns, model.coef_)}}
    }

def regularized_regression(target_col: str, feature_cols: list, model_type: str = 'ridge', alpha: float = 1.0) -> dict:
    df = _load_data()
    X, y, err = _prepare_data(df, feature_cols, target_col)
    if err: return {"error": err}
    
    if model_type.lower() == 'lasso': model = Lasso(alpha=alpha).fit(X, y)
    else: model = Ridge(alpha=alpha).fit(X, y)
    
    preds = model.predict(X)
    data = [{"x": float(act), "y": float(pred)} for act, pred in zip(y, preds)]
    return {
        "type": "visualization", "chart_type": "scatter_with_line", "data": data,
        "title": f"{model_type.capitalize()} Regression ({target_col})",
        "numerical_results": {"r2": float(r2_score(y, preds)), "mse": float(mean_squared_error(y, preds))}
    }

def poisson_regression(target_col: str, feature_cols: list) -> dict:
    df = _load_data()
    X, y, err = _prepare_data(df, feature_cols, target_col)
    if err: return {"error": err}
    # Poisson requires y >= 0
    if (y < 0).any(): return {"error": "Target must be non-negative for Poisson."}
    
    model = PoissonRegressor().fit(X, y)
    preds = model.predict(X)
    data = [{"x": float(act), "y": float(pred)} for act, pred in zip(y, preds)]
    return {
        "type": "visualization", "chart_type": "scatter_with_line", "data": data,
        "title": f"Poisson Regression ({target_col})",
        "numerical_results": {"score": float(model.score(X,y))}
    }

def training_testing_curves(target_col: str, feature_col: str, max_degree: int = 5) -> dict:
    df = _load_data()
    X, y, err = _prepare_data(df, [feature_col], target_col)
    if err: return {"error": err}
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    data = []
    
    for d in range(1, max_degree + 1):
        poly = PolynomialFeatures(degree=d)
        X_tr_p = poly.fit_transform(X_train)
        X_te_p = poly.transform(X_test)
        
        m = LinearRegression().fit(X_tr_p, y_train)
        r2_tr = r2_score(y_train, m.predict(X_tr_p))
        r2_te = r2_score(y_test, m.predict(X_te_p))
        
        data.append({"degree": d, "train_score": float(r2_tr), "test_score": float(r2_te)})
        
    return {
        "type": "visualization", "chart_type": "line_multiple", "data": data,
        "title": "Training vs Testing Curves (Complexity)",
        "numerical_results": {"max_degree_tested": max_degree}
    }

def polynomial_curve_fitting(target_col: str, feature_col: str, degree: int = 3) -> dict:
    df = _load_data()
    X, y, err = _prepare_data(df, [feature_col], target_col)
    if err: return {"error": err}
    
    poly = PolynomialFeatures(degree=degree)
    X_p = poly.fit_transform(X)
    model = LinearRegression().fit(X_p, y)
    
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    x_range_p = poly.transform(pd.DataFrame(x_range, columns=X.columns))
    preds_range = model.predict(x_range_p)
    
    scatter = [{"x": float(xv), "y": float(yv), "type": "actual"} for xv, yv in zip(X[feature_col], y)]
    curve = [{"x": float(xv[0]), "y": float(yv), "type": "fitted"} for xv, yv in zip(x_range, preds_range)]
    
    return {
        "type": "visualization", "chart_type": "scatter_curve", "data": scatter + curve,
        "title": f"Polynomial Fit (Degree {degree})",
        "numerical_results": {"r2": float(r2_score(y, model.predict(X_p)))}
    }

def cross_validation_analysis(target_col: str, feature_cols: list, cv_folds: int = 5) -> dict:
    df = _load_data()
    X, y, err = _prepare_data(df, feature_cols, target_col)
    if err: return {"error": err}
    
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
    
    data = [{"fold": i+1, "score": float(s)} for i, s in enumerate(scores)]
    return {
        "type": "visualization", "chart_type": "bar", "data": data,
        "title": f"Cross Validation R2 Scores ({cv_folds} folds)",
        "numerical_results": {"mean_r2": float(scores.mean()), "std_r2": float(scores.std())}
    }

def non_linear_relationship(target_col: str, feature_col: str) -> dict:
    df = _load_data()
    X, y, err = _prepare_data(df, [feature_col], target_col)
    if err: return {"error": err}
    
    model = DecisionTreeRegressor(max_depth=4).fit(X, y)
    
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    preds_range = model.predict(pd.DataFrame(x_range, columns=X.columns))
    
    scatter = [{"x": float(xv), "y": float(yv), "type": "actual"} for xv, yv in zip(X[feature_col], y)]
    curve = [{"x": float(xv[0]), "y": float(yv), "type": "fitted"} for xv, yv in zip(x_range, preds_range)]
    
    return {
        "type": "visualization", "chart_type": "scatter_curve", "data": scatter + curve,
        "title": "Non-Linear Fit (Decision Tree)",
        "numerical_results": {"r2": float(model.score(X,y))}
    }
