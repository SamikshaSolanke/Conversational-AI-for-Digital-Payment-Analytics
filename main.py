from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from stats_engine import calculate_what_if, get_user_clusters, get_whatif_engine
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import pandas as pd
import os
import algorithms
from langchain.tools import tool

# Load environment variables from .env file securely
load_dotenv()

app = FastAPI()

# Enable CORS for future React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dataset_path = "upi_transactions_2024.csv"

# Load the dataset at startup
if os.path.exists(dataset_path):
    # Depending on format of dataset, there should be no unexpected space in column names from what we probed.
    df = pd.read_csv(dataset_path)
else:
    df = pd.DataFrame() # Fallback if file doesn't exist

@app.get("/api/summary")
def get_summary():
    """
    Returns basic summary statistics of the dataset:
    - total transactions
    - overall success rate
    - average amount
    """
    if df.empty:
        return {"error": "Dataset not found or is empty."}
        
    total_transactions = int(df.shape[0])
    
    # Calculate overall success rate
    if "transaction_status" in df.columns:
        success_count = int(df[df["transaction_status"].str.upper() == "SUCCESS"].shape[0])
        success_rate = round((success_count / total_transactions) * 100, 2) if total_transactions > 0 else 0.0
    else:
        success_rate = None
        
    # Calculate average amount
    if "amount (INR)" in df.columns:
        average_amount = round(float(df["amount (INR)"].mean()), 2)
    else:
        average_amount = None
        
    return {
        "total_transactions": total_transactions,
        "overall_success_rate": success_rate,
        "average_amount": average_amount
    }

class WhatIfRequest(BaseModel):
    target_network: str

@app.post("/api/what-if")
def what_if_analysis(request: WhatIfRequest):
    """
    Performs a counterfactual using Logistic Regression to predict success rate
    for 3G/WiFi users if they switched to the specified target_network.
    """
    return calculate_what_if(request.target_network)

@app.post("/api/clusters")
def cluster_analysis():
    """
    Groups transactions based on amount (INR) and hour_of_day using K-Means into 3 clusters.
    """
    return get_user_clusters()

# --- What-If Lab Endpoints ---

class WhatIfSimulateRequest(BaseModel):
    overrides: dict = {}

class WhatIfSensitivityRequest(BaseModel):
    target: str = "success_rate"

@app.get("/api/what-if/options")
def whatif_options():
    """
    Returns all adjustable parameters with their possible values,
    types (categorical/numerical), and baseline values.
    """
    try:
        engine = get_whatif_engine()
        return engine.get_options()
    except Exception as e:
        return {"error": f"Failed to load options: {str(e)}"}

@app.post("/api/what-if/simulate")
def whatif_simulate(request: WhatIfSimulateRequest):
    """
    Accepts parameter overrides and returns baseline vs. simulated predictions
    for success rate, fraud probability, and expected amount.
    """
    try:
        engine = get_whatif_engine()
        return engine.simulate(request.overrides)
    except Exception as e:
        return {"error": f"Simulation failed: {str(e)}"}

@app.post("/api/what-if/sensitivity")
def whatif_sensitivity(request: WhatIfSensitivityRequest):
    """
    For a given target metric, returns sensitivity data showing how each
    parameter affects it — suitable for tornado chart visualization.
    """
    try:
        engine = get_whatif_engine()
        return engine.sensitivity(request.target)
    except Exception as e:
        return {"error": f"Sensitivity analysis failed: {str(e)}"}

class WhatIfNaturalRequest(BaseModel):
    query: str

@app.post("/api/what-if/natural")
def whatif_natural(request: WhatIfNaturalRequest):
    """
    Takes a natural language what-if query, uses LLM to parse it into
    parameter overrides, then runs simulation.
    """
    try:
        engine = get_whatif_engine()
        options = engine.get_options()

        # Build a structured prompt for the LLM
        params_desc = []
        for p in options["parameters"]:
            if p["type"] == "categorical":
                params_desc.append(f"  - {p['name']}: one of {p['values']}")
            elif p.get("range"):
                params_desc.append(f"  - {p['name']}: number from {p['range']['min']} to {p['range']['max']}")
            elif p.get("values"):
                params_desc.append(f"  - {p['name']}: one of {p['values']}")

        params_text = "\n".join(params_desc)

        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        system_prompt = (
            "You are a parameter extraction assistant. The user will ask a what-if question about UPI transactions. "
            "Extract the parameter overrides from their question and return ONLY a valid JSON object.\n\n"
            f"Available parameters:\n{params_text}\n\n"
            "Rules:\n"
            "- Only include parameters that the user explicitly mentions or implies\n"
            "- Use exact values from the allowed lists\n"
            "- For numerical values, use integers\n"
            "- Return ONLY the JSON object, no other text\n"
            "- If the user mentions '5G', set network_type to '5G'\n"
            "- If the user mentions 'iPhone' or 'Apple', set device_type to 'iOS'\n"
            "- If the user mentions 'weekend', set is_weekend to 1\n\n"
            'Example: {"network_type": "5G", "device_type": "iOS"}'
        )

        response = llm.invoke([
            ("system", system_prompt),
            ("user", request.query)
        ])

        # Parse the LLM response as JSON
        import json
        raw = response.content.strip()
        # Handle markdown code blocks
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        overrides = json.loads(raw)

        # Run simulation with parsed overrides
        sim_result = engine.simulate(overrides)

        return {
            "query": request.query,
            "parsed_overrides": overrides,
            "simulation": sim_result,
            "interpretation": f"Interpreted as: {', '.join(f'{k}={v}' for k, v in overrides.items())}"
        }
    except json.JSONDecodeError:
        return {"error": "Could not parse the query into parameters. Please try rephrasing."}
    except Exception as e:
        return {"error": f"Natural language processing failed: {str(e)}"}

class WhatIfRecommendRequest(BaseModel):
    target: str = "success_rate"
    goal: str = "maximize"

@app.post("/api/what-if/recommend")
def whatif_recommend(request: WhatIfRecommendRequest):
    """
    Finds optimal parameter combinations to maximize or minimize a target metric.
    Returns top 3 best + worst 1 scenario.
    """
    try:
        engine = get_whatif_engine()
        return engine.recommend(request.target, request.goal)
    except Exception as e:
        return {"error": f"Recommendation failed: {str(e)}"}

# --- Conversational AI Layer ---

def get_pandas_agent(dataframe: pd.DataFrame):
    """
    Initializes a LangChain Pandas DataFrame agent using Google Gemini.
    """
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    agent_executor = create_pandas_dataframe_agent(
        llm,
        dataframe,
        verbose=True,
        allow_dangerous_code=True, # Required in newer LangChain versions to execute pandas code
        handle_parsing_errors=True # Stop crash loops if LLM outputs raw code
    )
    return agent_executor

@tool
def aggregate_by_category(category_col: str, metric: str = 'count', chart_hint: str = 'bar', filters: dict = None) -> dict:
    """
    Aggregates data by a specific category column to create a visualization.
    Use this when the user asks for counts or sums grouped by a category (like state, merchant category, bank, age group).
    You can optionally apply filters to narrow down the dataset first, e.g., {"sender_age_group": "46-55"}.
    Allowed category_cols: 'sender_state', 'merchant_category', 'sender_bank', 'receiver_bank', 'sender_age_group', 'receiver_age_group', 'device_type', 'network_type', 'transaction_status'.
    Allowed metrics: 'count', 'sum_amount', 'failed_count'.
    Allowed chart_hints: 'bar', 'pie'.
    """
    return algorithms.aggregate_by_category(category_col, metric, chart_hint, filters)

@tool
def time_series_trend(time_col: str = 'hour_of_day', metric: str = 'count', chart_hint: str = 'line', filters: dict = None) -> dict:
    """
    Shows trends over time to create a visualization.
    Use this when the user asks for data over hours or days.
    You can optionally apply filters to narrow down the dataset first, e.g., {"sender_age_group": "46-55"}.
    Allowed time_cols: 'hour_of_day', 'day_of_week'.
    Allowed metrics: 'count', 'sum_amount'.
    Allowed chart_hints: 'line', 'bar'.
    """
    return algorithms.time_series_trend(time_col, metric, chart_hint, filters)

import json

@tool
def correlation_heatmap(columns: list = None) -> dict:
    """
    Computes a correlation matrix for numerical columns to return real heatmap visualization data.
    Use this when the user asks for a heatmap or correlation matrix.
    :param columns: Optional list of numerical column names. If None, considers all numerical columns.
    """
    return algorithms.correlation_heatmap(columns)

@tool
def categorical_heatmap(x_col: str, y_col: str, metric: str = 'count', filters: dict = None) -> dict:
    """
    Computes a cross-tabulation heatmap for two categorical or time-based columns.
    Use this when user asks for a heatmap showing relationship between X and Y (e.g. Hour vs Merchant Category).
    You can optionally apply filters to narrow down the dataset first, e.g., {"sender_age_group": "46-55"}.
    Allowed metrics: 'count', 'sum_amount'.
    """
    return algorithms.categorical_heatmap(x_col, y_col, metric, filters)

@tool
def k_means_clustering(feature_cols: list, n_clusters: int = 3) -> dict:
    """Groups data into clusters based on features using K-Means. Outputs clusters, inertia, and visual groupings."""
    return algorithms.k_means_clustering(feature_cols, n_clusters)

@tool
def multivariate_linear_regression(target_col: str, feature_cols: list) -> dict:
    """Predicts a continuous target numeric column based on multiple features using Multivariate Linear Regression."""
    return algorithms.multivariate_linear_regression(target_col, feature_cols)

@tool
def regularized_regression(target_col: str, feature_cols: list, model_type: str = 'ridge', alpha: float = 1.0) -> dict:
    """Predicts regularized target using Ridge Regression or Lasso Regression to prevent overfitting. model_type can be 'ridge' or 'lasso'."""
    return algorithms.regularized_regression(target_col, feature_cols, model_type, alpha)

@tool
def poisson_regression(target_col: str, feature_cols: list) -> dict:
    """Used for modeling count data predictions using Poisson Regression."""
    return algorithms.poisson_regression(target_col, feature_cols)

@tool
def training_testing_curves(target_col: str, feature_col: str, max_degree: int = 5) -> dict:
    """Demonstrates Theory of Generalization showing training testing curves (training vs testing error) for different complexity degrees."""
    return algorithms.training_testing_curves(target_col, feature_col, max_degree)

@tool
def polynomial_curve_fitting(target_col: str, feature_col: str, degree: int = 3) -> dict:
    """Demonstrates a Case Study of Polynomial Curve Fitting for non-linear relationships."""
    return algorithms.polynomial_curve_fitting(target_col, feature_col, degree)

@tool
def cross_validation_analysis(target_col: str, feature_cols: list, cv_folds: int = 5) -> dict:
    """Evaluates the stability and accuracy of a model using Cross validation."""
    return algorithms.cross_validation_analysis(target_col, feature_cols, cv_folds)

@tool
def non_linear_relationship(target_col: str, feature_col: str) -> dict:
    """Models a non linear relationship between feature and target using a decision tree."""
    return algorithms.non_linear_relationship(target_col, feature_col)

ML_TOOLS = [
    aggregate_by_category, time_series_trend, correlation_heatmap, categorical_heatmap,
    k_means_clustering, multivariate_linear_regression, regularized_regression,
    poisson_regression, training_testing_curves, polynomial_curve_fitting,
    cross_validation_analysis, non_linear_relationship
]

class ChatRequest(BaseModel):
    query: str

import traceback

@app.post("/api/chat")
def chat_with_data(request: ChatRequest):
    """
    Routes a user's natural language question:
    1. If it needs a visualization or an ML algorithm, it calls one from algorithms.py.
    2. Otherwise, it falls back to the Pandas dataframe agent.
    """
    if df.empty:
        return {"error": "Dataset not found or is empty."}
        
    try:
        # Step 1: LLM Router to check if we should visualize or run algorithms
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        llm_with_tools = llm.bind_tools(ML_TOOLS)
        
        system_msg = (
            f"You are an expert data visualization and machine learning assistant. "
            f"The dataset has the following columns: {', '.join(df.columns.tolist())}. "
            "You have access to a suite of tools that run ML algorithms on the dataset. "
            "If the user asks for a specific algorithm (k-means, linear regression, ridge/lasso, poisson, cross-validation, polynomial curve fitting, training/testing curves, non-linear relationships), strictly use the corresponding tool. "
            "If the user asks for basic visualisations like bar, pie, line, or heatmaps, use those tools. "
            "For heatmaps (e.g., 'customer spending and commodity' or 'age and merchant'), usually map them to 'sender_age_group' or 'merchant_category' and use metric='sum_amount'. "
            "Ensure you pass the correct columns (numeric where required by ML models). "
            "If the user specifies multiple items, use the tool that best fits."
        )
        
        msg = llm_with_tools.invoke([
            ("system", system_msg),
            ("user", request.query)
        ])
        
        if msg.tool_calls:
            tool_call = msg.tool_calls[0]
            tool_dict = {t.name: t for t in ML_TOOLS}
            if tool_call['name'] in tool_dict:
                result = tool_dict[tool_call['name']].invoke(tool_call['args'])
                if "error" not in result:
                     # Fetch text description from Pandas Agent so we can provide both text + viz
                     agent = get_pandas_agent(df)
                     prompt = (
                         f"You are a helpful data analyst. The user asked: '{request.query}'. "
                         f"I am showing them a visualization for this. "
                         f"Please provide a helpful textual analysis and explicitly answer their question STRICTLY based on the provided DataFrame."
                     )
                     try:
                         agent_res = agent.invoke(prompt)
                         result["text"] = agent_res.get("output", str(agent_res))
                     except Exception as e:
                         print(f"Agent warning: {e}")
                     return {"query": request.query, "response": result}
                     
        # Step 2: Fallback to Pandas Agent for general questions
        agent = get_pandas_agent(df)
        
        prompt = (
            f"You are a helpful data analyst. Please answer the user's question STRICTLY based on the provided DataFrame. "
            f"Think step-by-step and provide your final answer clearly.\n\n"
            f"User Question: {request.query}"
        )
        
        response = agent.invoke(prompt)
        
        return {
            "query": request.query,
            "response": response.get("output", str(response))
        }
    except Exception as e:
        return {"error": f"Failed to process query: {str(e)}\n{traceback.format_exc()}"}
