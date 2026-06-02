# Conversational AI for Digital Payment Analytics



https://github.com/user-attachments/assets/7aadac82-0756-4589-9e47-8396e358b6ad





## Overview
**Conversational AI for Digital Payment Analytics** is a data-driven full-stack project that turns UPI transaction analytics into a conversational experience. The application enables users to ask questions in natural language and receive automated analysis, visualizations, and predictive insights without writing SQL or Python code.

The system demonstrates an end-to-end pipeline for:
- LLM-guided query interpretation
- Automated analytics and machine learning
- Counterfactual scenario simulation
- Interactive visualization in a modern React UI

## Why this project matters
Financial transaction analytics is often locked behind technical expertise. This project is designed to bridge the gap by making payment analytics accessible to business stakeholders, product owners, and decision-makers through a user-friendly conversational interface.

It is especially relevant for:
- Payment operations teams analyzing success/failure rates
- Fraud and risk teams studying fraud probability patterns
- Product managers evaluating the impact of network, device, and merchant attributes
- Data science teams building scenario-driven business intelligence tools

## What the application does
The system supports a broad set of analytics capabilities, such as:
- Natural language question answering over UPI transaction data
- Category and time-series aggregations
- Correlation heatmaps and categorical heatmaps
- K-Means clustering for user segmentation
- Linear, regularized, and Poisson regression modeling
- Polynomial curve fitting and cross-validation analysis
- A What-If Lab for scenario simulation, sensitivity analysis, and recommendation generation

## Core components
### Backend (`main.py`)
- FastAPI serves REST endpoints for summary statistics, clustering, what-if analysis, and conversational query processing.
- Loads the dataset from `upi_transactions_2024.csv` and exposes analytics endpoints for the frontend.
- Uses LangChain with Groq LLM to parse natural language questions and route them to specialized analytics tools.

### Analytics and ML (`algorithms.py`)
- Implements data aggregation, filtering, and visualization-ready analytics functions.
- Supports K-Means clustering, multivariate regression, regularized regression (Ridge/Lasso), Poisson regression, polynomial curve fitting, and cross-validation.
- Produces structured outputs that can be rendered directly as charts or text summaries.

### What-If Engine (`stats_engine.py`)
- Builds gradient boosting models for success rate, fraud risk, and expected transaction amount.
- Provides a scenario simulator that compares baseline predictions with user-defined overrides.
- Generates sensitivity data for tornado-chart style analysis.
- Recommends parameter combinations that optimize a target metric.

### Frontend (`frontend/`)
- React + TypeScript frontend built with Vite.
- Uses Tailwind CSS for responsive styling.
- Renders charts and dashboards with Recharts.
- Offers an interactive conversational interface for querying and exploring analytics.

## Technical highlights
- LLM-based routing: Natural language questions are converted into tool calls, reducing manual query engineering.
- Multi-model analytics: Combines descriptive analytics, clustering, regression, and tree-based predictive modeling.
- Counterfactual simulation: The What-If Lab enables business-impact analysis by changing input parameters and comparing outcomes.
- Modular architecture: Backend logic is separated into clear modules for data processing (`algorithms.py`), predictive simulation (`stats_engine.py`), and API routing (`main.py`).

## How it works
1. The backend loads the UPI transaction dataset and prepares it for analysis.
2. Users send natural language questions through the frontend chat interface.
3. The LLM interprets the intent and selects the correct analytics tool.
4. The selected Python function executes data aggregation, ML inference, or scenario simulation.
5. Results are returned as structured JSON and displayed as charts or text in the frontend.

## Installation
### Backend
1. From the project root, create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Groq API key:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```
4. Start the API server:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend
1. Navigate to the frontend folder:
   ```bash
   cd frontend
   ```
2. Install packages:
   ```bash
   npm install
   ```
3. Start the frontend:
   ```bash
   npm run dev
   ```

### Dataset
Place `upi_transactions_2024.csv` in the project root so the backend can load it automatically.

## Project structure
- `main.py` — FastAPI app and LLM-driven analytics endpoint routing
- `algorithms.py` — Data aggregation, visualization, and regression utilities
- `stats_engine.py` — What-if modeling, sensitivity analysis, and recommendation engine
- `frontend/` — React TypeScript user interface
- `upi_transactions_2024.csv` — Transaction dataset used for analytics and model training

## Impact and relevance
This project is a strong demonstration of applied data science and machine learning because it:
- Integrates real transactional data with predictive modeling
- Uses modern LLM and agent techniques to simplify analytics workflows
- Includes structured scenario analysis for business decision support
- Demonstrates full-stack delivery from API to interactive front-end

## Future improvements
Potential next steps include:
- adding explicit model explainability outputs for each prediction
- expanding the dataset to support larger scale analysis
- improving the conversational parser for more nuanced financial questions
- adding user authentication and role-based access to analytics
