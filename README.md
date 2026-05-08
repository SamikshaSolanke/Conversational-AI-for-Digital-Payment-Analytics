# Conversational AI for Digital Payment Analytics

## 📌 Project Overview
**Conversational AI for Digital Payment Analytics** is a full-stack web application that empowers users to interactively analyze large-scale UPI (Unified Payments Interface) transaction datasets using natural language. 

Instead of writing complex SQL queries or Python scripts, users can simply ask questions like, *"Show me the correlation between transaction amount and hour of the day,"* or *"What happens to the success rate if more users switch to 5G?"* The system automatically interprets the query, runs the appropriate machine learning algorithms or data manipulations, and returns insightful text and dynamic visualizations.

## ⚠️ The Problem
Analyzing large financial datasets traditionally requires specialized technical skills. Business leaders, product managers, and non-technical stakeholders often face a bottleneck when trying to:
- Extract actionable insights quickly.
- Run "what-if" counterfactual scenarios to guide business decisions.
- Visualize complex relationships between multiple transaction parameters.
- Identify fraud patterns or transaction failure reasons without relying heavily on data engineering teams.

## 💡 Our Solution
We built an intelligent, LLM-powered analytics platform designed specifically for digital payment data. 

**Key Capabilities:**
- **Natural Language Interface:** Chat directly with your transaction data.
- **Automated Machine Learning:** The system automatically routes queries to the correct ML model (e.g., K-Means Clustering, Multivariate Regression, Poisson Regression) based on the context of the question.
- **What-If Lab:** A dedicated simulation engine that allows users to perform sensitivity analysis and counterfactual predictions (e.g., predicting the impact on success rates when network or device types change).
- **Dynamic Visualizations:** Instantly generates tailored charts (heatmaps, bar charts, line graphs, pie charts) using Recharts to make data easily digestible.

## 🔬 Methodology

1. **Data Ingestion & Processing:** The FastAPI backend securely loads and manages the UPI transactions dataset (`upi_transactions_2024.csv`) using Pandas.
2. **Intelligent Query Routing (LLM Layer):** 
   - User queries are sent to a LangChain-based router powered by Groq (Llama-3.3-70b-versatile).
   - The LLM determines if the query requires standard data aggregation, a specific statistical model, or a visualization.
   - It utilizes an agentic tool-calling approach to trigger the exact Python function needed (defined in `algorithms.py` and `stats_engine.py`).
3. **Statistical Engine:** Executes complex operations like logistic regression for what-if scenarios, cross-validation, and polynomial curve fitting.
4. **Interactive Frontend presentation:** The React frontend receives the processed data and renders interactive conversational UI and rich data visualizations.

## 🛠️ Tech Stack

**Backend & Data Science:**
- **Python 3**
- **FastAPI** (High-performance web framework)
- **Pandas** (Data manipulation and analysis)
- **Scikit-learn & Statsmodels** (Machine learning algorithms and statistical modeling)

**AI & LLM:**
- **LangChain & LangChain Experimental** (Agent orchestration and Pandas DataFrame Agent)
- **Groq API** (Ultra-fast LLM inference using Llama-3.3-70b-versatile)

**Frontend:**
- **React 18** (UI Library)
- **Vite** (Build tool)
- **TypeScript** (Static typing)
- **Tailwind CSS** (Styling)
- **Recharts** (Data visualization)

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Node.js & npm
- A Groq API Key

### Backend Setup
1. Navigate to the root directory.
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Mac/Linux: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```
6. Run the FastAPI server: `uvicorn main:app --reload`

### Frontend Setup
1. Navigate to the frontend directory: `cd frontend`
2. Install dependencies: `npm install`
3. Start the development server: `npm run dev`

### Data
Ensure that your `upi_transactions_2024.csv` file is placed in the root directory for the backend to load it successfully at startup.