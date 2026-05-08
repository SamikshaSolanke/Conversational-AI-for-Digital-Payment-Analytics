import sys
import os

# Add current directory to path so we can import main
sys.path.append(os.getcwd())

from main import aggregate_by_category, time_series_trend, generate_generic_chart
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
tools = [aggregate_by_category, time_series_trend, generate_generic_chart]
print("binding tools...")
llm_with_tools = llm.bind_tools(tools)
print("invoking...")
try:
    msg = llm_with_tools.invoke([
        ("system", "You are an expert data visualization assistant."),
        ("user", "count transactions by network_type")
    ])
    print(msg)
except Exception as e:
    import traceback
    traceback.print_exc()
