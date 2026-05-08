import sys
import os

# Add current directory to path so we can import main
sys.path.append(os.getcwd())

from main import chat_with_data, ChatRequest

request = ChatRequest(query="heatmap for the whole data")
print("calling chat_with_data...")
try:
    result = chat_with_data(request)
    print("Result:", result)
except Exception as e:
    import traceback
    traceback.print_exc()
