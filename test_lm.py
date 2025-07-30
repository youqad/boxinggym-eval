import os
import sys
import requests

# Add the repository to the path so we can import from it
sys.path.append('.')

# Find the LMExperimenter class (replace with actual import path)
# You'll need to adjust this import based on your repo structure
from your.module.path import LMExperimenter

# Check if vLLM server is running
try:
    response = requests.get("http://cocoflops1:8000/v1/models")
    print("vLLM server status:", response.status_code)
    print("Available models:", response.json())
except Exception as e:
    print("Error connecting to vLLM server:", e)
    sys.exit(1)

# Initialize the experimenter
experimenter = LMExperimenter(
    model_name="qwen2.5-3b-instruct",  # This is just a reference name
    temperature=0.7,
    max_tokens=256
)

# Set a system message
experimenter.set_system_message("You are a helpful assistant.")

# Test a simple prompt
print("\nTesting LMExperimenter with a simple prompt...")
response = experimenter.prompt_llm("Write a short haiku about artificial intelligence.")
print("Response:", response)

print("\nTest completed successfully!")
