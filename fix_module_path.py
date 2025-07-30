import sys
import os

# Get absolute path to ensure reliability
base_path = os.path.abspath('/sailhome/agam/boxing-gym')

# Check if boxing_gym exists directly
boxing_gym_path = os.path.join(base_path, 'boxing_gym')
src_boxing_gym_path = os.path.join(base_path, 'src', 'boxing_gym')

if os.path.exists(boxing_gym_path):
    print(f"Found boxing_gym at {boxing_gym_path}")
    if base_path not in sys.path:
        sys.path.insert(0, base_path)
elif os.path.exists(src_boxing_gym_path):
    print(f"Found boxing_gym at {src_boxing_gym_path}")
    src_path = os.path.join(base_path, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
else:
    print("Could not find boxing_gym module!")
    print(f"Paths checked: {boxing_gym_path}, {src_boxing_gym_path}")
    # Create a directory listing to help debug
    print("Directory contents:")
    for item in os.listdir(base_path):
        print(f"  {item}")
    if os.path.exists(os.path.join(base_path, 'src')):
        print("src directory contents:")
        for item in os.listdir(os.path.join(base_path, 'src')):
            print(f"  {item}")

# Try importing the module to verify setup
try:
    from boxing_gym.agents.agent import LMExperimenter
    print("Successfully imported LMExperimenter!")
except ImportError as e:
    print(f"Import error: {e}")
    print("Python path:")
    for p in sys.path:
        print(f"  {p}")
