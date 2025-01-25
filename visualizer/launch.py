"""
MIT License
ROCm Profile Data Tools Launcher

"""

import streamlit as st
import subprocess
import sys
import os

st.set_page_config(page_title="ROCm Profile Data Tools")
st.title("ROCm Profile Data Tools")

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

tool = st.selectbox(
    "Select Tool",
    ["RPD Visualizer", "Schema Inspector"]
)

if st.button("Launch"):
    if tool == "RPD Visualizer":
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", 
                         os.path.join(current_dir, "rpd_visualizer.py")])
    else:
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", 
                         os.path.join(current_dir, "rpd_schema_check.py")])

st.markdown("""
## Tool Description

- **RPD Visualizer**: Visualize and analyze kernel executions, memory operations, and API calls
- **Schema Inspector**: Explore database structure, run custom queries, and inspect data

Choose a tool and click Launch to start.
""")