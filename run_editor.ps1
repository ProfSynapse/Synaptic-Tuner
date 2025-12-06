$ErrorActionPreference = "Continue"

Write-Host "Checking for Streamlit..."
python -c "import streamlit; import pandas; import st_aggrid; import dotenv; import yaml" 2>$null

if ($LASTEXITCODE -ne 0) {
    Write-Host "Streamlit, Pandas, AgGrid, DotEnv, or PyYAML not found. Installing..."
    pip install streamlit pandas streamlit-aggrid python-dotenv pyyaml
}

Write-Host "Starting Dataset Editor..."
python -m streamlit run dataset_editor.py