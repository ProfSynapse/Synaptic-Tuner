$ErrorActionPreference = "Continue"

Write-Host "Checking for Streamlit..."
python -c "import streamlit; import pandas; import st_aggrid" 2>$null

if ($LASTEXITCODE -ne 0) {
    Write-Host "Streamlit, Pandas, or AgGrid not found. Installing..."
    pip install streamlit pandas streamlit-aggrid
}

Write-Host "Starting Dataset Editor..."
python -m streamlit run dataset_editor.py