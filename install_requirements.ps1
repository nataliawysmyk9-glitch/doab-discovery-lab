Write-Host "Installing Discovery Lab dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Write-Host ""
Write-Host "Installation complete."
Write-Host "Next run: python src/prepare_data.py"
