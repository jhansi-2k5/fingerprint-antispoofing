# Intelligent Fingerprint Anti-Spoofing System (Streamlit Mini Project)

## Important limitation
This is an educational prototype. The bundled CSV is synthetically generated, and the model is trained on synthetic tabular feature indicators—not on real fingerprint images. The image upload page computes basic image statistics and passes them to that demo model. Its prediction must not be treated as evidence that a fingerprint is genuine or spoofed, and the app must not be used for real access-control decisions.

## Features
- Streamlit dashboard and navigation
- Random Forest classifier trained on bundled synthetic sample features
- Manual feature-input classification
- Upload PNG/JPG/JPEG and calculate basic image-statistic proxies
- Dataset preview and CSV download
- Class distribution, feature importance, held-out metrics and confusion matrix
- Project documentation page

## Requirements
- Python 3.10 or later recommended
- Internet access for the initial dependency installation

## Run on Windows
1. Extract the ZIP file.
2. Open the extracted `fingerprint_antispoof_streamlit` folder in VS Code.
3. Open Terminal > New Terminal.
4. Create a virtual environment:
   ```powershell
   py -m venv .venv
   ```
5. Activate it:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, use Command Prompt and run:
   ```bat
   .venv\Scripts\activate.bat
   ```
6. Install packages:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
7. Start Streamlit:
   ```powershell
   streamlit run app.py
   ```
8. Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Folder structure
```
fingerprint_antispoof_streamlit/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── sample_fingerprint_features.csv
└── docs/
    ├── project_report.md
    └── viva_questions.md
```

## Demo steps
1. Open Overview and explain the workflow.
2. Choose Analyze sample features.
3. Test the two example presets and click Classify sample.
4. Open Dataset & model and explain the synthetic data, feature importance and confusion matrix.
5. Optionally upload a sample fingerprint image on Analyze an image, explaining that its output is illustrative only.
6. Open Project documentation to review objectives and limitations.

## Troubleshooting
- `streamlit is not recognized`: activate the virtual environment and run `python -m streamlit run app.py`.
- Missing package: run `pip install -r requirements.txt` again.
- File not found: make sure `app.py`, `requirements.txt`, and the `data` folder remain in the same project folder.
- Port already in use: run `streamlit run app.py --server.port 8502`.

## Replacing synthetic data
Replace `data/sample_fingerprint_features.csv` with a lawful, labeled dataset containing the six feature columns and `label`, where label values are `Genuine` and `Spoof`. Keep in mind that feature definitions and scales must be consistent with the data. For a scientifically valid image model, use image-level data and a validated feature-extraction pipeline; do not simply relabel synthetic data.
