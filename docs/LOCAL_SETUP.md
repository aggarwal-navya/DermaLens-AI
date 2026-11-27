# Run DermaLens AI on your own computer

You need **no datasets** to run the app — only the trained model bundle (~630 MB) from Google Drive.
Tested with Python 3.11, CPU only. One prediction takes about 2-4 seconds on a laptop CPU.

## 1. Get the code
```bash
git clone https://github.com/aggarwal-navya/DermaLens-AI.git
cd DermaLens-AI
```

## 2. Create a virtual environment and install
Windows (CMD):
```powershell
py -m venv .venv
.venv\Scripts\Activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r app/requirements.txt
```
macOS / Linux:
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r app/requirements.txt
```
(The CPU build of PyTorch is ~200 MB instead of ~2.5 GB for the CUDA build.)

## 3. Download the model bundle
From Google Drive, open `DermaLens_AI/models/deploy/` and download **all four files**
(`model_card.json`, `member_0.pt`, `member_1.pt`, `member_2.pt`) into `models/deploy/` in this folder.

**Important (v3.1 fix):** if `model_card.json` contains `"confidence_threshold": 0.0`, the bundle was exported
before the threshold bug was fixed. Either re-run notebooks 10 and 14 on Colab (about 5 minutes), or open
`model_card.json` in a text editor and change it to `"confidence_threshold": 0.2` (the value notebook 10
now computes from your validation set).

## 4. Start the website
```bash
uvicorn app.api:app --port 8000
```
Open http://localhost:8000 — Home, Analyse and Model & Research pages.
Check http://localhost:8000/health shows `"model_loaded": true`.

## 5. (Optional) run the tests
```bash
pip install pytest httpx imagehash
python -m pytest -q
```
The tests use small generated images and a tiny model, so they need no data and no bundle.

## Troubleshooting
| Problem | Fix |
|---|---|
| `Model bundle not found` | the four files are not in `models/deploy/` |
| `No module named torch` | step 2, first `pip install` line |
| Port 8000 busy | `uvicorn app.api:app --port 8080` |
| Very slow first prediction | normal: models load and warm up on the first request |
