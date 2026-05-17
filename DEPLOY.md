# Deploying Stockfolio API to Render (Free, Always-On)

## Steps

### 1. Push to GitHub
```bash
cd "C:\Last Attempt\stockfolio-api"
git init
git add .
git commit -m "Stockfolio API initial commit"
# Create a new repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/stockfolio-api.git
git push -u origin main
```

### 2. Deploy on Render
1. Go to https://render.com and sign up (free)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account and select the `stockfolio-api` repo
4. Fill in:
   - **Name:** `stockfolio-api`
   - **Runtime:** Python
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --workers 1 --timeout 120 --bind 0.0.0.0:$PORT`
   - **Instance Type:** Free
5. Click **"Create Web Service"**

### 3. Get your URL
After deploy (2–5 min), Render gives you a URL like:
`https://stockfolio-api.onrender.com`

### 4. Update Flutter app
Open `C:\Last Attempt\Stockfolio\lib\constants\constants.dart` and replace:
```dart
static const String baseUrl = 'https://stockfolio-api.onrender.com';
```
with your actual Render URL.

## Notes
- **Free tier sleeps after 15 min inactivity** — the first request after sleep takes ~30s.
  To keep it warm, use UptimeRobot (free) to ping `/api/health` every 10 minutes.
- Memory: 512MB is enough for VADER sentiment. If you want FinBERT (much better quality),
  upgrade to Render's $7/month Starter plan and uncomment the torch/transformers lines
  in `requirements.txt`, then add `USE_FINBERT=true` to Render env vars.

## Test API locally first
```bash
pip install -r requirements.txt
python app.py
# Visit: http://localhost:5000/api/health
# Visit: http://localhost:5000/api/companies
```
