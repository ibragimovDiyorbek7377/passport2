# RAILWAY DEPLOYMENT FIX - CRITICAL
# Date: 2026-01-15
# Issue: Railway deploying from wrong branch with old dependencies
# Status: READY TO DEPLOY

## PROBLEM IDENTIFIED
Railway is deploying from: `claude/passport-scanner-tma-D4pVo`
That branch has: Mistral AI code ✅ + OLD dependencies ❌

## ROOT CAUSE
Branch `claude/passport-scanner-tma-D4pVo` has:
- python-telegram-bot==20.7  ❌ (conflicts with Mistral AI)
- python-dateutil==2.8.2     ❌ (conflicts with Mistral AI)
- mistralai>=1.0.0           ✅ (correct)

Result: DEPENDENCY CONFLICT → BUILD FAILS

## SOLUTION
Deploy from: `claude/fix-gemini-model-error-ONHAo`
This branch has ALL fixes:
- python-telegram-bot>=22.0  ✅ (compatible)
- python-dateutil>=2.9.0     ✅ (compatible)
- mistralai>=1.0.0           ✅ (correct)
- Mistral AI Vision code     ✅ (complete)

## HOW TO FIX (Choose ONE method)

### METHOD 1: Update Railway Settings (RECOMMENDED - 2 minutes)
1. Go to Railway Dashboard
2. Click on your project "passporteye"
3. Go to Settings → Deploy
4. Find "Source" or "Branch" setting
5. Change from `claude/passport-scanner-tma-D4pVo`
         to `claude/fix-gemini-model-error-ONHAo`
6. Click "Save" or "Redeploy"
7. Railway will auto-rebuild with correct dependencies

### METHOD 2: Create Pull Request on GitHub (if using GitHub)
1. Go to GitHub repository
2. Create PR: `claude/fix-gemini-model-error-ONHAo` → `claude/passport-scanner-tma-D4pVo`
3. Title: "Fix dependency conflicts for Mistral AI deployment"
4. Merge the PR
5. Railway will auto-detect and redeploy

### METHOD 3: Manual Trigger (if Railway has webhook)
1. In Railway Dashboard
2. Click "Deploy" or "Redeploy"
3. Select branch: `claude/fix-gemini-model-error-ONHAo`
4. Click "Deploy Now"

## VERIFICATION
After deployment succeeds, check Railway logs for:

✅ BUILD SUCCESS:
```
Collecting python-telegram-bot>=22.0
Downloading python_telegram_bot-22.5...
Collecting mistralai>=1.0.0
Downloading mistralai-1.10.0...
Successfully installed python-telegram-bot-22.5 mistralai-1.10.0
```

✅ RUNTIME SUCCESS:
```
🤖 Mistral MRZ Scanner initialized with model: pixtral-12b-2409
✅ FastAPI server started in background thread
✅ Bojxona Passport Scanner is now running!
```

## IF STILL FAILING
Check Railway environment variables:
- MISTRAL_API_KEY=JWSVnIJhbnyhc80PY32AhKkxEbS4SFFi (should be set)
- PORT=8000 (auto-set by Railway)

## FILES UPDATED IN WORKING BRANCH
✅ main.py - Complete Mistral AI implementation
✅ requirements.txt - Fixed dependencies (telegram-bot>=22.0, dateutil>=2.9.0)
✅ templates/index.html - Updated branding to Mistral AI
✅ Dockerfile - Added cache-busting and updated comments
✅ .railway-deploy - Deployment marker file

## COMMIT HISTORY (Latest First)
5f5d828 - ci: Force Railway deployment with latest Mistral AI code
03733c7 - fix: Resolve dependency conflicts for Railway deployment
f1c941a - refactor: Replace Google Gemini with Mistral AI Vision API

## BRANCH INFO
Branch: claude/fix-gemini-model-error-ONHAo
Status: ✅ PUSHED TO REMOTE
Ready: ✅ YES
Protected: ❌ NO (can deploy from this)

## EXPECTED RESULT
✅ Build succeeds with no dependency conflicts
✅ Mistral AI Vision API initializes correctly
✅ Telegram bot responds to /start command
✅ Mini App opens and camera works
✅ Passport scanning works with Mistral AI (no more 404 errors)

---
Generated: 2026-01-15 11:05 UTC
Last verified: Branch pushed and ready
Action required: Update Railway to deploy from correct branch
