# deploy_to_hf.ps1
# Usage: .\deploy_to_hf.ps1
# Deploys current 'deployment' branch to HF Space via orphan commit
# (avoids binary-file-in-history rejection from HF)

$ErrorActionPreference = "Stop"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$branch = "hf-tmp-$timestamp"

Write-Host "=== Deploying to Hugging Face Space ===" -ForegroundColor Cyan
Write-Host "Creating orphan branch: $branch"

# Ensure we start from deployment branch
git checkout deployment

# Create orphan branch with current state
git checkout --orphan $branch

# Stage only deployment-relevant files (no binary history)
git add web_app/ hybrid_music_engine/ audio_model/ models/ requirements.txt Dockerfile README.md .gitattributes spotify_client.py

# Commit
git commit -m "deploy: $timestamp"

# Force push to HF Space main
Write-Host "Pushing to HF Space..." -ForegroundColor Yellow
git push space "${branch}:main" --force

# Clean up
git checkout deployment
git branch -D $branch

Write-Host "=== Deploy complete! ===" -ForegroundColor Green
Write-Host "HF Space will rebuild in ~5-10 minutes."
