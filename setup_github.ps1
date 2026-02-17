# PowerShell script to setup and push to GitHub
# Usage: .\setup_github.ps1

Write-Host "=== GitHub Setup Script ===" -ForegroundColor Cyan
Write-Host ""

# Check if git is installed
try {
    $gitVersion = git --version
    Write-Host "Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Git is not installed!" -ForegroundColor Red
    Write-Host "Please install Git from: https://git-scm.com/downloads" -ForegroundColor Yellow
    exit 1
}

# Check if already a git repo
if (Test-Path ".git") {
    Write-Host "Git repository already initialized." -ForegroundColor Yellow
} else {
    Write-Host "Initializing git repository..." -ForegroundColor Cyan
    git init
    Write-Host "Git repository initialized." -ForegroundColor Green
}

# Check if .gitignore exists
if (Test-Path ".gitignore") {
    Write-Host ".gitignore found." -ForegroundColor Green
} else {
    Write-Host "WARNING: .gitignore not found!" -ForegroundColor Yellow
}

# Show current status
Write-Host ""
Write-Host "Current git status:" -ForegroundColor Cyan
git status --short

# Ask for remote URL
Write-Host ""
Write-Host "Please provide your GitHub repository URL:" -ForegroundColor Yellow
Write-Host "Example: https://github.com/username/repo-name.git" -ForegroundColor Gray
Write-Host "Or: git@github.com:username/repo-name.git" -ForegroundColor Gray
$remoteUrl = Read-Host "GitHub repository URL"

if ([string]::IsNullOrWhiteSpace($remoteUrl)) {
    Write-Host "No URL provided. Skipping remote setup." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To add remote later, run:" -ForegroundColor Cyan
    Write-Host "  git remote add origin YOUR_URL" -ForegroundColor Gray
    exit 0
}

# Check if remote already exists
$existingRemote = git remote get-url origin 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Remote 'origin' already exists: $existingRemote" -ForegroundColor Yellow
    $overwrite = Read-Host "Overwrite? (y/N)"
    if ($overwrite -eq "y" -or $overwrite -eq "Y") {
        git remote set-url origin $remoteUrl
        Write-Host "Remote URL updated." -ForegroundColor Green
    } else {
        Write-Host "Keeping existing remote." -ForegroundColor Yellow
    }
} else {
    git remote add origin $remoteUrl
    Write-Host "Remote 'origin' added." -ForegroundColor Green
}

# Stage all files
Write-Host ""
Write-Host "Staging all files..." -ForegroundColor Cyan
git add .

# Show what will be committed
Write-Host ""
Write-Host "Files to be committed:" -ForegroundColor Cyan
git status --short

# Ask for commit message
Write-Host ""
$defaultMessage = "Initial commit: CT slice interpolation via 3D Gaussian Splatting"
Write-Host "Commit message (press Enter for default):" -ForegroundColor Yellow
Write-Host "Default: $defaultMessage" -ForegroundColor Gray
$commitMessage = Read-Host "Message"

if ([string]::IsNullOrWhiteSpace($commitMessage)) {
    $commitMessage = $defaultMessage
}

# Commit
Write-Host ""
Write-Host "Committing changes..." -ForegroundColor Cyan
git commit -m $commitMessage

if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Commit failed or nothing to commit." -ForegroundColor Yellow
}

# Set default branch to main
Write-Host ""
Write-Host "Setting default branch to 'main'..." -ForegroundColor Cyan
git branch -M main 2>$null

# Ask if user wants to push
Write-Host ""
$pushNow = Read-Host "Push to GitHub now? (y/N)"
if ($pushNow -eq "y" -or $pushNow -eq "Y") {
    Write-Host ""
    Write-Host "Pushing to GitHub..." -ForegroundColor Cyan
    Write-Host "Note: You may need to authenticate." -ForegroundColor Yellow
    git push -u origin main
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "Successfully pushed to GitHub!" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "Push failed. Common reasons:" -ForegroundColor Red
        Write-Host "  1. Authentication required (use Personal Access Token)" -ForegroundColor Yellow
        Write-Host "  2. Repository doesn't exist yet" -ForegroundColor Yellow
        Write-Host "  3. Network issues" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "To push manually later, run:" -ForegroundColor Cyan
        Write-Host "  git push -u origin main" -ForegroundColor Gray
    }
} else {
    Write-Host ""
    Write-Host "Skipping push. To push later, run:" -ForegroundColor Cyan
    Write-Host "  git push -u origin main" -ForegroundColor Gray
}

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
