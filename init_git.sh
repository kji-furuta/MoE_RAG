#!/bin/bash

# Initialize Git repository for AI Fine-Tuning Project

echo "Initializing Git repository..."

# Initialize git
git init

# Set up Git user (update these with your information)
git config user.name "kji-furuta"
git config user.email "your-email@example.com"

# Create initial commit
git add .
git commit -m "Initial commit: AI Fine-Tuning Project structure"

# Add remote origin (update with your repository URL)
echo ""
echo "To connect to GitHub, run:"
echo "git remote add origin https://github.com/kji-furuta/AI_FT_7.git"
echo "git branch -M main"
echo "git push -u origin main"

echo ""
echo "Git repository initialized successfully!"
