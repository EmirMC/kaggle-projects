#!/bin/bash

# Initialize git repository
git init

# Add all files to the repository
git add .

# Commit the files
git commit -m "Initial commit"

# Set the remote origin
git remote add origin git@github.com:EmirMC/kaggle-projects.git

# Push the files to the repository
git push -u origin main
