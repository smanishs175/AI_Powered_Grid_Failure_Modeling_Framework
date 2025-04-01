#!/bin/bash
# This script permanently removes problematic paths from Git history

echo "Creating backup branch..."
git checkout -b backup_before_cleanup

echo "Removing problematic paths from Git history..."
git filter-branch --force --index-filter \
  "git rm -r --cached --ignore-unmatch \"data_collection_by_manish/Weather Data/.DS_Store\" \"data_collection_by_manish/Weather Data /.DS_Store\" \"data_collection_by_manish/IEEE Power System Test Cases/Dynamic Test Cases/30 Bus New England Dynamic Test/30 bus dynamic test case .txt\"" \
  --prune-empty --tag-name-filter cat -- --all

echo "Cleaning up references..."
git for-each-ref --format="delete %(refname)" refs/original/ | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now

echo "The repository history has been cleaned."
echo "You will need to force push this to the remote repository with:"
echo "git push origin --force --all"
echo "git push origin --force --tags"
echo ""
echo "IMPORTANT: All collaborators will need to re-clone the repository after this change!" 