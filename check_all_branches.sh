#!/bin/bash

# Script to check for files with spaces on all branches

echo "Checking all branches for files with spaces..."

# Get the current branch to return to it later
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

# Save all branch names to an array (excluding remotes)
BRANCHES=($(git branch | sed 's/^[ \t*]*//' | tr '\n' ' '))
echo "Branches to check: ${BRANCHES[@]}"

# For each branch
for branch in "${BRANCHES[@]}"; do
    echo "Checking branch: $branch"
    git checkout "$branch"
    
    # Find files with spaces
    FILES_WITH_SPACES=$(find . -name "* *" -type f | grep -v ".venv")
    if [ -z "$FILES_WITH_SPACES" ]; then
        echo "✅ No files with spaces found in branch $branch"
    else
        echo "⚠️ Files with spaces found in branch $branch:"
        echo "$FILES_WITH_SPACES"
        echo "-------------------"
    fi
done

# Return to the original branch
echo "Returning to branch: $CURRENT_BRANCH"
git checkout "$CURRENT_BRANCH"

echo "Check completed!" 