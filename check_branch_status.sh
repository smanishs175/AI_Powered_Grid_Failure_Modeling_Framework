#!/bin/bash

# Script to check the status of all branches versus their remote
echo "Checking branch status..."

# Get the current branch to return to it later
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

# Save all branch names to an array (excluding remotes)
BRANCHES=($(git branch | sed 's/^[ \t*]*//' | tr '\n' ' '))
echo "Branches to check: ${BRANCHES[@]}"

# For each branch
for branch in "${BRANCHES[@]}"; do
    echo "Checking branch: $branch"
    git checkout "$branch" > /dev/null 2>&1
    
    # Get the status versus remote
    git status -sb
    
    echo "-------------------"
done

# Return to the original branch
echo "Returning to branch: $CURRENT_BRANCH"
git checkout "$CURRENT_BRANCH" > /dev/null 2>&1

echo "Status check completed!" 