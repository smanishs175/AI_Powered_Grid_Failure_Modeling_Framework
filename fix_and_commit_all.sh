#!/bin/bash

# Script to properly handle file renaming and commit changes on all branches

echo "Starting filename fix and commit process for all branches..."

# Get the current branch to return to it later
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

# Save all branch names to an array (excluding remotes)
BRANCHES=($(git branch | grep -v "^\*" | sed 's/^[ \t]*//' | tr '\n' ' '))
echo "Branches to update: ${BRANCHES[@]}"

# Add current branch to beginning of list to ensure it's processed first
BRANCHES=("$CURRENT_BRANCH" "${BRANCHES[@]}")

# For each branch
for branch in "${BRANCHES[@]}"; do
    echo "Switching to branch: $branch"
    git checkout "$branch"
    
    echo "Finding files with spaces in $branch..."
    # First, let's identify files with spaces
    files_with_spaces=$(find . -name "* *" -type f | grep -v ".venv")
    
    if [ -z "$files_with_spaces" ]; then
        echo "No files with spaces found in $branch"
    else
        echo "Found $(echo "$files_with_spaces" | wc -l) files with spaces"
        
        # Process each file
        echo "$files_with_spaces" | while read old_file; do
            # Create new filename (replace spaces with underscores)
            new_file=$(echo "$old_file" | sed 's/ /_/g')
            
            echo "Renaming: $old_file -> $new_file"
            
            # Ensure the directory exists
            mkdir -p "$(dirname "$new_file")"
            
            # Move the file
            mv "$old_file" "$new_file"
            
            # Stage both the removal and addition
            git rm --cached "$old_file"
            git add "$new_file"
        done
        
        # Commit the changes
        git commit -m "Renamed files to remove spaces for Windows compatibility"
        echo "Changes committed on branch $branch"
    fi
    
    echo "Finished processing branch: $branch"
    echo "-------------------------------"
done

# Return to the original branch
echo "Returning to branch: $CURRENT_BRANCH"
git checkout "$CURRENT_BRANCH"

echo "All branches have been processed!"
echo "Don't forget to push all branches:"
echo "git push --all origin" 