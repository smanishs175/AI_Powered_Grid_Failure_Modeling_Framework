#!/bin/bash

# Script to apply filename fixes to all branches

echo "Applying filename fixes to all branches..."

# Get the current branch to return to it later
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

# Save all branch names to an array (excluding remotes)
BRANCHES=($(git branch | grep -v "^\*" | sed 's/^[ \t]*//' | tr '\n' ' '))
echo "Branches to update: ${BRANCHES[@]}"

# For each branch
for branch in "${BRANCHES[@]}"; do
    echo "Switching to branch: $branch"
    git checkout "$branch"
    
    # Run the filename fixing script
    if [ -f ./fix_filenames.sh ]; then
        echo "Running fix_filenames.sh on branch $branch"
        ./fix_filenames.sh
        
        # Commit changes
        git commit -m "Renamed files to remove spaces for Windows compatibility"
    else
        echo "fix_filenames.sh not found on branch $branch, creating it..."
        
        # Create the script on this branch
        cat > fix_filenames.sh << 'EOF'
#!/bin/bash

# Script to rename files with spaces in their names and update git

echo "Fixing filenames with spaces..."

# Find all files with spaces in their names
find . -name "* *" -type f | grep -v ".venv" | while read file; do
    # Create a new filename by replacing spaces with underscores
    newname=$(echo "$file" | sed 's/ /_/g')
    
    # Create the directory if it doesn't exist
    mkdir -p "$(dirname "$newname")"
    
    echo "Renaming: $file -> $newname"
    
    # Move the file
    mv "$file" "$newname"
    
    # Add the renamed file to git (if this is a git repository)
    if [ -d .git ] || git rev-parse --git-dir > /dev/null 2>&1; then
        git add "$newname"
        echo "Added to git: $newname"
    fi
done

echo "Filename fixing completed!"
EOF
        
        chmod +x fix_filenames.sh
        git add fix_filenames.sh
        git commit -m "Add filename fixing script"
        
        # Run the script
        ./fix_filenames.sh
        
        # Commit changes
        git commit -m "Renamed files to remove spaces for Windows compatibility"
    fi
    
    echo "Finished updating branch: $branch"
    echo "-----------------------------"
done

# Return to the original branch
echo "Returning to branch: $CURRENT_BRANCH"
git checkout "$CURRENT_BRANCH"

echo "All branches have been updated!"
echo "Don't forget to push all branches:"
echo "git push --all origin" 