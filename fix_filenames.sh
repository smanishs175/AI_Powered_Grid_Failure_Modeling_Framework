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
echo "Remember to commit the changes with: git commit -m 'Renamed files to remove spaces for Windows compatibility'" 