#!/bin/bash
# This script finds all paths in the Git history that contain spaces or other problematic characters

echo "Finding all paths with spaces in Git history (this may take a while)..."
git log --all --pretty=format: --name-only | sort -u | grep -E "[ ]" > problematic_paths.txt

echo "Done! Found $(wc -l < problematic_paths.txt) problematic paths."
echo "The paths have been saved to problematic_paths.txt"
echo ""
echo "To modify clean_history.sh to include all these paths, run:"
echo "cat problematic_paths.txt | sed 's/^/\"/' | sed 's/$/\"/' | tr '\n' ' ' > paths.txt"
echo ""
echo "Then replace the list of paths in clean_history.sh with the contents of paths.txt" 