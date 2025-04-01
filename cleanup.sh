#!/bin/bash

# Script to clean up deleted files and commit our utility scripts across all branches

echo "Starting cleanup of branches..."

# Get the current branch to return to it later
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

# Save all branch names to an array (excluding remotes)
BRANCHES=($(git branch | sed 's/^[ \t*]*//' | tr '\n' ' '))
echo "Branches to clean: ${BRANCHES[@]}"

# For each branch
for branch in "${BRANCHES[@]}"; do
    echo "Cleaning branch: $branch"
    git checkout "$branch"
    
    # Remove deleted files from git's index (but not from disk)
    git rm --cached "data_collection_by_manish/IEEE Power System Test Cases/Dynamic Test Cases/30 Bus New England Dynamic Test/30 bus dynamic test case .txt" 2>/dev/null || true
    git rm --cached "data_collection_by_manish/IEEE Power System Test Cases/Dynamic Test Cases/50 Generator Dynamic Test Case/50 Bus Dyn Test A.pdf" 2>/dev/null || true
    git rm --cached "data_collection_by_manish/IEEE Power System Test Cases/Dynamic Test Cases/50 Generator Dynamic Test Case/50 Generator System.pdf" 2>/dev/null || true
    git rm --cached "data_collection_by_manish/Outage Data/DOE-147 Data/2020/Electric Power Monthly with Data For January 2020.pdf" 2>/dev/null || true
    git rm --cached "data_collection_by_manish/Outage Data/DOE-147 Data/2021/DOE 147 Outage data 2021.pdf" 2>/dev/null || true
    git rm --cached "data_collection_by_manish/Outage Data/DOE-147 Data/2022/DOE 147 Outage Data February 2022.pdf" 2>/dev/null || true
    git rm --cached "data_collection_by_manish/RTS_Data/RTS_Data/FormattedData/PSO/RTS-GMLC - BASE - PSOv28.zip" 2>/dev/null || true
    
    # Add our scripts to this branch
    git add check_all_branches.sh check_branch_status.sh fix_filenames.sh update_all_branches.sh fix_and_commit_all.sh cleanup.sh 2>/dev/null || true
    
    # Commit the changes if there are any
    if git diff --cached --quiet; then
        echo "No changes to commit on branch $branch"
    else
        git commit -m "Clean up deleted files and add utility scripts"
        echo "Changes committed on branch $branch"
    fi
    
    echo "Finished cleaning branch: $branch"
    echo "-------------------"
done

# Return to the original branch
echo "Returning to branch: $CURRENT_BRANCH"
git checkout "$CURRENT_BRANCH"

echo "All branches have been cleaned!"
echo "Don't forget to push all branches:"
echo "git push --all origin" 