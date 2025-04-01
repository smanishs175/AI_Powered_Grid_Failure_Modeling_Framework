# Utility Scripts for Repository Maintenance

This repository includes several utility scripts to help with maintenance tasks, particularly related to handling files with spaces in their names, which can cause issues on some operating systems, especially Windows.

## Available Scripts

### `fix_filenames.sh`
- **Purpose**: Renames files by replacing spaces with underscores.
- **Usage**: `./fix_filenames.sh`
- **What it does**: Finds all files with spaces in their names and renames them with underscores instead.

### `fix_and_commit_all.sh`
- **Purpose**: Renames files with spaces and commits the changes on the current branch.
- **Usage**: `./fix_and_commit_all.sh`
- **What it does**: Renames files with spaces to use underscores, adds these changes to Git, and commits them.

### `update_all_branches.sh`
- **Purpose**: Updates all branches to reflect the renamed files.
- **Usage**: `./update_all_branches.sh`
- **What it does**: Applies the filename changes to all branches in the repository.

### `check_branch_status.sh`
- **Purpose**: Checks the status of all branches.
- **Usage**: `./check_branch_status.sh`
- **What it does**: Goes through each branch and shows its status relative to the remote repository.

### `check_all_branches.sh`
- **Purpose**: Checks all branches for files with spaces.
- **Usage**: `./check_all_branches.sh`
- **What it does**: Examines each branch to find any remaining files with spaces in their names.

### `cleanup.sh`
- **Purpose**: Cleans up deleted files and adds utility scripts to all branches.
- **Usage**: `./cleanup.sh`
- **What it does**: Removes references to deleted files from Git's index and adds utility scripts to each branch.

### `push_all.sh`
- **Purpose**: Pushes all branches to the remote repository.
- **Usage**: `./push_all.sh`
- **What it does**: Pushes all local branches to the remote repository.

## How to Use

1. Make these scripts executable with `chmod +x script_name.sh`
2. Run the script you need using `./script_name.sh`

## Important Note

Some of these scripts make significant changes to your repository. Always ensure you have a backup or that your changes are committed before running them.