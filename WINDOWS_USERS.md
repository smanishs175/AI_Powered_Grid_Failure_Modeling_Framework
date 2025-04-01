# Windows Compatibility Guide

## Known Issues

Windows users may encounter issues when cloning this repository due to file paths containing spaces or special characters. This is because Git on Windows has stricter requirements for file paths.

## Solutions

If you are encountering cloning errors on Windows (e.g., `error: invalid path`), try the following:

### Method 1: Use the latest repository version (recommended)

The repository has been cleaned to remove problematic file paths. Make sure you're cloning the latest version:

```
git clone https://github.com/smanishs175/AI_Powered_Grid_Failure_Modeling_Framework.git
```

### Method 2: Clone with depth 1

If you still have issues, try a shallow clone:

```
git clone --depth 1 https://github.com/smanishs175/AI_Powered_Grid_Failure_Modeling_Framework.git
```

### Method 3: Adjust Git configuration

Some Windows Git issues can be resolved with these configurations:

```
git config --global core.protectNTFS false
git config --global core.longpaths true
```

Then try cloning again.

### Method 4: Download as ZIP

As a last resort, you can download the repository as a ZIP file from GitHub:
1. Go to the repository page
2. Click the green "Code" button
3. Select "Download ZIP"

## For Repository Maintainers

If problematic paths continue to cause issues:

1. Run `find_problem_paths.sh` to identify all problematic paths
2. Update `clean_history.sh` with these paths
3. Run the clean script to permanently remove these paths from Git history
4. Force push the cleaned history to the repository

**Important:** After cleaning the repository history, all collaborators will need to clone the repository again. 