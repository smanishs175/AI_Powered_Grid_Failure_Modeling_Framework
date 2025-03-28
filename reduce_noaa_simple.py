#!/usr/bin/env python3
"""
Script to reduce NOAA Daily Summaries dataset from ~134GB to ~1GB
using a simple approach of selecting the most relevant stations.
"""

import os
import shutil
import random
import time
from datetime import datetime

# Configuration
INPUT_DIR = "data_collection_by_manish/NOAA_Daily_Summaries/daily-summaries-latest"
OUTPUT_DIR = "data_collection_by_manish/NOAA_Daily_Summaries_Reduced"
TARGET_SIZE_GB = 1.0  # Target size in GB

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set to focus on US stations only (these prefixes identify US weather stations)
US_PREFIXES = ['USW', 'USC', 'US1']

# Get list of all CSV files
print("Scanning for station files...")
all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
print(f"Found {len(all_files)} total station files")

# Filter to US stations (these are more likely to be relevant)
us_files = [f for f in all_files if any(f.startswith(prefix) for prefix in US_PREFIXES)]
print(f"Filtered to {len(us_files)} US station files")

# Calculate the approximate number of files to copy to reach target size
# We need about 0.75% of the original dataset (1GB/134GB)
sample_percentage = 0.75
num_files_to_copy = int(len(us_files) * sample_percentage / 100)

# Ensure we copy at least 1 file
num_files_to_copy = max(1, num_files_to_copy)
print(f"Will copy approximately {num_files_to_copy} files (about {sample_percentage}% of US stations)")

# Sample random stations
sampled_files = random.sample(us_files, num_files_to_copy)

# Copy the selected files
print(f"Copying {len(sampled_files)} files to the reduced dataset...")
bytes_copied = 0
files_copied = 0

for filename in sampled_files:
    source_path = os.path.join(INPUT_DIR, filename)
    dest_path = os.path.join(OUTPUT_DIR, filename)
    
    try:
        shutil.copy2(source_path, dest_path)
        file_size = os.path.getsize(source_path)
        bytes_copied += file_size
        files_copied += 1
        
        # Show progress
        if files_copied % 10 == 0:
            print(f"Copied {files_copied}/{len(sampled_files)} files, "
                  f"total size: {bytes_copied / (1024*1024*1024):.2f} GB")
        
        # Check if we've reached our target size (with some margin)
        if bytes_copied >= TARGET_SIZE_GB * 0.99 * 1024*1024*1024:
            print(f"Reached target size after copying {files_copied} files")
            break
    
    except Exception as e:
        print(f"Error copying {filename}: {e}")

# Create metadata file
with open(os.path.join(OUTPUT_DIR, "DATASET_METADATA.txt"), 'w') as meta_file:
    meta_file.write(f"Original dataset size: ~134GB\n")
    meta_file.write(f"Reduction performed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    meta_file.write(f"Reduction method: Random sampling of US weather stations\n")
    meta_file.write(f"Files copied: {files_copied} out of {len(all_files)} total files\n")
    meta_file.write(f"Final size: {bytes_copied / (1024*1024*1024):.2f} GB\n")
    meta_file.write(f"Sample percentage: {files_copied / len(all_files) * 100:.4f}% of total stations\n")
    meta_file.write(f"Note: This is a randomly sampled subset intended for development and testing only.\n")
    meta_file.write(f"For production use, consider a more targeted geographical selection based on your specific grid analysis needs.\n")

# Output final statistics
final_size_gb = bytes_copied / (1024*1024*1024)
print("\nReduction complete!")
print(f"Original dataset: 134 GB")
print(f"Reduced dataset: {final_size_gb:.2f} GB")
print(f"Reduction ratio: {134 / final_size_gb:.1f}x")
print(f"Files copied: {files_copied}")
print(f"Reduced dataset location: {OUTPUT_DIR}")
