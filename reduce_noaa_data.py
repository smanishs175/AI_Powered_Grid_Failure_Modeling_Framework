#!/usr/bin/env python3
"""
Script to reduce NOAA Daily Summaries dataset from ~134GB to ~1GB
by applying multiple filtering techniques:
1. Time period filtering (recent years only)
2. Geographic filtering (US stations only)
3. Column filtering (select critical variables only)
4. Station sampling (select representative stations)
"""

import os
import shutil
import csv
import random
from datetime import datetime
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Reduce NOAA Daily Summaries dataset')
    parser.add_argument('--input_dir', type=str, 
                        default='data_collection_by_manish/NOAA_Daily_Summaries/daily-summaries-latest',
                        help='Input directory containing NOAA daily summary CSV files')
    parser.add_argument('--output_dir', type=str, 
                        default='data_collection_by_manish/NOAA_Daily_Summaries_Reduced',
                        help='Output directory for reduced dataset')
    parser.add_argument('--start_year', type=int, default=2018,
                        help='Start year for filtering (inclusive)')
    parser.add_argument('--station_sample_rate', type=float, default=0.01,
                        help='Fraction of stations to keep (0-1)')
    parser.add_argument('--us_only', action='store_true', default=True,
                        help='Filter to include only US stations')
    return parser.parse_args()

def is_us_station(station_id):
    """Check if station ID is from the United States based on its prefix"""
    us_prefixes = ['US', 'USW', 'USC', 'USE', 'USR']
    return any(station_id.startswith(prefix) for prefix in us_prefixes)

def main():
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of all CSV files
    all_files = [f for f in os.listdir(args.input_dir) if f.endswith('.csv')]
    print(f"Found {len(all_files)} station files")
    
    # Apply station sampling
    if args.us_only:
        us_files = [f for f in all_files if is_us_station(os.path.splitext(f)[0])]
        print(f"Filtered to {len(us_files)} US station files")
        all_files = us_files
    
    # Sample stations
    num_stations_to_keep = max(1, int(len(all_files) * args.station_sample_rate))
    sampled_files = random.sample(all_files, num_stations_to_keep)
    print(f"Sampled {len(sampled_files)} station files to process")
    
    # Critical columns to keep (index-based for efficiency)
    critical_columns = [
        "STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION", "NAME",
        "PRCP", "TMAX", "TMIN", "SNOW", "WSFG"  # Keep only essential weather parameters
    ]
    
    processed_files = 0
    # Set to track which stations were processed
    processed_stations = set()
    
    # Process each sampled file
    for filename in sampled_files:
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)
        
        station_id = os.path.splitext(filename)[0]
        processed_stations.add(station_id)
        
        try:
            with open(input_path, 'r', encoding='utf-8') as infile:
                reader = csv.reader(infile)
                header = next(reader)
                
                # Map column names to indices
                column_indices = {}
                for i, col in enumerate(header):
                    if col in critical_columns:
                        column_indices[col] = i
                
                # Check if we have all required columns
                if not all(col in column_indices for col in critical_columns):
                    print(f"Skipping {filename} - missing required columns")
                    continue
                
                # Filter rows by date and select critical columns
                filtered_rows = []
                filtered_rows.append([header[column_indices[col]] for col in critical_columns])
                
                for row in reader:
                    try:
                        date_str = row[column_indices["DATE"]]
                        year = int(date_str.split('-')[0])
                        
                        # Filter by year
                        if year >= args.start_year:
                            filtered_rows.append([row[column_indices[col]] for col in critical_columns])
                    except (ValueError, IndexError):
                        # Skip malformed rows
                        continue
                
                # Only write the file if we have data beyond the header
                if len(filtered_rows) > 1:
                    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
                        writer = csv.writer(outfile)
                        writer.writerows(filtered_rows)
                    
                    processed_files += 1
                    if processed_files % 10 == 0:
                        print(f"Processed {processed_files}/{len(sampled_files)} files")
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Create metadata file
    with open(os.path.join(args.output_dir, "DATASET_METADATA.txt"), 'w') as meta_file:
        meta_file.write(f"Original dataset size: ~134GB\n")
        meta_file.write(f"Reduction performed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        meta_file.write(f"Filtering criteria:\n")
        meta_file.write(f"  - Time period: {args.start_year} to present\n")
        meta_file.write(f"  - Geographic filter: {'US stations only' if args.us_only else 'Worldwide'}\n")
        meta_file.write(f"  - Station sampling rate: {args.station_sample_rate}\n")
        meta_file.write(f"  - Critical variables: {', '.join(critical_columns)}\n")
        meta_file.write(f"Number of stations in reduced dataset: {len(processed_stations)}\n")
        meta_file.write(f"Processed files: {processed_files}\n")
        meta_file.write(f"Station IDs included: {', '.join(sorted(processed_stations))}\n")
    
    print(f"Reduction complete. Processed {processed_files} files.")
    print(f"Reduced dataset stored in: {args.output_dir}")

if __name__ == "__main__":
    main()
