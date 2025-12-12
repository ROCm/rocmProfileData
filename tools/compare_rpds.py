import os
import sqlite3
import pandas as pd
import sys

def extract_rccl_kernels_from_file(db_path):
    """
    Extract RCCL kernels from a single SQLite file.
    """
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query to extract RCCL kernels
        query = """
        SELECT id, gpuId, queueId, sequenceId, start, end, duration, stream, 
               gridX, gridY, gridZ, workgroupX, workgroupY, workgroupZ, 
               groupSegmentSize, privateSegmentSize, kernelName 
        FROM kernel 
        WHERE kernelName LIKE "%rccl%"
        """
        cursor.execute(query)

        # Fetch all results
        rows = cursor.fetchall()

        # Define column names
        columns = [desc[0] for desc in cursor.description]

        # Create a pandas DataFrame
        df = pd.DataFrame(rows, columns=columns)

        # Close the connection
        conn.close()

        return df
    except Exception as e:
        print(f"Error processing file {db_path}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

def process_rpd_files(input_dir):
    """
    Process all .rpd files in the input directory and return a dictionary of DataFrames.
    """
    data_frames = {}

    # Iterate over all files in the input directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".rpd"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")

                # Extract RCCL kernels from the file
                df = extract_rccl_kernels_from_file(file_path)
                if not df.empty:
                    # Store the DataFrame in the dictionary with the file path as the key
                    data_frames[file_path] = df

    return data_frames

def compare_data_frames(data_frames):
    """
    Compare multiple DataFrames to check for consistency in RCCL calls.
    """
    if not data_frames:
        print("No data frames to compare.")
        return

    # Extract all DataFrames and their file paths
    file_paths = list(data_frames.keys())
    dfs = list(data_frames.values())

    # Check if all DataFrames have the same number of RCCL calls
    num_calls = [len(df) for df in dfs]
    if len(set(num_calls)) != 1:
        print("Mismatch in the number of RCCL calls across files:")
        for path, count in zip(file_paths, num_calls):
            print(f"{path}: {count} calls")
        return

    print("All files have the same number of RCCL calls.")

    # Check for overlapping start and end times for corresponding RCCL calls
    error_count = 0
    total_overlap_percentage = 0
    num_calls = len(dfs[0])  # Number of RCCL calls in each DataFrame
    for i in range(num_calls):
        for j in range(len(dfs)):
            for k in range(j + 1, len(dfs)):
                start_j = dfs[j].iloc[i]["start"]
                end_j = dfs[j].iloc[i]["end"]
                start_k = dfs[k].iloc[i]["start"]
                end_k = dfs[k].iloc[i]["end"]

                # Calculate overlap duration
                overlap_start = max(start_j, start_k)
                overlap_end = min(end_j, end_k)
                overlap_duration = max(0, overlap_end - overlap_start)

                # Calculate the total duration of the two calls
                total_duration = max(end_j, end_k) - min(start_j, start_k)

                # Calculate overlap percentage
                if total_duration > 0:
                    overlap_percentage = (overlap_duration / total_duration) * 100
                    total_overlap_percentage += overlap_percentage

                # Check if there is no overlap
                if overlap_duration == 0:
                    error_count += 1
                    print(f"Error in RCCL call {i} between files {file_paths[j]} and {file_paths[k]}: No overlap in start/end times.")

    # Calculate overall average overlap percentage
    if num_calls > 0 and len(dfs) > 1:
        total_comparisons = num_calls * (len(dfs) * (len(dfs) - 1)) / 2
        average_overlap_percentage = total_overlap_percentage / total_comparisons
        print(f"Overall average overlap percentage: {average_overlap_percentage:.2f}%")

    print(f"Total number of errors: {error_count}")

if __name__ == "__main__":
    # Input directory containing .rpd files
    # input_directory = "/home/docker/jax-llm-examples/llama3/clock_sync_results/with_sync"
    # input_directory = "/home/docker/jax-llm-examples/llama3/clock_sync_results/without_sync"

    if len(sys.argv) != 2:
        print("Usage: python compare_rpds.py <input_directory>")
        sys.exit(1)

    input_directory = sys.argv[1]
    # Process the .rpd files and extract RCCL kernels
    data_frames = process_rpd_files(input_directory)

    compare_data_frames(data_frames)
    # Save each DataFrame to a separate CSV file
    for file_path, df in data_frames.items():
        output_csv = f"{file_path}.rccl_kernels.csv"
        df.to_csv(output_csv, index=False)
        print(f"RCCL kernel data for {file_path} saved to {output_csv}")

# python3 /home/docker/rocmProfileData/tools/compare_rpds.py