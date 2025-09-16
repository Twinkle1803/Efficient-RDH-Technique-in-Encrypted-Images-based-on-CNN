import os
import cv2
import numpy as np
import pandas as pd
import argparse

def calculate_metrics(imageA_temp, imageB):
    """
    Calculate MSE, Mean, and Variance between two images for pixels where (i + j) % 2 != 0.
    Args:
        imageA_temp: Original image (grayscale).
        imageB: Predicted or processed image (grayscale).
    Returns:
        mse: Mean Squared Error
        mean: Mean of differences
        var: Variance of differences
    """
    # Convert images to float for accurate computation
    imageA_temp = imageA_temp.astype(np.float32)
    imageB = imageB.astype(np.float32)
    
    # Create an empty array for selected pixels
    imageA = np.zeros((512, 512), dtype=np.float32)
    for i in range(512):
        for j in range(512):
            if (i + j) % 2 != 0:
                imageA[i, j] = imageA_temp[i, j]
    
    # Compute difference only on the selected pixels
    diff_values = imageA - imageB
    
    # Compute metrics
    mse = np.mean(diff_values ** 2)
    mean = np.mean(np.abs(diff_values))
    var = np.var(diff_values)
    
    return mse, mean, var

def process_ucid_dataset(input_dir, output_dir, output_xlsx, train_state):
    """
    Process all images in the UCID dataset, calculate metrics, and save to XLSX.
    Args:
        input_dir: Directory containing the original images.
        output_dir: Directory containing the predicted/processed images.
        output_xlsx: Path to save the output Excel file.
        train_state: The train state number.
    """
    results = []
    
    # Ensure the dataset paths exist
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory '{output_dir}' does not exist.")
    
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Check if corresponding predicted image exists
        if not os.path.exists(output_path):
            print(f"Predicted image not found for {filename}. Skipping...")
            continue
        
        # Read images in grayscale
        original_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        predicted_image = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
        
        if original_image is None or predicted_image is None:
            print(f"Error reading images for {filename}. Skipping...")
            continue
        
        # Calculate metrics
        mse, mean, var = calculate_metrics(original_image, predicted_image)
        results.append({"Image Name": filename, "MSE": mse, "Mean": mean, "Variance": var})
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate averages
    avg_mse = results_df["MSE"].mean()
    avg_mean = results_df["Mean"].mean()
    avg_var = results_df["Variance"].mean()
    
    averages = pd.DataFrame([{
        "Image Name": "AVERAGE",
        "MSE": avg_mse,
        "Mean": avg_mean,
        "Variance": avg_var
    }])
    
    # Concatenate averages with results
    results_df = pd.concat([results_df, averages], ignore_index=True)
    
    # Save results to an XLSX file
    results_df.to_excel(output_xlsx, index=False, engine='openpyxl')
    print(f"Metrics calculated and saved to '{output_xlsx}'")

    # Save overall averages for all train states
    update_overall_averages(train_state, avg_mse, avg_mean, avg_var)

def update_overall_averages(train_state, avg_mse, avg_mean, avg_var):
    """
    Update the overall averages file that stores MSE, Mean, and Variance for all train states.
    Args:
        train_state: The train state number.
        avg_mse: Average MSE for the train state.
        avg_mean: Average Mean for the train state.
        avg_var: Average Variance for the train state.
    """
    overall_file = "overall_averages.xlsx"
    
    if os.path.exists(overall_file):
        overall_df = pd.read_excel(overall_file, engine='openpyxl')
    else:
        overall_df = pd.DataFrame(columns=["Train State", "MSE", "Mean", "Variance"])
    
    # Append new entry
    new_entry = pd.DataFrame([{
        "Train State": train_state,
        "MSE": avg_mse,
        "Mean": avg_mean,
        "Variance": avg_var
    }])
    
    overall_df = pd.concat([overall_df, new_entry], ignore_index=True)
    
    # Save back to Excel
    overall_df.to_excel(overall_file, index=False, engine='openpyxl')
    print(f"Overall averages updated in '{overall_file}'")

if __name__ == "__main__":
    # Argument parser for train state input
    parser = argparse.ArgumentParser(description="Calculate image metrics for a given train state.")
    parser.add_argument("train_state", type=int, help="Train state number (e.g., 255)")
    args = parser.parse_args()
    
    train_state = args.train_state
    
    # Define directories dynamically
    input_directory = "./UCID1338"  # Original images directory
    output_directory = f"./predicted_images_method5_ts{train_state}"  # Predicted images directory
    output_xlsx_path = f"ucid_metrics_method5_ts{train_state}.xlsx"  # Output Excel file
    
    # Process dataset and store results in XLSX
    process_ucid_dataset(input_directory, output_directory, output_xlsx_path, train_state)
