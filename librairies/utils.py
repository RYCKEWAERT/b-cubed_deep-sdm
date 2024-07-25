import torch
import random
import numpy as np
import warnings
import os
import numpy as np
import pandas as pd
import rasterio
import torch


def import_belgium_data():
    # Path to the directory containing the TIFF files
    rasters_dir = 'data/Belgium/rasters_crop/'
    # list files in the directory with .tif extension
    raster_files = [f for f in os.listdir(rasters_dir) if f.endswith('.tif')]

    # Path to the CSV file
    csv_file = 'data/Belgium/belgium_2010.csv'
    df = pd.read_csv(csv_file, sep='\t')



    # Create an empty tensor to store the raster values
    tensor = torch.zeros((df.shape[0],len(raster_files)), dtype = torch.float32)
    idx =0
    # Iterate over the TIFF files in the directory
    for filename in os.listdir(rasters_dir):
        if filename.endswith('.tif'):
            tiff_path = os.path.join(rasters_dir, filename)
            
            # Open the TIFF file
            with rasterio.open(tiff_path) as src:
                # Read the raster values within the latitude and longitude range
                values = list(src.sample([(lon, lat) for lon, lat in zip(df['decimallongitude'], df['decimallatitude'])]))
                
                # Store the values in the tensor
                list_to_tensor = np.array(values).flatten()
                tensor[:,idx] = torch.tensor(list_to_tensor, dtype = torch.float32)
                idx += 1
    return tensor, df

def disable_warnings():
    warnings.filterwarnings("ignore")

def set_seed(global_seed):
    """
    Sets the random seeds for reproducibility.

    Args:
        args (argparse.Namespace): The arguments containing the seed values.

    Note:

    """
    random.seed(global_seed)  # Set the random seed for the random module
    np.random.seed(global_seed)  # Set the random seed for the numpy module
    torch.manual_seed(global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(global_seed)

    
def get_random_seed_list(nbr):
    return [random.randint(1, 1000) for _ in range(nbr)]
