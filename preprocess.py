import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from config import Config, FilePaths


def main():
    config = Config()
    file_paths = FilePaths()
    num_cells = config.num_O_cells
    
    # Load all feeedstock data
    logger.info(f"Loading all feedstock data from {file_paths.data_file} ...")
    start_time = time.time()
    feedstock_df = pd.read_excel(
        file_paths.data_file,
        sheet_name="All_Feedstock",
        header=0,
        usecols=[
            "FW_available", #Tonnes of foodwaste available in each cell
            "M_available", #Tonnes of manure available in each cell
            "M_CH4_potential", #manure methane potential (m^3/tonne)
            "Dist_pipeline", #distance of cell to nearest pipeline
            "M_pot_out_plant" #Potential plant output of manure #Energy content of manure in the cell (GJ/tonne)?
        ],
        nrows = num_cells
    )
    logger.info(f"Loaded all feedstock data in {time.time() - start_time:.2f} seconds.")
    logger.debug(f"All feedstock data: {feedstock_df.shape}")

    # Load distance data
    start_time = time.time() 
    distance_df = pd.read_excel(
        file_paths.data_file,
        sheet_name="OD",
        header= 0,
        index_col= 0,
        nrows= num_cells
    )
    logger.info(f"Loaded distance data in {time.time() - start_time:.2f} seconds.")
    logger.debug(f"Distance data: {distance_df.shape}")

    # Load feedstock by category data
    logger.info("Loading feedstock by category data...")
    start_time = time.time()
    feedstock_by_category_df = pd.read_excel(
        file_paths.data_file,
        sheet_name="Feedstock Categorized",
        header=0,
        usecols=[
            "Beef Manure",
            "Dairy Manure",
            "Broiler Manure",
            "Pigs Manure",
        ],
        nrows=num_cells
    )
    logger.info(f"Loaded feedstock by category data in {time.time() - start_time:.2f} seconds.")
    logger.debug(f"Feedstock by category data: {feedstock_by_category_df.shape}")
    
    # Convert dataframes to PyArrow tables and save as Parquet files
    feedstock_table = pa.Table.from_pandas(feedstock_df)
    pq.write_table(feedstock_table, file_paths.feedstock_data)
    logger.info(f"Feedstock data saved to {file_paths.feedstock_data}")

    distance_table = pa.Table.from_pandas(distance_df)
    pq.write_table(distance_table, file_paths.distance_data)
    logger.info(f"Distance data saved to {file_paths.distance_data}")

    feedstock_by_type_table = pa.Table.from_pandas(feedstock_by_category_df)
    pq.write_table(feedstock_by_type_table, file_paths.feedstock_by_type_data)
    logger.info(f"Feedstock by category data saved to {file_paths.feedstock_by_type_data}")

    # Load the Parquet files to verify
    logger.info("Loading Parquet files to verify...")
    start_time = time.time()
    feedstock_df = pd.read_parquet(file_paths.feedstock_data)
    logger.info(f"Loaded feedstock data in {time.time() - start_time:.2f} seconds.")
    logger.info(f"Feedstock data loaded from Parquet: {feedstock_df.shape}")
    
    start_time = time.time()
    distance_df = pd.read_parquet(file_paths.distance_data)
    logger.info(f"Distance data loaded from Parquet: {distance_df.shape}")
    logger.info(f"Loaded distance data in {time.time() - start_time:.2f} seconds.")

    start_time = time.time()
    feedstock_by_category_df = pd.read_parquet(file_paths.feedstock_by_type_data)
    logger.info(f"Feedstock by category data loaded from Parquet: {feedstock_by_category_df.shape}")
    logger.info(f"Loaded feedstock by category data in {time.time() - start_time:.2f} seconds.")


if __name__ == '__main__':
    main()
