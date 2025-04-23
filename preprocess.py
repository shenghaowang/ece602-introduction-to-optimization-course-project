import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger


def main():
    data_file = "data_class_project.xlsx"
    num_cells = 3996
        
    logger.info(f"Loading feedstock data from {data_file} ...")
    start_time = time.time()
    feedstock_df = pd.read_excel(
        data_file,
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
    logger.info(f"Loaded feedstock data in {time.time() - start_time:.2f} seconds.")
    logger.debug(f"Feedstock data: {feedstock_df.shape}")

    start_time = time.time() 
    distance_df = pd.read_excel(
        data_file,
        sheet_name="OD",
        header= 0,
        index_col= 0,
        nrows= num_cells
    )
    logger.info(f"Loaded distance data in {time.time() - start_time:.2f} seconds.")
    logger.debug(f"Distance data: {distance_df.shape}")
    
    # Convert dataframes to PyArrow tables and save as Parquet files
    feedstock_table = pa.Table.from_pandas(feedstock_df)
    pq.write_table(feedstock_table, "data/feedstock_data.parquet")
    logger.info("Feedstock data saved to feedstock_data.parquet")

    distance_table = pa.Table.from_pandas(distance_df)
    pq.write_table(distance_table, "data/distance_data.parquet")
    logger.info("Distance data saved to distance_data.parquet")

    # Load the Parquet files to verify
    logger.info("Loading Parquet files to verify...")
    start_time = time.time()
    feedstock_df = pd.read_parquet("data/feedstock_data.parquet")
    logger.info(f"Loaded feedstock data in {time.time() - start_time:.2f} seconds.")
    logger.info(f"Feedstock data loaded from Parquet: {feedstock_df.shape}")
    
    start_time = time.time()
    distance_df = pd.read_parquet("data/distance_data.parquet")
    logger.info(f"Distance data loaded from Parquet: {distance_df.shape}")
    logger.info(f"Loaded distance data in {time.time() - start_time:.2f} seconds.")



if __name__ == '__main__':
    main()
