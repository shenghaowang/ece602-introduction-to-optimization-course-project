from typing import List

from loguru import logger


def validate_distance_data(
    origin_cells: List[int],
    destination_cells: List[int],
    expected_cell_id_sum: int
) -> None:
    try:
        origin_cells = [float(cell) for cell in origin_cells]
        destination_cells = [float(cell) for cell in destination_cells]
    
    except ValueError as e:
        logger.error("Error converting cell IDs to float: {}", e)
        raise 

    assert abs(sum(origin_cells) - sum(destination_cells)) < 1e-6, (
        f"Sum of origin cells {sum(origin_cells)} does not match sum of destination cells {sum(destination_cells)}"
    )
    logger.info("Origin cells check passed")

    assert abs(sum(destination_cells) - expected_cell_id_sum) < 1e-6, (
        f"Sum of destination cells {sum(origin_cells)} does not match expected sum {expected_cell_id_sum}"
    )

    logger.info("Destination cells check passed")
