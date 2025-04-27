from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    num_O_cells: int = 3996
    num_suit_bg_cells: int = 3996
    DM_FW: float = 0.25 #Fraction [-]
    VS_off_farm: float = 0.9    #Fraction [-] of VS that off-gases?
    BGy_off_farm: int = 500 #biogas yield in m^3/tonne of VS
    CH4_cont_off_farm: float = 0.6  #fraction of CH4 content ...?
    OPEX_factor_AD: float = 0.07   #fraction [-]
    En_cont_CH4: float = 0.037  #energy content in GJ/m^3 of CH4 (aka H)
    Project_life: int = 20  #project lifetime, years (aka T)
    Amortization_factor: float = 0.0872 #//interest rate - 6% and number of periods or payments - 20
    MCE: float = 0.85   #methane capture efficiency from the digestor [-] (aka eta, small)
    Amt_financed: float = 0.75  #portion of capital cost financed? fraction [-]
    C_pipe: int = 1208194   #$/km capital cost for building connecting pipelines from the digestor
    f: float = 3.0  # Energy / Emissions constraint; set f=1 as default
    PV_FP_coeff: float = 10.7   #coefficient derived from a formula used to calculate the present value of future payments
    FW_energy: float = 2.122875 #GJ/tonne of food waste (energy content) (aka s)
    FW_transport_cost: float = 1.09869  # $/km cost of transporting foodwaste
    rng_price: float = 60.0  #May need updated value? $/GJ
    Min_Capacity: int = 30000   #tonnes of feedstock per year, capacity of digester
    Max_Capacity: int = 120000  #tonnes of feedstock per year, capacity of digester
    expected_cell_id_sum: int = 408848344

    # Part 2
    Beef_EF: float = 0.143  # kg N2O per tonne of manure
    Dairy_EF: float = 0.024  # kg N2O per tonne of manure
    Pigs_EF: float = 0.0001 # kg N2O per tonne of manure
    Broiler_EF: float = 1.3  # kg N2O per tonne of manure
    Transport_EF: float = 0.002  # g/km per tonne of manure
    Upgrade_biogas_EF: float = 0.00022  # lb / MMBtu
    Manure_truck_capacity: float = 22.7  # tonnes
    FW_truck_capacity: float = 20.0  # tonnes
    Upgrading_efficiency: float = 0.85  # fraction of biogas that is upgraded


@dataclass
class FilePaths:
    # Data
    data_file: Path = Path("data/data_class_project.xlsx")
    feedstock_data: Path = Path("data/feedstock_data.parquet")
    distance_data: Path = Path("data/distance_data.parquet")
    feedstock_by_type_data: Path = Path("data/feedstock_by_type_data.parquet")
    
    # Results
    part1_results: Path = Path("results/part1_solution.xlsx")
    part2_results: Path = Path("results/part2_solution.xlsx")
