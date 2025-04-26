import numpy as np
import pandas as pd

from config import Config, FilePaths
from utils import validate_distance_data
from loguru import logger
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    minimize,
    # NonNegativeIntegers,
    NonNegativeReals,
    Objective,
    Param,
    RangeSet,
    Set,
    SolverFactory,
    TerminationCondition,
    value,
    Var,
)


def main():
    # Load feedstock and distance data
    files = FilePaths()
    feedstock_df = pd.read_parquet(files.feedstock_data)
    feedstock_by_type_df = pd.read_parquet(files.feedstock_by_type_data)
    distance_df = pd.read_parquet(files.distance_data)

    # Load model parameters
    config = Config()

    # Validate distance data
    origin_cells = distance_df.index.tolist()
    destination_cells = distance_df.columns.tolist()
    validate_distance_data(
        origin_cells, destination_cells, config.expected_cell_id_sum
    )

    # Create optimization model
    model = create_model(
        feedstock_df, feedstock_by_type_df, distance_df.values, config
    )

    model.O_labels = origin_cells
    model.D_labels = destination_cells
    solve_and_output(model, "part2_solution.xlsx")


def create_model(
    feedstock_df: pd.DataFrame,
    feedstock_by_type_df: pd.DataFrame,
    distance_matrix: np.ndarray,
    config: Config
) -> ConcreteModel:

    m = ConcreteModel()

    # Initialize origin and destination cells using 1-indexed positions
    m.numO = config.num_O_cells
    m.numD = config.num_suit_bg_cells
    m.O = RangeSet(1, m.numO)
    m.D = RangeSet(1, m.numD)

    # Declare model parameters
    m.DM_FW = Param(initialize=config.DM_FW)
    m.VS_off_farm = Param(initialize=config.VS_off_farm)
    m.BGy_off_farm = Param(initialize=config.BGy_off_farm)
    m.CH4_cont_off_farm = Param(initialize=config.CH4_cont_off_farm)
    m.OPEX_factor_AD = Param(initialize=config.OPEX_factor_AD)
    m.En_cont_CH4 = Param(initialize=config.En_cont_CH4)

    # Indexed parameters (for each O or D)
    ## initial manure available in the cell
    def m_beef_avail_init(m_, i):
        return feedstock_by_type_df["Beef Manure"].tolist()[i-1]

    def m_dairy_avail_init(m_, i):
        return feedstock_by_type_df["Dairy Manure"].tolist()[i-1]
    
    def m_broiler_avail_init(m_, i):
        return feedstock_by_type_df["Broiler Manure"].tolist()[i-1]
    
    def m_pigs_avail_init(m_, i):
        return feedstock_by_type_df["Pigs Manure"].tolist()[i-1]

    ## initial food waste available in the cell
    def fw_avail_init(m_, i):
        return feedstock_df['FW_available'].tolist()[i-1]
    
    # Distance between cells (origin-destination pairs)
    def dist_od_init(m_, i, j):
        return distance_matrix[i-1, j-1]
    
    m.M_beef_available = Param(m.O, initialize=m_beef_avail_init)
    m.M_dairy_available = Param(m.O, initialize=m_dairy_avail_init)
    m.M_broiler_available = Param(m.O, initialize=m_broiler_avail_init)
    m.M_pigs_available = Param(m.O, initialize=m_pigs_avail_init)
    m.FW_available = Param(m.O, initialize=fw_avail_init)
    m.Distance_OD = Param(m.O, m.D, initialize=dist_od_init)

    #FILTER OD PAIRS  - Eliminating unfeasible pairs to reduce model requirement
    feas_M_pairs = []
    feas_FW_pairs = []

    #Factors used for calculating the breakeven distance
    Cost_transp = config.FW_energy  #$/km, representing cost of transporting food waste per km.
    En_cont_FW = config.FW_transport_cost   #GJ/tonne of FW, so the energy content of food waste.
    rng_price = config.rng_price    #May need updated value? $/GJ

    feasible_fw_req = (En_cont_FW * rng_price) / Cost_transp
    for i in range(1, m.numO+1):
        # Manure check
        if feedstock_df['M_available'].tolist()[i-1] > 0:
            feasible_m_req = (feedstock_df['M_pot_out_plant'].tolist()[i-1] * rng_price) / Cost_transp
            for j in range(1, m.numD+1):
                if distance_matrix[i-1, j-1] <= feasible_m_req:
                    feas_M_pairs.append((i,j))

        # Foodwaste check
        if feedstock_df['FW_available'].tolist()[i-1] > 0:
            for j in range(1, m.numD+1):
                if distance_matrix[i-1, j-1] <= feasible_fw_req:
                    feas_FW_pairs.append((i,j))

    m.FEAS_M = Set(initialize=feas_M_pairs, dimen=2)
    m.FEAS_FW = Set(initialize=feas_FW_pairs, dimen=2)

    # Declare decision variables
    m.M_beef_moved   = Var(m.FEAS_M, within=NonNegativeReals)
    m.M_diary_moved   = Var(m.FEAS_M, within=NonNegativeReals)
    m.M_pigs_moved   = Var(m.FEAS_M, within=NonNegativeReals)
    m.M_broiler_moved   = Var(m.FEAS_M, within=NonNegativeReals)
    m.FW_moved  = Var(m.FEAS_FW, within=NonNegativeReals)
    m.Alpha     = Var(m.D, within=Binary)
    m.Emissions  = Var()
    m.Num_AD_plants = Var(domain=NonNegativeReals)
    
    # Define model constraints
    def emission_eq_rule(m_):
        # transport emissions
        transport_emissions = sum(
            (m_.M_beef_moved[i, j] + m_.M_diary_moved[i, j] + \
             m_.M_broiler_moved[i, j] + m_.M_pigs_moved[i, j]) / \
            config.Manure_truck_capacity * m_.Distance_OD[i, j]
            for (i, j) in m_.FEAS_M
        ) * config.Transport_EF / 1000 # convert to kg

        # Leftover manure management emissions
        leftover_emissions = sum(
            (m_.M_beef_available[i] - sum(m_.M_beef_moved[i, j] for j in m_.D if (i, j) in m_.FEAS_M)) * config.Beef_EF + \
            (m_.M_dairy_available[i] - sum(m_.M_diary_moved[i, j] for j in m_.D if (i, j) in m_.FEAS_M)) * config.Dairy_EF + \
            (m_.M_broiler_available[i] - sum(m_.M_broiler_moved[i, j] for j in m_.D if (i, j) in m_.FEAS_M)) * config.Broiler_EF + \
            (m_.M_pigs_available[i] - sum(m_.M_pigs_moved[i, j] for j in m_.D for (i, j) in m_.FEAS_M)) * config.Pigs_EF
            for i in m_.O
        )

        return m_.Emissions == transport_emissions + leftover_emissions

    # Feedstock constraints
    def m_beef_supply_rule(m_, i):
        feasible_Ds = [j for (ii, j) in m_.FEAS_M if ii==i]
        if not feasible_Ds:
            return Constraint.Skip
        return sum(m_.M_beef_moved[i, j] for j in feasible_Ds) <= m_.M_beef_available[i]
    
    def m_dairy_supply_rule(m_, i):
        feasible_Ds = [j for (ii, j) in m_.FEAS_M if ii==i]
        if not feasible_Ds:
            return Constraint.Skip
        return sum(m_.M_diary_moved[i, j] for j in feasible_Ds) <= m_.M_dairy_available[i]
    
    def m_broiler_supply_rule(m_, i):
        feasible_Ds = [j for (ii, j) in m_.FEAS_M if ii==i]
        if not feasible_Ds:
            return Constraint.Skip
        return sum(m_.M_broiler_moved[i, j] for j in feasible_Ds) <= m_.M_broiler_available[i]
    
    def m_pigs_supply_rule(m_, i):
        feasible_Ds = [j for (ii, j) in m_.FEAS_M if ii==i]
        if not feasible_Ds:
            return Constraint.Skip
        return sum(m_.M_pigs_moved[i, j] for j in feasible_Ds) <= m_.M_pigs_available[i]

    def fw_supply_rule(m_, i):
        feasible_Ds = [j for (ii, j) in m_.FEAS_FW if ii==i]
        if not feasible_Ds:
            return Constraint.Skip
        return sum(m_.FW_moved[i, j] for j in feasible_Ds) <= m_.FW_available[i]

    # AD capacity constraints
    def min_capacity_rule(m_, j):
        m_in = sum(m_.M_beef_moved[i, j] + m_.M_diary_moved[i, j] + m_.M_broiler_moved[i, j] + m_.M_pigs_moved[i, j]
                   for (i,jj) in m_.FEAS_M if jj==j)
        fw_in = sum(m_.FW_moved[i, j] for (i, jj) in m_.FEAS_FW if jj==j)
    
        return m_in + fw_in >= m_.Alpha[j] * config.Min_Capacity

    def max_capacity_rule(m_, j):
        m_in = sum(m_.M_beef_moved[i, j] + m_.M_diary_moved[i, j] + m_.M_broiler_moved[i, j] + m_.M_pigs_moved[i, j]
                   for (i,jj) in m_.FEAS_M if jj==j)
        fw_in = sum(m_.FW_moved[i, j] for (i, jj) in m_.FEAS_FW if jj==j)

        return m_in + fw_in <= m_.Alpha[j] * config.Max_Capacity
    
    # AD plants construction constraints
    def num_ad_plants_rule(m_, j):
        return m_.Num_AD_plants == sum(m_.Alpha[j] for j in m_.D)

    def min_ad_plants_rule(m_):
        Max_num_plants = 50
        return m_.Num_AD_plants >= Max_num_plants

    m.ManureBeefSupply = Constraint(m.O, rule = m_beef_supply_rule)
    m.ManureDairySupply = Constraint(m.O, rule = m_dairy_supply_rule)
    m.ManureBroilerSupply = Constraint(m.O, rule = m_broiler_supply_rule)
    m.ManurePigsSupply = Constraint(m.O, rule = m_pigs_supply_rule)
    m.FWSupply = Constraint(m.O, rule = fw_supply_rule)
    m.MinCapacity = Constraint(m.D, rule=min_capacity_rule)
    m.MaxCapacity = Constraint(m.D, rule=max_capacity_rule)
    m.NumADPlantsDefinition = Constraint(rule=num_ad_plants_rule)
    m.MinADPlants = Constraint(rule=min_ad_plants_rule)

    # Declare objective function
    m.EmissionDefinition = Constraint(rule=emission_eq_rule)
    m.Obj = Objective(expr=m.Emissions, sense=minimize)

    return m


def solve_and_output(model, out_file="part2_solution.xlsx"):
    solver = SolverFactory('gurobi')

    solver.options['threads'] = 16  #Changed to processor count for my machine
    solver.options['MIPGap'] = 0.075
    solver.options['OutputFlag'] = 1    # 0 = no output, 1 = output
    solver.options['Symmetry'] = 0
    solver.options['NodefileStart'] = 0.5

    results = solver.solve(model, tee=True)

    #Need to check if solver "worked"
    logger.info(f"Solver Status: {results.solver.status}")
    logger.info(f"Termination Condition: {results.solver.termination_condition}")

    feasible_conditions = {
        TerminationCondition.optimal,
        TerminationCondition.feasible,
        TerminationCondition.locallyOptimal,
        TerminationCondition.globallyOptimal
    }

    #We want only to write to excel if the solver "worked" and found a feasible solution
    if (results.solver.status == 'ok' and
        results.solver.termination_condition in feasible_conditions):

        #Only output transfers/flows where the value is greater than 0 (ie feedstock moved from O to D)
        manure_solution = [(
            model.O_labels[i-1],
            model.D_labels[j-1],
            value(model.M_beef_moved[i, j]),
            value(model.M_diary_moved[i, j]),
            value(model.M_broiler_moved[i, j]),
            value(model.M_pigs_moved[i, j])
        ) for (i, j) in model.FEAS_M if value(model.M_beef_moved[i, j]) + \
            value(model.M_diary_moved[i, j]) + \
            value(model.M_broiler_moved[i, j]) + \
            value(model.M_pigs_moved[i, j]) > 0
        ]
        # fw_solution = [
        #     (model.O_labels[i-1], model.D_labels[j-1], value(model.FW_moved[i, j]))
        #     for (i, j) in model.FEAS_FW if value(model.FW_moved[i, j]) > 0
        # ]
        ad_solution = [
            (model.D_labels[j-1], value(model.Alpha[j]))
            for j in model.D if value(model.Alpha[j]) > 0
        ]

        #EXTRACTING FINAL COST AND ENERGY VALUES
        emissions_result = value(model.Emissions)
        num_plants_result = value(model.Num_AD_plants)

        #Write solutions to Excel
        with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
            df_manure = pd.DataFrame(
                data=manure_solution,
                columns=['Origin','Destination','Manure_moved_beef', 'Manure_moved_dairy',
                'Manure_moved_broiler', 'Manure_moved_pigs']
            )
            df_manure.to_excel(writer, sheet_name='ManureTransfers', index=False)

            # df_fw = pd.DataFrame(fw_solution, columns=['Origin','Destination','FW_moved'])
            # df_fw.to_excel(writer, sheet_name='FW Transfers', index=False)

            df_emissions_num_plants = pd.DataFrame({'Value':[emissions_result, num_plants_result]},
                                              index=['Emissions', 'Num_AD_plants'])
            df_emissions_num_plants.to_excel(writer, sheet_name='Emissions_Num_Plants', index=True)

            df_ad = pd.DataFrame(ad_solution, columns=['Destination','AD_plant'])
            df_ad.to_excel(writer, sheet_name='AD_plant', index=False)

        logger.info(f'Solution written to {out_file}')

    else:
        logger.info('No feasible (or optimal) solution found. Solver status:')
        logger.info(f"Status = {results.solver.status}")
        logger.info(f"Termination = {results.solver.termination_condition}")
        logger.info("No solution file will be written.")


if __name__ == '__main__':
    main()
