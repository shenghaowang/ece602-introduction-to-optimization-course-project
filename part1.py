import numpy as np
import pandas as pd
from loguru import logger
from pyomo.environ import (
    ConcreteModel,
    RangeSet,
    Param,
    Var,
    NonNegativeReals,
    Binary,
    Constraint,
    Objective,
    minimize,
    Set,
    value,
    SolverFactory,
    TerminationCondition
)

from config import Config, FilePaths
from utils import validate_distance_data


def main():
    # Load feedstock and distance data
    files = FilePaths()
    feedstock_df = pd.read_parquet(files.feedstock_data)
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
    model = create_model(feedstock_df, distance_df.values, config)

    model.O_labels = origin_cells
    model.D_labels = destination_cells
    solve_and_output(model, "part1_solution.xlsx")


# Defining Model
def create_model(
    feedstock_df: pd.DataFrame,
    distance_matrix: pd.DataFrame,
    config: Config
) -> ConcreteModel:
    """
    Building Pyomo Model
    V1: Minimize cost, enforcing minimum energy production, some fraction of the maximum 303,759,344.8.
    """
    
    m = ConcreteModel() # Model creation / defining the model

    # Set for origins and destinations (using 1-indexed positions)
    m.numO = config.num_O_cells
    m.numD = config.num_suit_bg_cells
    m.O = RangeSet(1, m.numO)
    m.D = RangeSet(1, m.numD)

    #Scalar parameters (need to move into the model)
    m.DM_FW = Param(initialize=config.DM_FW)
    m.VS_off_farm = Param(initialize=config.VS_off_farm)
    m.BGy_off_farm = Param(initialize=config.BGy_off_farm)
    m.CH4c_off_farm = Param(initialize=config.CH4_cont_off_farm)
    m.OPEX_factor_AD = Param(initialize=config.OPEX_factor_AD)
    m.En_cont_CH4 = Param(initialize=config.En_cont_CH4)
    m.Amortization_factor = Param(initialize=config.Amortization_factor)
    m.MCE = Param(initialize=config.MCE)
    m.Amt_financed = Param(initialize=config.Amt_financed)
    m.C_pipe = Param(initialize=config.C_pipe)
    m.f = Param(initialize=config.f)

    #Indexed parameters (for each O or D)
    #initial manure available in the cell
    def m_avail_init(m_, i):
        return feedstock_df['M_available'].tolist()[i-1] #Need to shift (data for correct indexing (due to header row?))
    
    #initial food waste available in the cell
    def fw_avail_init(m_, i):
        return feedstock_df['FW_available'].tolist()[i-1]

    #Manure methane potential (volumetric) (m^3/tonne)
    def m_ch4_pot_init(m_, i):
        return feedstock_df['M_CH4_potential'].tolist()[i-1]

    #distance to pipeline (from destination cell, D)
    def dist_pl_init(m_, j):
        return feedstock_df['Dist_pipeline'].tolist()[j-1]

    # Manure potential energy production (GJ/tonne)
    def m_en_pot_init(m_, i):
        return feedstock_df['M_pot_out_plant'].tolist()[i-1]
    
    # Distance between cells (origin-destination pairs)
    def dist_od_init(m_, i, j):
        return distance_matrix[i-1, j-1]

    m.M_available = Param(m.O, initialize = m_avail_init) #initial available manure of origin cell (O)
    m.FW_available = Param(m.O, initialize = fw_avail_init)
    m.M_CH4_potential = Param(m.O, initialize = m_ch4_pot_init)
    m.Dist_pipeline = Param(m.D, initialize = dist_pl_init)
    m.M_pot_out_plant = Param(m.O, initialize = m_en_pot_init)
    m.Distance_OD = Param(m.O, m.D, initialize=dist_od_init)

    #FILTER OD PAIRS  - Eliminating unfeasible pairs to reduce model requirement
    feas_M_pairs = []
    feas_FW_pairs = []

    # Factors used for calculating the breakeven distance
    Cost_transp = config.FW_energy #$/km, representing cost of transporting food waste per km.
    En_cont_FW = config.FW_transport_cost #GJ/tonne of FW, so the energy content of food waste.
    rng_price = config.rng_price # $/GJ

    feasible_fw_req = (En_cont_FW * rng_price) / Cost_transp
    for i in range(1, m.numO+1):
        # Manure check
        if feedstock_df['M_available'][i-1] > 0:
            feasible_m_req = (feedstock_df['M_pot_out_plant'][i-1] * rng_price) / Cost_transp
            for j in range(1, m.numD+1):
                if distance_matrix[i-1, j-1] <= feasible_m_req:
                    feas_M_pairs.append((i,j))

        # Foodwaste check
        if feedstock_df['FW_available'][i-1] > 0:
            for j in range(1, m.numD+1):
                if distance_matrix[i-1,j-1] <= feasible_fw_req:
                    feas_FW_pairs.append((i,j))

    m.FEAS_M = Set(initialize=feas_M_pairs, dimen=2)
    m.FEAS_FW = Set(initialize=feas_FW_pairs, dimen=2)

    # DECISION VARIABLES
    m.M_moved   = Var(m.FEAS_M, within=NonNegativeReals)
    m.FW_moved  = Var(m.FEAS_FW, within=NonNegativeReals)
    m.Alpha     = Var(m.D, within=Binary)
    m.Cost      = Var()
    m.Energy    = Var(domain=NonNegativeReals)

    # Defining Model Constraints
    def cost_eq_rule(m_):
        PlaceC = 150.3 #digester sizing cost for manure ($/t capacity)
        PlaceD = 466.3 #digester sizing cost for foodwaste ($/t capacity)
        PlaceE = 2119947 #fixed cost of biogas plant ($) (Hendry & Bida, 2022)
        PlaceF = 0.15 #coefficient for transportation cost assoc. with manure
        PlaceG = 0.051 #coefficient for transportation cost assoc. with foodwaste
        PlaceH = 10.7 #coefficient derived from a formula used to calculate the present value of future payments.

        #Transport cost = manure transp cost + fw transp cost
        capex_manure = sum(PlaceC * m_.M_moved[i, j] for (i, j) in m_.FEAS_M) 
        capex_fw = sum(PlaceD * m_.FW_moved[i, j] for (i, j) in m_.FEAS_FW)
        capex_fixed = sum(PlaceE * m_.Alpha[j] for j in m_.D)
        
        capex_sum = capex_fixed + capex_manure + capex_fw 
        + sum(m_.Dist_pipeline[j] * m_.C_pipe * m_.Alpha[j] for j in m_.D)
        
        unfinanced = capex_sum * (1 - m_.Amt_financed)
        financed = capex_sum * m_.Amortization_factor * m_.Amt_financed
        opex_part = (capex_fixed + capex_manure + capex_fw) * m_.OPEX_factor_AD

        haul_manure = sum((PlaceF * m_.Distance_OD[i, j] + 0.5) * m_.M_moved[i, j]
                          for (i, j) in m_.FEAS_M)
        haul_fw = sum(PlaceG * m_.Distance_OD[i, j] * m_.FW_moved[i, j] 
                      for (i, j) in m_.FEAS_FW)
        feedstock_transport = haul_manure + haul_fw
        return  m_.Cost == unfinanced + PlaceH * (feedstock_transport + financed + opex_part)

    #FEEDSTOCK CONSTRAINTS
    def m_supply_rule(m_, i):
        feasible_Ds = [j for (ii, j) in m_.FEAS_M if ii==i]
        if not feasible_Ds:
            return Constraint.Skip
        return sum(m_.M_moved[i, j] for j in feasible_Ds) <= m_.M_available[i]


    def fw_supply_rule(m_, i):
        feasible_Ds = [j for (ii, j) in m_.FEAS_FW if ii==i]
        if not feasible_Ds:
            return Constraint.Skip
        return sum(m_.FW_moved[i, j] for j in feasible_Ds) <= m_.FW_available[i]
    
    #AD CAPACITY CONSTRAINTS
    def min_capacity_rule(m_, j):
        Min_Capacity = 30000 #tonnes of feedstock per year, capacity of digester

        m_in = sum(m_.M_moved[i, j] for (i,jj) in m_.FEAS_M if jj==j)
        fw_in = sum(m_.FW_moved[i, j] for (i, jj) in m_.FEAS_FW if jj==j)
        return m_in + fw_in >= m_.Alpha[j] * Min_Capacity

    def max_capacity_rule(m_, j):
        Max_Capacity = 120000 #tonnes of feedstock per year, capacity of digester

        m_in = sum(m_.M_moved[i, j] for (i,jj) in m_.FEAS_M if jj==j)
        fw_in = sum(m_.FW_moved[i, j] for (i, jj) in m_.FEAS_FW if jj==j)
        return m_in + fw_in <= m_.Alpha[j] * Max_Capacity
    
    # ENERGY CONSTRAINTS
    # Energy production calculation
    def energy_eq_rule(m_):
        #10.7 coefficient derived from a formula used to calculate the present value of future payments.
        energy_m = sum(m_.M_moved[i,j] * m_.M_pot_out_plant[i]
                              for (i,j) in m_.FEAS_M)
        energy_fw = sum(m_.MCE * m_.En_cont_CH4 * 
                        (m_.FW_moved[i,j] * m_.DM_FW * m_.VS_off_farm * 
                         m_.BGy_off_farm * m_.CH4c_off_farm) * m_.Alpha[j] 
                        for (i, j) in m_.FEAS_FW)
        return m_.Energy == energy_m + energy_fw
    

    #Energy production constraint
    def min_energy_rule(m_):
        Max_Energy = 29186554.47 #From re-run maximum energy scenario
        return m_.Energy >= m_.f * Max_Energy

    m.CostDefinition = Constraint(rule=cost_eq_rule)
    m.ManureSupply = Constraint(m.O, rule = m_supply_rule)
    m.FWSupply = Constraint(m.O, rule = fw_supply_rule)
    m.MinCapacity = Constraint(m.D, rule=min_capacity_rule)
    m.MaxCapacity = Constraint(m.D, rule=max_capacity_rule)
    m.EnergyDefinition = Constraint(rule=energy_eq_rule)
    m.MinEnergy = Constraint(rule=min_energy_rule)

    # Objective function
    #Minimizing system cost while meeting the constraints (above)
    m.Obj = Objective(expr = m.Cost, sense = minimize)

    return m


def solve_and_output(model, out_file="part1_solution.xlsx"):
    """
    Solve the (Pyomo) model using CPLEX & write the results to an Excel file
    """
    solver = SolverFactory('gurobi')

    solver.options['threads'] = 16 # Changed to processor count for my machine
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
        manure_solution = [
            (model.O_labels[i-1], model.D_labels[j-1], value(model.M_moved[i, j]))
            for (i, j) in model.FEAS_M if value(model.M_moved[i, j]) > 0
        ]
        fw_solution = [
            (model.O_labels[i-1], model.D_labels[j-1], value(model.FW_moved[i, j]))
            for (i, j) in model.FEAS_FW if value(model.FW_moved[i, j]) > 0
        ]
        ad_solution = [
            (model.D_labels[j-1], value(model.Alpha[j]))
            for j in model.D if value(model.Alpha[j]) > 0
        ]

        #EXTRACTING FINAL COST AND ENERGY VALUES
        cost_result = value(model.Cost)
        energy_result = value(model.Energy)

        #Write solutions to Excel
        with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
            df_manure = pd.DataFrame(manure_solution, columns=['Origin','Destination','Manure_moved'])
            df_manure.to_excel(writer, sheet_name='ManureTransfers', index=False)

            df_fw = pd.DataFrame(fw_solution, columns=['Origin','Destination','FW_moved'])
            df_fw.to_excel(writer, sheet_name='FW Transfers', index=False)

            df_cost_energy = pd.DataFrame({'Value':[cost_result, energy_result]},
                                              index=['Cost', 'Energy'])
            df_cost_energy.to_excel(writer, sheet_name='Cost_Energy', index=True)

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
