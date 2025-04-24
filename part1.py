
import time

import pandas as pd
from loguru import logger
from pyomo.environ import (
    ConcreteModel, RangeSet, Param, Var, NonNegativeReals, Binary,
    Constraint, Objective, minimize, Set, value, SolverFactory, TerminationCondition
)


def main():
    feedstock_data = "data/feedstock_data.parquet"
    distance_data = "data/distance_data.parquet"
    
    # Load the data
    data = load_data(feedstock_data, distance_data)
    model = create_model(data)

    #Attach OD sheet labels to the model for later use in output
    model.O_labels = data['O_labels']
    model.D_labels = data['D_labels']
    solve_and_output(model, "Solution_file_cost_min_f100.xlsx")


#Import data
def load_data(feedstock_data, distance_data):

    #Parameters
    num_O_cells = 3996
    
    DM_FW = 0.25 #Fraction [-]
    VS_off_farm = 0.9 #Fraction [-] of VS that off-gases?
    BGy_off_farm = 500 #biogas yield in m^3/tonne of VS
    CH4_cont_off_farm = 0.6 #fraction of CH4 content ...?
    OPEX_factor_AD = 0.07 #fraction [-]
    En_cont_CH4 = 0.037 #energy content in GJ/m^3 of CH4 (aka H)
    #Project_life = 20 #project lifetime, years (aka T)
    Amortization_factor = 0.0872 #//interest rate - 6% and number of periods or payments - 20
    MCE = 0.85 #methane capture efficiency from the digestor [-] (aka eta, small)
    Amt_financed = 0.75 #portion of capital cost financed? fraction [-]
    C_pipe = 1208194 #$/km capital cost for building connecting pipelines from the digestor
    f = 0.25 #Energy constraint; set f=1 as default
    #PV_FP_coeff = 10.7 #coefficient derived from a formula used to calculate the present value of future payments

    FW_energy = 2.122875 #GJ/tonne of food waste (energy content) (aka s)
    FW_transport_cost = 1.098695892 # $/km cost of transporting foodwaste

    # Load feedstock data
    start_time = time.time()
    d_feedstock = pd.read_parquet(feedstock_data)
    logger.info(f"Loaded feedstock data in {time.time() - start_time:.2f} seconds.")

    # Load OD pair distance data
    start_time = time.time()
    d_distance = pd.read_parquet(distance_data)
    logger.info(f"Loaded distance data in {time.time() - start_time:.2f} seconds.")

    #Matrix of distances as a numpy array
    distance_matrix = d_distance.values
        #Store the correct OD labelsfrom the dataframe
    O_labels = list(d_distance.index)
    D_labels = list(d_distance.columns)

   #Making data series into list form
    FW_available = d_feedstock["FW_available"].tolist()
    M_available = d_feedstock["M_available"].tolist()
    M_CH4_potential = d_feedstock["M_CH4_potential"].tolist() #manure methane potential (m^3/tonne)
    Dist_pipeline = d_feedstock["Dist_pipeline"].tolist()
    M_pot_out_plant = d_feedstock["M_pot_out_plant"].tolist()
   
    #TEST - header sum (should be expected_sum)
    expected_sum = 408848344
    try:
        O_labels_numeric = [float(x) for x in O_labels]
        D_labels_numeric = [float(x) for x in D_labels]
        O_sum = sum(O_labels_numeric)
        D_sum = sum(D_labels_numeric)

        if abs(O_sum - D_sum) > 1e-6:
            logger.warning(f"Sum of O-headers is: {O_sum}. Expected: {expected_sum}")

        else:
            logger.info("O-headers check passed")
        
        if abs(D_sum - expected_sum) > 1e-6:
            logger.warning(f"Sum of D-headers is: {D_sum}. Expected: {expected_sum}")

        else:
            logger.info("D-headers check passed")

    except Exception as e:
        logger.exception("Could not convert headers to numeric for checking:", e)


    #create "data" dictionary to pass to model
    data = {
        #Data per-cell
        "FW_available": FW_available,
        "M_available": M_available,
        "M_CH4_potential": M_CH4_potential, #manure methane potential (m^3/tonne)
        "Dist_pipeline": Dist_pipeline,
        "M_pot_out_plant": M_pot_out_plant,
        "Distance": distance_matrix,
        
        #Parameters/ factors
        "num_O_cells": num_O_cells,
        "num_suit_bg_cells": num_O_cells,
        "DM_FW": DM_FW,
        "VS_off_farm": VS_off_farm,
        "BGy_off_farm": BGy_off_farm,
        "CH4c_off_farm": CH4_cont_off_farm, #CH4 content off farm
        "OPEX_factor_AD": OPEX_factor_AD,
        "En_cont_CH4": En_cont_CH4,
        #"Project_life" : Project_life,
        "Amortization_factor": Amortization_factor,
        "MCE": MCE,
        "Amt_financed": Amt_financed,
        "C_pipe": C_pipe,
        "f": f,
        #"PV_FP_coeff": PV_FP_coeff,
        "FW_energy": FW_energy,
        "FW_transport_cost": FW_transport_cost,
        "O_labels": O_labels,
        "D_labels": D_labels
    }
    return data

#%%Defining Model

def create_model(data):
    """
    Building Pyomo Model
    V1: Minimize cost, enforcing minimum energy production, some fraction of the maximum 303,759,344.8.
    """
    
    m = ConcreteModel() #Model creation / defining the model

    #Set for origins and destinations (using 1-indexed positions)
    m.numO = data['num_O_cells']
    m.numD = data['num_suit_bg_cells'] 
    m.O = RangeSet(1, m.numO)
    m.D = RangeSet(1, m.numD)

    #Scalar parameters (need to move into the model)
    m.DM_FW = Param(initialize=data['DM_FW'])
    m.VS_off_farm = Param(initialize=data['VS_off_farm'])
    m.BGy_off_farm = Param(initialize=data['BGy_off_farm'])
    m.CH4c_off_farm = Param(initialize=data['CH4c_off_farm'])
    m.OPEX_factor_AD = Param(initialize=data['OPEX_factor_AD'])
    m.En_cont_CH4 = Param(initialize=data['En_cont_CH4'])
    m.Amortization_factor = Param(initialize=data['Amortization_factor'])
    m.MCE = Param(initialize=data['MCE'])
    m.Amt_financed = Param(initialize=data['Amt_financed'])
    m.C_pipe = Param(initialize=data['C_pipe'])
    m.f = Param(initialize=data['f'])

    #Indexed parameters (for each O or D)
    #initial manure available in the cell
    def m_avail_init(m_, i):
        return data['M_available'][i-1] #Need to shift (data for correct indexing (due to header row?))
    m.M_available = Param(m.O, initialize = m_avail_init) #initial available manure of origin cell (O)
    #initial food waste available in the cell
   
    def fw_avail_init(m_, i):
        return data['FW_available'][i-1]
    m.FW_available = Param(m.O, initialize = fw_avail_init)
    #Manure methane potential (volumetric) (m^3/tonne)
    def m_ch4_pot_init(m_, i):
        return data['M_CH4_potential'][i-1]
    m.M_CH4_potential = Param(m.O, initialize = m_ch4_pot_init)
    #distance to pipeline (from destination cell, D)
    def dist_pl_init(m_, j):
        return data['Dist_pipeline'][j-1]
    m.Dist_pipeline = Param(m.D, initialize = dist_pl_init)
    #Manure potential energy production (GJ/tonne)
    def m_en_pot_init(m_, i):
        return data['M_pot_out_plant'][i-1]
    m.M_pot_out_plant = Param(m.O, initialize = m_en_pot_init)
    #Distance between cells (origin-destination pairs)
    def dist_od_init(m_, i, j):
        return data['Distance'][i-1, j-1]
    m.Distance_OD = Param(m.O, m.D, initialize=dist_od_init)


    #FILTER OD PAIRS  - Eliminating unfeasible pairs to reduce model requirement
    feas_M_pairs = []
    feas_FW_pairs = []

    #Factors used for calculating the breakeven distance
    Cost_transp = data['FW_energy'] #$/km, representing cost of transporting food waste per km.
    En_cont_FW = data['FW_transport_cost'] #GJ/tonne of FW, so the energy content of food waste.
    rng_price = 60 #May need updated value? $/GJ

    feasible_fw_req = (En_cont_FW * rng_price) / Cost_transp
    for i in range(1, m.numO+1):
        #Manure check
        if data['M_available'][i-1] > 0:
            feasible_m_req = (data['M_pot_out_plant'][i-1] * rng_price) / Cost_transp
            for j in range(1, m.numD+1):
                if data['Distance'][i-1, j-1] <= feasible_m_req:
                    feas_M_pairs.append((i,j))

        #Foodwaste check
        if data['FW_available'][i-1] > 0:
            for j in range(1, m.numD+1):
                if data['Distance'][i-1,j-1] <= feasible_fw_req:
                    feas_FW_pairs.append((i,j))

    m.FEAS_M = Set(initialize=feas_M_pairs, dimen=2)
    m.FEAS_FW = Set(initialize=feas_FW_pairs, dimen=2)

    #DECISION VARIABLES
    m.M_moved   = Var(m.FEAS_M, within=NonNegativeReals)
    m.FW_moved  = Var(m.FEAS_FW, within=NonNegativeReals)
    m.Alpha     = Var(m.D, within=Binary)
    m.Cost      = Var()
    m.Energy    = Var(domain=NonNegativeReals)

#%% Defining Model Constraints

    def cost_eq_rule(m_):

        PlaceC = 150.3
        PlaceD = 466.3
        PlaceE = 2119947
        PlaceF = 0.15 #coefficient for transportation cost assoc. with manure
        PlaceG = 0.051 #coefficient for transportation cost assoc. with foodwaste
        PlaceH = 10.7 #coefficient derived from a formula used to calculate the present value of future payments.

        #Transport cost = manure transp cost + fw transp cost
        capex_manure = sum(PlaceC * m_.M_moved[i, j] for (i, j) in m_.FEAS_M) 
        capex_fw = sum(PlaceD * m_.FW_moved[i, j] for (i, j) in m_.FEAS_FW)
        capex_sum = (sum(PlaceE * m_.Alpha[j] for j in m_.D)
                        + capex_manure + capex_fw
                        + sum(m_.Dist_pipeline[j] * m_.C_pipe * m_.Alpha[j] for j in m_.D))
        
        unfinanced = capex_sum * (1 - m_.Amt_financed)
        financed = capex_sum * m_.Amortization_factor * m_.Amt_financed
        opex_part = capex_sum * m_.OPEX_factor_AD

        #Feedstock transport = manure transp + feadstock transp
        haul_manure = sum((PlaceF * m_.Distance_OD[i, j] + 0.5) * m_.M_moved[i, j]
                          for (i, j) in m_.FEAS_M)
        haul_fw = sum(PlaceG * m_.Distance_OD[i, j] * m_.FW_moved[i, j] 
                      for (i, j) in m_.FEAS_FW)
        feedstock_transport = haul_manure + haul_fw
        return  m_.Cost == unfinanced + PlaceH * (feedstock_transport + financed + opex_part)
    m.CostDefinition = Constraint(rule=cost_eq_rule)

    #FEEDSTOCK CONSTRAINTS
    def m_supply_rule(m_, i):
        feasible_Ds = [j for (ii, j) in m_.FEAS_M if ii==i]
        if not feasible_Ds:
            return Constraint.Skip
        return sum(m_.M_moved[i, j] for j in feasible_Ds) <= m_.M_available[i]
    m.ManureSupply = Constraint(m.O, rule = m_supply_rule)

    def fw_supply_rule(m_, i):
        feasible_Ds = [j for (ii, j) in m_.FEAS_FW if ii==i]
        if not feasible_Ds:
            return Constraint.Skip
        return sum(m_.FW_moved[i, j] for j in feasible_Ds) <= m_.FW_available[i]
    m.FWSupply = Constraint(m.O, rule = fw_supply_rule)


    #AD CAPACITY CONSTRAINTS
    def min_capacity_rule(m_, j):
        Min_Capacity = 30000 #tonnes of feedstock per year, capacity of digester

        m_in = sum(m_.M_moved[i, j] for (i,jj) in m_.FEAS_M if jj==j)
        fw_in = sum(m_.FW_moved[i, j] for (i, jj) in m_.FEAS_FW if jj==j)
        return m_in + fw_in >= m_.Alpha[j] * Min_Capacity
    m.MinCapacity = Constraint(m.D, rule=min_capacity_rule)

    def max_capacity_rule(m_, j):
        Max_Capacity = 120000 #tonnes of feedstock per year, capacity of digester

        m_in = sum(m_.M_moved[i, j] for (i,jj) in m_.FEAS_M if jj==j)
        fw_in = sum(m_.FW_moved[i, j] for (i, jj) in m_.FEAS_FW if jj==j)
        return m_in + fw_in <= m_.Alpha[j] * Max_Capacity
    m.MaxCapacity = Constraint(m.D, rule=max_capacity_rule)


    #ENERGY CONSTRAINTS
    #Energy production calculation
    def energy_eq_rule(m_):
        #10.7 coefficient derived from a formula used to calculate the present value of future payments.
        energy_m = sum(10.7*(m_.M_moved[i,j] * m_.M_pot_out_plant[i])
                              for (i,j) in m_.FEAS_M)
        energy_fw = sum(10.7 * m_.MCE * m_.En_cont_CH4 * 
                        (m_.FW_moved[i,j] * m_.DM_FW * m_.VS_off_farm * 
                         m_.BGy_off_farm * m_.CH4c_off_farm) * m_.Alpha[j] 
                        for (i, j) in m_.FEAS_FW)
        return m_.Energy == energy_m + energy_fw
    m.EnergyDefinition = Constraint(rule=energy_eq_rule)

    #Energy production constraint
    def min_energy_rule(m_):
        Max_Energy = 312426630.2 #From re-run maximum energy scenario
        return m_.Energy >= m_.f * Max_Energy
    m.MinEnergy = Constraint(rule=min_energy_rule)

#%% Objective

    m.Obj = Objective(expr = m.Cost, sense = minimize)
    return m
    #Minimizing system cost while meeting the constraints (above)

    """
    Solve the (Pyomo) model using CPLEX & write the results to an Excel file
    """
def solve_and_output(model, out_file = "Solution_cost_min.xlsx"):

    solver = SolverFactory('gurobi')


    solver.options['threads'] = 16 #Changed to processor count for my machine
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