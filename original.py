import pandas as pd
import numpy as np
from pyomo.environ import (
    ConcreteModel, RangeSet, Param, Var, NonNegativeReals, Binary,
    Constraint, Objective, minimize, Set, value, SolverFactory, TerminationCondition
)

# def load_data(excel_file="2021_Paper1_Data_File_Updated.xlsx"):
def load_data(excel_file="data_class_project.xlsx"):
    """
    Reads input data from Excel.
    For the OD sheet, we read it with index_col=0 so that the row indices (origins)
    and column headers (destinations) are preserved. We also check that the numeric 
    headers sum to 408848344.
    """
    number_of_origin_cells = 3996
    number_of_suitable_biogas_cells = 3996

    # Other technical parameters
    DM_Content_FW = 0.25
    VS_off_farm = 0.9
    Biogas_yield_off_farm = 500
    Methane_content_off_farm = 0.6
    OPEX_factor_AD = 0.07
    Energy_content_CH4 = 0.037
    Amortization_factor = 0.0872
    MCE = 0.85
    Amount_Financed = 0.75
    C_Pipe = 1208194
    f = 1.0   # For the energy constraint, set f = 1 for now

    # Prompt the user for the RNG Price (used in breakeven calculations)
    rng_price = float(input("Please enter the RNG Price: "))

    # Read the "All_Feedstock" sheet and include the extra column
    df_feedstock = pd.read_excel(
        excel_file,
        sheet_name="All_Feedstock",
        header=0,
        usecols=[
            "Uncaptured FW (tonnes)",
            "Manure (tonnes)",
            "Methane Potential (m3/tonne)",
            "Distance to pipeline (km)",
            "Potential Plant Output Manure (GJ/tonne)"
        ],
        nrows=number_of_origin_cells
    )

    df_feedstock.rename(
        columns={
            "Uncaptured FW (tonnes)": "Food_Waste_available",
            "Manure (tonnes)": "Manure_Available",
            "Methane Potential (m3/tonne)": "manure_methane_potential",
            "Distance to pipeline (km)": "Dist_Pipeline",
            "Potential Plant Output Manure (GJ/tonne)": "Potential_Plant_Output_Manure"
        },
        inplace=True
    )

    food_waste_available = df_feedstock["Food_Waste_available"].tolist()
    manure_available = df_feedstock["Manure_Available"].tolist()
    manure_methane_potential = df_feedstock["manure_methane_potential"].tolist()
    dist_pipeline = df_feedstock["Dist_Pipeline"].tolist()
    potential_plant_output_manure = df_feedstock["Potential_Plant_Output_Manure"].tolist()

    # Read the OD distance matrix from the "OD" sheet.
    # Use index_col=0 so that the origin labels come from the first column.
    df_distance = pd.read_excel(
        excel_file,
        sheet_name="OD",
        header=0,
        index_col=0,
        nrows=number_of_origin_cells
    )
    # Extract the distance matrix as a numpy array
    distance_matrix = df_distance.values
    # Store the correct origin and destination labels from the dataframe
    origin_labels = list(df_distance.index)
    destination_labels = list(df_distance.columns)

    # --- Check header sums ---
    expected_sum = 408848344
    try:
        # Attempt to convert headers to float for a numeric sum check
        origin_labels_numeric = [float(x) for x in origin_labels]
        destination_labels_numeric = [float(x) for x in destination_labels]
        origin_sum = sum(origin_labels_numeric)
        destination_sum = sum(destination_labels_numeric)
        if abs(origin_sum - expected_sum) > 1e-6:
            print("Warning: Sum of origin headers is", origin_sum, "but expected", expected_sum)
        else:
            print("Origin headers check passed.")
        if abs(destination_sum - expected_sum) > 1e-6:
            print("Warning: Sum of destination headers is", destination_sum, "but expected", expected_sum)
        else:
            print("Destination headers check passed.")
    except Exception as e:
        print("Could not convert headers to numeric for checking:", e)
    
    data = {
        "number_of_origin_cells": number_of_origin_cells,
        "number_of_suitable_biogas_cells": number_of_suitable_biogas_cells,
        "DM_Content_FW": DM_Content_FW,
        "VS_off_farm": VS_off_farm,
        "Biogas_yield_off_farm": Biogas_yield_off_farm,
        "Methane_content_off_farm": Methane_content_off_farm,
        "OPEX_factor_AD": OPEX_factor_AD,
        "Energy_content_CH4": Energy_content_CH4,
        "Amortization_factor": Amortization_factor,
        "MCE": MCE,
        "Amount_Financed": Amount_Financed,
        "C_Pipe": C_Pipe,
        "f": f,
        "rng_price": rng_price,
        "Manure_Available": manure_available,
        "Food_Waste_available": food_waste_available,
        "manure_methane_potential": manure_methane_potential,
        "Dist_Pipeline": dist_pipeline,
        "Potential_Plant_Output_Manure": potential_plant_output_manure,
        "Distance": distance_matrix,
        "origin_labels": origin_labels,
        "destination_labels": destination_labels
    }

    return data

def create_biogas_model(data):
    """
    Builds the Pyomo model.
    This version minimizes cost while enforcing that the produced energy is at least
    f * 303759344.8. (For now, f=1.)
    The cost is computed via an equality constraint.
    """
    m = ConcreteModel()

    # Sets for origins and destinations (using 1-indexed positions)
    m.numO = data['number_of_origin_cells']
    m.numD = data['number_of_suitable_biogas_cells']
    m.O = RangeSet(1, m.numO)
    m.D = RangeSet(1, m.numD)

    # Scalar Parameters
    m.DM_Content_FW = Param(initialize=data['DM_Content_FW'])
    m.VS_off_farm   = Param(initialize=data['VS_off_farm'])
    m.Biogas_yield_off_farm = Param(initialize=data['Biogas_yield_off_farm'])
    m.Methane_content_off_farm = Param(initialize=data['Methane_content_off_farm'])
    m.OPEX_factor_AD = Param(initialize=data['OPEX_factor_AD'])
    m.Energy_content_CH4 = Param(initialize=data['Energy_content_CH4'])
    m.Amortization_factor = Param(initialize=data['Amortization_factor'])
    m.MCE = Param(initialize=data['MCE'])
    m.Amount_Financed = Param(initialize=data['Amount_Financed'])
    m.C_Pipe = Param(initialize=data['C_Pipe'])
    m.f = Param(initialize=data['f'])
    # Note: rng_price is used only for filtering

    # Indexed Parameters for each origin or destination
    def manure_avail_init(m_, i):
        return data['Manure_Available'][i-1]
    m.Manure_Available = Param(m.O, initialize=manure_avail_init)

    def fw_init(m_, i):
        return data['Food_Waste_available'][i-1]
    m.Food_Waste_available = Param(m.O, initialize=fw_init)

    def methane_pot_init(m_, i):
        return data['manure_methane_potential'][i-1]
    m.manure_methane_potential = Param(m.O, initialize=methane_pot_init)

    def dist_pipeline_init(m_, j):
        return data['Dist_Pipeline'][j-1]
    m.Dist_Pipeline = Param(m.D, initialize=dist_pipeline_init)

    def dist_init(m_, i, j):
        return data['Distance'][i-1, j-1]
    m.Distance = Param(m.O, m.D, initialize=dist_init)

    def potential_output_init(m_, i):
        return data['Potential_Plant_Output_Manure'][i-1]
    m.Potential_Plant_Output_Manure = Param(m.O, initialize=potential_output_init)

    # -------------------------------------------------------------------------
    # Build filtering sets for (origin, destination) pairs
    # -------------------------------------------------------------------------
    feasible_manure_pairs = []
    feasible_fw_pairs = []
    for i in range(1, m.numO+1):
        # For manure:
        if data['Manure_Available'][i-1] > 0:
            breakeven_manure = (data['Potential_Plant_Output_Manure'][i-1] * data['rng_price']) / 1.098695892
            for j in range(1, m.numD+1):
                if data['Distance'][i-1, j-1] <= breakeven_manure:
                    feasible_manure_pairs.append((i, j))
        # For food waste:
        if data['Food_Waste_available'][i-1] > 0:
            breakeven_fw = (2.122875 * data['rng_price']) / 1.098695892
            for j in range(1, m.numD+1):
                if data['Distance'][i-1, j-1] <= breakeven_fw:
                    feasible_fw_pairs.append((i, j))

    m.FEASIBLE_MANURE = Set(initialize=feasible_manure_pairs, dimen=2)
    m.FEASIBLE_FW = Set(initialize=feasible_fw_pairs, dimen=2)

    # -------------------------------------------------------------------------
    # Decision Variables:
    # -------------------------------------------------------------------------
    m.Manure_moved = Var(m.FEASIBLE_MANURE, domain=NonNegativeReals)
    m.FW_moved     = Var(m.FEASIBLE_FW, domain=NonNegativeReals)
    m.Alpha        = Var(m.D, domain=Binary)
    m.Cost         = Var()
    m.Energy       = Var(domain=NonNegativeReals)

    # -------------------------------------------------------------------------
    # Constraints:
    # -------------------------------------------------------------------------
    def cost_eq_rule(m_):
        transport_cost_manure = sum(150.3 * m_.Manure_moved[i,j] for (i,j) in m_.FEASIBLE_MANURE)
        transport_cost_fw = sum(466.3 * m_.FW_moved[i,j] for (i,j) in m_.FEASIBLE_FW)
        transport_cost = transport_cost_manure + transport_cost_fw

        capex_sum = ( sum(2119947 * m_.Alpha[j] for j in m_.D)
                     + transport_cost
                     + sum(m_.Dist_Pipeline[j] * m_.C_Pipe * m_.Alpha[j] for j in m_.D) )
        portion_unfinanced = capex_sum * (1 - m_.Amount_Financed)
        financed_part = capex_sum * m_.Amortization_factor * m_.Amount_Financed
        opex_part = capex_sum * m_.OPEX_factor_AD

        feedstock_transport_manure = sum((0.15*m_.Distance[i,j] + 0.5) * m_.Manure_moved[i,j] for (i,j) in m_.FEASIBLE_MANURE)
        feedstock_transport_fw = sum(0.051 * m_.Distance[i,j] * m_.FW_moved[i,j] for (i,j) in m_.FEASIBLE_FW)
        feedstock_transport = feedstock_transport_manure + feedstock_transport_fw

        return m_.Cost == portion_unfinanced + 10.7*(feedstock_transport + financed_part + opex_part)
    m.CostDefinition = Constraint(rule=cost_eq_rule)

    def manure_supply_rule(m_, i):
        feasible_dests = [j for (ii, j) in m_.FEASIBLE_MANURE if ii == i]
        if not feasible_dests:
            return Constraint.Skip
        return sum(m_.Manure_moved[i, j] for j in feasible_dests) <= m_.Manure_Available[i]
    m.ManureSupply = Constraint(m.O, rule=manure_supply_rule)

    def fw_supply_rule(m_, i):
        feasible_dests = [j for (ii, j) in m_.FEASIBLE_FW if ii == i]
        if not feasible_dests:
            return Constraint.Skip
        return sum(m_.FW_moved[i, j] for j in feasible_dests) <= m_.Food_Waste_available[i]
    m.FWSupply = Constraint(m.O, rule=fw_supply_rule)

    def capacity_upper_rule(m_, j):
        manure_in = sum(m_.Manure_moved[i,j] for (i,jj) in m_.FEASIBLE_MANURE if jj == j)
        fw_in = sum(m_.FW_moved[i,j] for (i,jj) in m_.FEASIBLE_FW if jj == j)
        return manure_in + fw_in <= m_.Alpha[j] * 1000000
    m.CapacityUpper = Constraint(m.D, rule=capacity_upper_rule)

    def capacity_lower_rule(m_, j):
        manure_in = sum(m_.Manure_moved[i,j] for (i,jj) in m_.FEASIBLE_MANURE if jj == j)
        fw_in = sum(m_.FW_moved[i,j] for (i,jj) in m_.FEASIBLE_FW if jj == j)
        return manure_in + fw_in >= m_.Alpha[j] * 30000
    m.CapacityLower = Constraint(m.D, rule=capacity_lower_rule)

    def energy_def_rule(m_):
        energy_manure = sum(10.7 * m_.MCE * m_.Energy_content_CH4 *
                              (m_.Manure_moved[i,j] * m_.manure_methane_potential[i])
                              for (i,j) in m_.FEASIBLE_MANURE)
        energy_fw = sum(10.7 * m_.MCE * m_.Energy_content_CH4 *
                        (m_.FW_moved[i,j] * m_.DM_Content_FW * m_.VS_off_farm *
                         m_.Biogas_yield_off_farm * m_.Methane_content_off_farm)
                        for (i,j) in m_.FEASIBLE_FW)
        return m_.Energy == energy_manure + energy_fw
    m.EnergyDefinition = Constraint(rule=energy_def_rule)

    # -------------------------------------------------------------------------
    # New Energy Constraint:
    # Ensure that the produced energy is at least f * 303759344.8
    # -------------------------------------------------------------------------
    def min_energy_rule(m_):
        return m_.Energy >= m_.f * 303759344.8
    m.MinEnergy = Constraint(rule=min_energy_rule)

    # -------------------------------------------------------------------------
    # Objective: Minimize total cost.
    # -------------------------------------------------------------------------
    m.Obj = Objective(expr=m.Cost, sense=minimize)

    return m

def solve_and_write_solution(model, out_file="Solution_File_cost_min.xlsx"):
    """
    Solve the Pyomo model using CPLEX and write results to an Excel file.
    For the flow sheets, only origin–destination pairs with flow > 0 are output.
    The origin and destination labels (from the OD sheet) are used.
    """
    #solver_path = r"C:\Program Files\IBM\ILOG\CPLEX_Studio2211\cplex\bin\x64_win64\cplex.exe"
    #solver = SolverFactory('cplex', executable=solver_path)

    solver = SolverFactory('gurobi')

    # Set solver options (adjust if necessary)
    # solver.options['threads'] = 24
    # solver.options['mip_tolerances_mipgap'] = 0.075
    # solver.options['mip_display'] = 5
    # solver.options['preprocessing_symmetry'] = 0
    # solver.options['mip_strategy_search'] = 2
    # solver.options['mip_strategy_file'] = 3
    
    solver.options['threads']= 4
    solver.options['MIPGap']= 0.075
    solver.options['OutputFlag']= 1
    solver.options['Symmetry']= 0
    solver.options['NodefileStart']= 0.0

    results = solver.solve(model, tee=True)

    print("Solver Status:", results.solver.status)
    print("Termination Condition:", results.solver.termination_condition)

    feasible_conditions = {
        TerminationCondition.optimal,
        TerminationCondition.feasible,
        TerminationCondition.locallyOptimal,
        TerminationCondition.globallyOptimal
    }

    if (results.solver.status == 'ok' and 
        results.solver.termination_condition in feasible_conditions):

        # Only output flows where the value is greater than 0.
        manure_solution = [
            (model.origin_labels[i-1], model.destination_labels[j-1], value(model.Manure_moved[i,j]))
            for (i,j) in model.FEASIBLE_MANURE if value(model.Manure_moved[i,j]) > 0
        ]
        fw_solution = [
            (model.origin_labels[i-1], model.destination_labels[j-1], value(model.FW_moved[i,j]))
            for (i,j) in model.FEASIBLE_FW if value(model.FW_moved[i,j]) > 0
        ]

        # Extract cost and energy values
        cost_val = value(model.Cost)
        energy_val = value(model.Energy)

        # Write solutions to Excel
        with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
            df_manure = pd.DataFrame(manure_solution, columns=['Origin', 'Destination', 'Manure_moved'])
            df_manure.to_excel(writer, sheet_name='ManureFlows', index=False)

            df_fw = pd.DataFrame(fw_solution, columns=['Origin', 'Destination', 'FW_moved'])
            df_fw.to_excel(writer, sheet_name='FoodWasteFlows', index=False)

            df_ce = pd.DataFrame({'Value': [cost_val, energy_val]},
                                 index=['Cost', 'Energy'])
            df_ce.to_excel(writer, sheet_name='Cost_Energy', index=True)

        print(f"Solution written to {out_file}")
    else:
        print("No feasible (or optimal) solution found. Solver status:")
        print("  Status =", results.solver.status)
        print("  Termination =", results.solver.termination_condition)
        print("No solution file will be written.")

if __name__ == "__main__":
    # data = load_data("2021_Paper1_Data_File_Updated.xlsx")
    data = load_data("data_class_project.xlsx")
    model = create_biogas_model(data)
    # Attach the OD sheet labels to the model for later use in output
    model.origin_labels = data["origin_labels"]
    model.destination_labels = data["destination_labels"]
    solve_and_write_solution(model, "Solution_File_cost_min_f100.xlsx")