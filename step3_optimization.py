import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import random
import functools
import pandas as pd
from deap import base, creator, tools, algorithms
import os
import time # Import time for benchmarking

# --- 1. Constants and Cost Definitions ---

# N_ENGINES will be loaded dynamically from data
PLANNING_HORIZON = 90  # planning horizon (days)
MAX_TEAMS = 5          # number of maintenance teams (resource constraint)

# Cost and time parameters (using user's latest values)
MAINTENANCE_COST = 20000   # scheduled maintenance cost
MAINTENANCE_TIME = 2       # scheduled maintenance time (days)
FAILURE_COST = 80000       # failure repair cost (higher)
FAILURE_TIME = 5           # failure repair time (longer)
DOWNTIME_COST_PER_DAY = 5000 # cost of downtime per day

# RUL range for simulation (after repair)
MIN_RUL_AFTER_REPAIR = 120 # min RUL for a *full* repair
MAX_RUL_CONFIG = 200     # default max RUL

# --- NSGA-II Genetic Algorithm Parameters ---
POP_SIZE = 50       # population size (e.g., 100-200)
NGEN = 50           # number of generations (e.g., 100-500)
CXPB = 0.8          # crossover probability
MUTPB = 0.2         # mutation probability (0.8 + 0.2 = 1.0)

# Optimization range for fuzzy thresholds
MIN_THRESHOLD = 5.0
MAX_THRESHOLD = 9.5

# --- 2. RUL Simulation (for post-repair) ---

def get_repaired_rul(repair_type):
    """
    Simulate RUL after maintenance, implementing partial restoration.
    (Implements Suggestion #2)
    """
    if repair_type == 'repair':
        # Failure repair: full restoration
        return np.random.uniform(MIN_RUL_AFTER_REPAIR, MAX_RUL_CONFIG)
    else: # 'maintenance'
        # Scheduled maintenance: partial restoration (e.g., 60-80% of full)
        partial_min = MIN_RUL_AFTER_REPAIR * 0.6  # e.g., 120 * 0.6 = 72
        partial_max = MAX_RUL_CONFIG * 0.8        # e.g., 200 * 0.8 = 160
        return np.random.uniform(partial_min, partial_max)


def create_urgency_lookup_table(max_rul_value):
    """
    *** SPEED OPTIMIZATION (NEW FUNCTION) ***
    Pre-computes all possible RUL/Importance combinations into a
    fast lookup table (LUT) to avoid expensive fuzzy logic calls
    during the simulation.
    """
    global MAX_RUL_CONFIG
    MAX_RUL_CONFIG = max_rul_value # update global config
    
    # 1. Define the Fuzzy System (same as before)
    rul_range = np.arange(0, max_rul_value + 1, 1)
    imp_range = np.arange(1, 4, 1) # Discrete values 1, 2, 3
    
    rul = ctrl.Antecedent(rul_range, 'rul')
    importance = ctrl.Antecedent(imp_range, 'importance')
    urgency = ctrl.Consequent(np.arange(0, 11, 1), 'urgency')

    rul['low'] = fuzz.trimf(rul.universe, [0, 0, max_rul_value * 0.4])
    rul['medium'] = fuzz.trimf(rul.universe, [max_rul_value * 0.2, max_rul_value * 0.5, max_rul_value * 0.8])
    rul['high'] = fuzz.trimf(rul.universe, [max_rul_value * 0.6, max_rul_value, max_rul_value])

    importance['low'] = fuzz.trimf(importance.universe, [1, 1, 2])
    importance['medium'] = fuzz.trimf(importance.universe, [1, 2, 3])
    importance['high'] = fuzz.trimf(importance.universe, [2, 3, 3])

    urgency['low'] = fuzz.trapmf(urgency.universe, [0, 0, 2, 4])
    urgency['medium'] = fuzz.trapmf(urgency.universe, [3, 4, 6, 7])
    urgency['high'] = fuzz.trapmf(urgency.universe, [6, 8, 10, 10])

    rule1 = ctrl.Rule(rul['high'], urgency['low'])
    rule2 = ctrl.Rule(rul['medium'] & importance['low'], urgency['low'])
    rule3 = ctrl.Rule(rul['medium'] & importance['medium'], urgency['medium'])
    rule4 = ctrl.Rule(rul['medium'] & importance['high'], urgency['high'])
    rule5 = ctrl.Rule(rul['low'] & importance['low'], urgency['medium'])
    rule6 = ctrl.Rule(rul['low'] & (importance['medium'] | importance['high']), urgency['high'])

    urgency_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
    urgency_simulator = ctrl.ControlSystemSimulation(urgency_ctrl)
    
    # 2. Create the Lookup Table (LUT)
    # We need indices from 0-200 for RUL and 0-3 for Importance
    # We'll make Importance 1-based, so array size is max_rul+1 by 4
    print("Pre-computing Fuzzy Logic Lookup Table (LUT)...")
    start_time = time.time()
    
    # Note: RUL is index, Importance is index (1, 2, 3)
    lut = np.zeros((max_rul_value + 1, 4)) 
    
    for r in range(max_rul_value + 1):
        for i in range(1, 4): # 1, 2, 3
            try:
                urgency_simulator.input['rul'] = r
                urgency_simulator.input['importance'] = i
                urgency_simulator.compute()
                lut[r, i] = urgency_simulator.output['urgency']
            except:
                lut[r, i] = 5.0 # Fallback
    
    print(f"LUT pre-computation finished in {time.time() - start_time:.2f} seconds.")
    return lut

def get_urgency_score_from_lut(lut, current_rul, engine_importance):
    """ 
    *** SPEED OPTIMIZATION (NEW FUNCTION) ***
    Gets the urgency score from the pre-computed Lookup Table.
    This is an extremely fast array lookup.
    """
    if current_rul <= 0:
        return 10.0 # if RUL <=0, max urgency
    
    # Convert RUL to integer index and clip
    idx_rul = int(np.clip(current_rul, 0, MAX_RUL_CONFIG))
    
    # Importance is already 1, 2, or 3, which are valid indices
    return lut[idx_rul, engine_importance]


def evaluate_schedule(individual_thresholds, n_engines, initial_ruls_data, urgency_lut, engine_importance_data):
    """
    DEAP evaluation function: Simulates the strategy using the
    fast Lookup Table (urgency_lut).
    """
    
    # *** OPTIMIZATION: No simulator object needed here! ***
    
    # Simulation state
    engine_ruls = initial_ruls_data.copy()
    engine_status = ['operational'] * n_engines 
    days_in_shop = np.zeros(n_engines)
    
    # Objective variables
    total_cost = 0
    total_downtime = 0 # Still needed to calculate downtime_cost
    total_failures = 0 # *** NEW OBJECTIVE (Suggestion #1) ***

    for day in range(PLANNING_HORIZON):
        
        current_teams_used = 0
        service_candidates = []
        engines_down_this_day = np.zeros(n_engines, dtype=bool)

        # --- Loop 1: Update engines in shop, find candidates from operational engines ---
        for i in range(n_engines):
            status = engine_status[i]
            
            if 'maintenance' in status or 'repair' in status:
                engines_down_this_day[i] = True 
                if 'awaiting' not in status:
                    current_teams_used += 1
                    days_in_shop[i] += 1
                    
                    if status == 'maintenance' and days_in_shop[i] >= MAINTENANCE_TIME:
                        engine_status[i] = 'operational'
                        days_in_shop[i] = 0
                        engine_ruls[i] = get_repaired_rul('maintenance')
                    elif status == 'repair' and days_in_shop[i] >= FAILURE_TIME:
                        engine_status[i] = 'operational'
                        days_in_shop[i] = 0
                        engine_ruls[i] = get_repaired_rul('repair')
                
                if status == 'awaiting_repair':
                    service_candidates.append((100.0, i, 'repair'))
                elif status == 'awaiting_maintenance':
                    rul = engine_ruls[i]
                    importance = engine_importance_data[i]
                    # *** OPTIMIZATION: Use fast LUT lookup ***
                    urgency = get_urgency_score_from_lut(urgency_lut, rul, importance)
                    service_candidates.append((urgency, i, 'maintenance'))

            elif status == 'operational':
                rul = engine_ruls[i]
                importance = engine_importance_data[i]
                
                if rul <= 0:
                    total_failures += 1
                    service_candidates.append((100.0, i, 'repair'))
                    engine_status[i] = 'awaiting_repair'
                    engines_down_this_day[i] = True
                else:
                    # *** OPTIMIZATION: Use fast LUT lookup ***
                    urgency = get_urgency_score_from_lut(urgency_lut, rul, importance)
                    if urgency > individual_thresholds[i]:
                        service_candidates.append((urgency, i, 'maintenance'))
                        engine_status[i] = 'awaiting_maintenance'
                        engines_down_this_day[i] = True
            
        # --- Loop 2: Assign available teams ---
        available_teams = MAX_TEAMS - current_teams_used
        sorted_candidates = sorted(service_candidates, key=lambda x: x[0], reverse=True)
        
        for (priority, i, service_type) in sorted_candidates:
            if available_teams > 0 and 'awaiting' in engine_status[i]:
                available_teams -= 1
                engine_status[i] = service_type 
                days_in_shop[i] = 1
                
                if service_type == 'repair':
                    total_cost += FAILURE_COST
                elif service_type == 'maintenance':
                    total_cost += MAINTENANCE_COST
            
        # --- Loop 3: Update RUL and calculate total downtime for the day ---
        for i in range(n_engines):
            if engine_status[i] == 'operational':
                engine_ruls[i] -= 1
            
            if engines_down_this_day[i]:
                total_downtime += 1

    total_cost += total_downtime * DOWNTIME_COST_PER_DAY
    
    return total_cost, total_failures


# --- 5. Main Execution ---

if __name__ == "__main__":

    # 1. reading data test_processed.csv and define initial RULs and importance
    print("reading data test_processed.csv")
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        test_csv_path = os.path.join(base_dir, 'output/test_processed.csv')
        df = pd.read_csv(test_csv_path)
    except FileNotFoundError:
        print(f"error, no such file '{test_csv_path}'")
        exit()

    # dataframe columns auto-detect
    engine_id_candidates = ['engine_id', 'engine', 'unit', 'id', 'ID']
    rul_candidates = ['RUL', 'rul', 'RemainingUsefulLife', 'remaining_useful_life', 'remaining']
    engine_col = next((c for c in engine_id_candidates if c in df.columns), None)
    rul_col = next((c for c in rul_candidates if c in df.columns), None)

    if engine_col is None or rul_col is None:
        print("error: there is no 'engine_id' or 'RUL' column.")
        exit()

    # getting intial RULs for each engine (for simulation)
    grouped = df.groupby(engine_col)[rul_col]
    initial_ruls = grouped.last().values

    N_ENGINES = len(initial_ruls)
    if N_ENGINES == 0:
        print("error: no engine data loaded from CSV.")
        exit()
    
    n_sample = min(60, N_ENGINES) # Sample 60 engines or fewer
    print(f"Total engines found: {N_ENGINES}. Sampling {n_sample} for demo.")
    sampled_indices = random.sample(range(N_ENGINES), n_sample)
    initial_ruls = initial_ruls[sampled_indices]
    N_ENGINES = n_sample 

    # Dynamically create importance groups based on n_sample
    n_low = n_sample // 3
    n_med = n_sample // 3
    n_high = n_sample - n_low - n_med 
    
    importance_low = np.full(n_low, 1)      # low importance
    importance_medium = np.full(n_med, 2)   # medium importance
    importance_high = np.full(n_high, 3)     # high importance
    simulated_engine_importance = np.concatenate([importance_low, importance_medium, importance_high])
    np.random.shuffle(simulated_engine_importance)
    print(f"Created simulated 'Task Importance' (1, 2, 3) for {N_ENGINES} engines.")

    data_max_rul = 200 
    print(f"Finished loading initial RUL data for {N_ENGINES} engines.")
    print(f"Average *initial* RUL (at simulation start): {np.mean(initial_ruls):.2f} days")
    print(f"Fuzzy system MAX_RUL set to: {data_max_rul} days")

    # 2. Define fuzzy logic system
    # *** OPTIMIZATION: Create the Lookup Table (LUT) ONCE ***
    start_main = time.time()
    urgency_lut = create_urgency_lookup_table(data_max_rul)
    print("Fuzzy logic LUT created.")

    # 3. Define DEAP genetic algorithm components
    print(f"\n--- defining NSGA-II genetic algorithm ---")
    print(f"Pop_size: {POP_SIZE}, N_gen: {NGEN}, N_engines: {N_ENGINES}")

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_thresh", random.uniform, MIN_THRESHOLD, MAX_THRESHOLD)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_thresh, N_ENGINES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register evaluation function (*** key update: pass the LUT ***)
    toolbox.register("evaluate", evaluate_schedule,
                     n_engines=N_ENGINES,
                     initial_ruls_data=initial_ruls, 
                     urgency_lut=urgency_lut,
                     engine_importance_data=simulated_engine_importance)

    toolbox.register("mate", tools.cxBlend, alpha=0.5) 
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.1)
    toolbox.register("select", tools.selNSGA2) 

    # 4. Run baseline strategy (Run-to-Failure)
    print("\n--- evaluating [baseline: Run-to-Failure] strategy ---")
    r2f_thresholds = [11.0] * N_ENGINES
    r2f_cost, r2f_failures = toolbox.evaluate(r2f_thresholds)

    print(f"Total cost: ${r2f_cost:,.0f}")
    print(f"Total failures: {r2f_failures:,.0f} events")


    # 5. Run NSGA-II optimization
    print("\n--- running [optimization: NSGA-II + Fuzzy Policy] strategy ---")
    print("This may take a few minutes...")
    
    pop = toolbox.population(n=POP_SIZE)
    algorithms.eaMuPlusLambda(pop, toolbox, mu=POP_SIZE, lambda_=POP_SIZE, 
                              cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, verbose=False)

    print("NSGA-II optimization completed.")
    print(f"\n--- Total Optimization Time: {time.time() - start_main:.2f} seconds ---")

    # 6. Results
    print("\n--- Optimization Results (Pareto Front) ---")
    pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

    print(f"Baseline (R2F):   Cost=${r2f_cost:,.0f}, Failures={r2f_failures:,.0f}")
    print("---")
    print(f"NSGA-II found {len(pareto_front)} Pareto optimal solutions:")
    
    for i, sol in enumerate(pareto_front):
        cost, failures = sol.fitness.values
        # Note: Corrected typo from Faililures to Failures
        print(f"  Solution {i+1}: Cost=${cost:,.0f}, Failures={failures:,.0f}") 

    # Find the lowest cost solution
    best_cost_sol = min(pareto_front, key=lambda sol: sol.fitness.values[0])
    cost, failures = best_cost_sol.fitness.values
    
    print("\n---")
    print(f"Example Selection (Lowest Cost Solution):")
    print(f"  Cost: ${cost:,.0f} (Savings: ${r2f_cost - cost:,.0f})")
    print(f"  Failures: {failures:,.0f} (Reduction: {r2f_failures - failures:,.0f} events)")

    # *** NEW ANALYSIS (Suggestion #5) ***
    print("\n--- Analysis of Best Policy (Lowest Cost Solution) ---")
    best_policy_thresholds = np.array(best_cost_sol)
    
    # Calculate average thresholds for each importance group
    # Add a small check in case a group has 0 engines (due to sampling)
    avg_thresh_low_imp = best_policy_thresholds[simulated_engine_importance == 1].mean() if (simulated_engine_importance == 1).any() else -1
    avg_thresh_med_imp = best_policy_thresholds[simulated_engine_importance == 2].mean() if (simulated_engine_importance == 2).any() else -1
    avg_thresh_high_imp = best_policy_thresholds[simulated_engine_importance == 3].mean() if (simulated_engine_importance == 3).any() else -1

    print(f"  Overall average threshold: {np.mean(best_policy_thresholds):.2f}")
    print(f"  Avg threshold for LOW importance (1) engines: {avg_thresh_low_imp:.2f}")
    print(f"  Avg threshold for MED importance (2) engines: {avg_thresh_med_imp:.2f}")
    print(f"  Avg threshold for HIGH importance (3) engines: {avg_thresh_high_imp:.2f}")
    
    # Final conclusion
    if (r2f_cost - cost) > 0 or (r2f_failures - failures) > 0:
        print("\nConclusion: NSGA-II strategy successfully found Pareto solutions that outperform the baseline!")
    else:
        print("\nConclusion: NSGA-II did not find solutions that significantly outperform the baseline.")
        print("Tip: Consider increasing POP_SIZE and NGEN for a more thorough search.")