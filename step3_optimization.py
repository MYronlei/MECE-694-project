import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import random
import functools
import pandas as pd
from deap import base, creator, tools, algorithms
import os
import time 
import matplotlib.pyplot as plt
    

#  1. Constants and Cost Definitions

# N_ENGINES will be loaded dynamically from data
PLANNING_HORIZON = 120  # planning horizon (shifts) - increased for better strategy differentiation
MAX_TEAMS = 5           # number of maintenance teams (resource constraint)

# Cost and time parameters
MAINTENANCE_COST = 30000   # scheduled maintenance cost
MAINTENANCE_TIME = 4       # scheduled maintenance time (shifts)
FAILURE_COST = 80000       # failure repair cost (higher)
FAILURE_TIME = 10           # failure repair time (longer)
LOW_DOWNTIME_COST_PER_SHIFT = 3000 # cost of downtime per shift for low importance engines
MEDIUM_DOWNTIME_COST_PER_SHIFT = 4000 # cost of downtime per shift for medium importance engines
HIGH_DOWNTIME_COST_PER_SHIFT = 5000 # cost of downtime per shift for high importance engines

# RUL range for simulation (after repair)
MIN_RUL_AFTER_REPAIR = 250 # min RUL for a *full* repair (in cycles)
MAX_RUL_CONFIG = 350     # default max RUL (in cycles)

# RUL failure threshold: when RUL drops below this, engine MUST be repaired (cannot run anymore)
RUL_FAILURE_THRESHOLD = 50  # cycles (if RUL <= this value, force repair)

# Simulation time step configuration
CYCLES_PER_SHIFT = 2  # number of cycles consumed per shift (adjustable)

# --- NSGA-II Genetic Algorithm Parameters ---
POP_SIZE = 50       # population size
NGEN = 50           # number of generations 
CXPB = 0.8          # crossover probability 
MUTPB = 0.2         # mutation probability

# Optimization range for fuzzy thresholds - expanded for more diversity
MIN_THRESHOLD = 1.0  # allow very aggressive maintenance strategies
MAX_THRESHOLD = 9.5 # allow very conservative strategies (near failure)

#%% 2. RUL Simulation (for post-repair) 

def get_repaired_rul(repair_type):
    """
    Simulate RUL after maintenance, implementing partial restoration.
    (Implements Suggestion #2)
    Returns RUL in cycles.
    """
    if repair_type == 'repair':
        # Simulate repair: partial restoration (e.g., 70% of full)
        partial_min = MIN_RUL_AFTER_REPAIR * 0.9
        partial_max = MAX_RUL_CONFIG * 0.7        
        return np.random.uniform(partial_min, partial_max)
    else: # 'maintenance'
        # Scheduled maintenance: partial restoration (e.g., 80% of full)
        partial_min = MIN_RUL_AFTER_REPAIR   
        partial_max = MAX_RUL_CONFIG * 0.8        
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
    
    # Define the Fuzzy System
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
    
    # Create the Lookup Table (LUT)
    # We need indices from 0-350 for RUL and 0-3 for Importance
    # We'll make Importance 1-based, so array size is max_rul+1 by 4
    print("Pre-computing Fuzzy Logic Lookup Table (LUT)...")
    start_time = time.time()
    
    # RUL is index, Importance is index (1, 2, 3)
    lut = np.zeros((max_rul_value + 1, 4)) 
    
    for r in range(max_rul_value + 1):
        for i in range(1, 4): # 1, 2, 3
            urgency_simulator.input['rul'] = r
            urgency_simulator.input['importance'] = i
            urgency_simulator.compute()
            lut[r, i] = urgency_simulator.output['urgency']
    
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

def evaluate_by_importance(individual, n_engines, initial_ruls_data, urgency_lut, engine_importance_data):
    """
    Evaluate an individual by mapping its three thresholds to engines based on importance.
    # individual[0] -> threshold for importance==1 (low)
    # individual[1] -> threshold for importance==2 (medium)
    # individual[2] -> threshold for importance==3 (high)
    """
    thr_map = {
        1: np.clip(float(individual[0]), MIN_THRESHOLD, MAX_THRESHOLD),
        2: np.clip(float(individual[1]), MIN_THRESHOLD, MAX_THRESHOLD),
        3: np.clip(float(individual[2]), MIN_THRESHOLD, MAX_THRESHOLD)
    }
    per_engine_thresh = [thr_map[int(imp)] for imp in engine_importance_data] # assign thresholds based on importance to each engine
    return evaluate_schedule(per_engine_thresh, n_engines, initial_ruls_data, urgency_lut, engine_importance_data)

def evaluate_schedule(individual_thresholds, n_engines, initial_ruls_data, urgency_lut, engine_importance_data):
    """
    This is the system dynamic model.

    Status space:
        - 'operational'             : engine is running normally
        - 'awaiting_maintenance'    : maintenance has been scheduled/queued, engine can continue running
        - 'maintenance'             : maintenance is in progress (team assigned, engine down)
        - 'awaiting_repair'         : repair has been scheduled/queued, engine down
        - 'repair'                  : repair is in progress (team assigned, engine down)

    Logic:
        - RUL_FAILURE_THRESHOLD: if RUL <= this value, engine must be repaired and can't run anymore
        - Maintenance THRESHOLD: if RUL <= this value, engine must be maintained (but can still run)
        - awaiting_maintenance: engine continues to run (RUL decreases) until:
            1. RUL hits failure threshold -> escalate to awaiting_repair
            2. Team becomes available -> start maintenance

    Target: Find best individual maintenance thresholds for engines that running different important missions that can:
    - Minimize total cost (maintenance + failure + downtime)
    - Minimize total failures
    """

    # Simulation state
    engine_ruls = initial_ruls_data.copy()
    engine_status = ['operational'] * n_engines
    shifts_in_shop = np.zeros(n_engines)

    # Objective variables
    total_cost = 0
    total_failures = 0
    # We'll accumulate downtime cost directly per-engine per-shift using importance-specific rates
    total_downtime_cost = 0

    for shift in range(PLANNING_HORIZON):
        current_teams_used = 0
        service_candidates = []
        engines_down_this_shift = np.zeros(n_engines, dtype=bool)

        # Loop 1: Update engines in shop, find candidates
        for i in range(n_engines):
            status = engine_status[i]

            # If in maintenance or repair, update days in shop, record downtime
            if status == 'maintenance' or status == 'repair':
                engines_down_this_shift[i] = True
                current_teams_used += 1
                shifts_in_shop[i] += 1
                # Check if maintenance is complete (measured in shifts)
                if status == 'maintenance' and shifts_in_shop[i] >= MAINTENANCE_TIME:
                    engine_status[i] = 'operational'
                    shifts_in_shop[i] = 0
                    engine_ruls[i] = get_repaired_rul('maintenance')
                # Check if repair is complete (measured in shifts)
                elif status == 'repair' and shifts_in_shop[i] >= FAILURE_TIME:
                    engine_status[i] = 'operational'
                    shifts_in_shop[i] = 0
                    engine_ruls[i] = get_repaired_rul('repair')

            # If awaiting repair, engine is down, must be repaired right away
            elif status == 'awaiting_repair':
                engines_down_this_shift[i] = True
                service_candidates.append((100.0, i, 'repair'))

            # If awaiting maintenance, engine can still run
            elif status == 'awaiting_maintenance':
                rul = engine_ruls[i]
                importance = engine_importance_data[i]

                # Check if RUL hit failure threshold while waiting
                if rul <= RUL_FAILURE_THRESHOLD:
                    # Escalate to repair
                    total_failures += 1
                    engine_status[i] = 'awaiting_repair'
                    engines_down_this_shift[i] = True
                    service_candidates.append((100.0, i, 'repair'))
                else:
                    # Still waiting for maintenance, continue running
                    urgency = get_urgency_score_from_lut(urgency_lut, rul, importance)
                    service_candidates.append((urgency, i, 'maintenance'))

            # For operational engines
            elif status == 'operational':
                rul = engine_ruls[i]
                importance = engine_importance_data[i]

                # Check if RUL hit failure threshold
                if rul <= RUL_FAILURE_THRESHOLD:
                    total_failures += 1
                    service_candidates.append((100.0, i, 'repair'))
                    engine_status[i] = 'awaiting_repair'
                    engines_down_this_shift[i] = True
                else:
                    # Check if urgency triggers maintenance
                    urgency = get_urgency_score_from_lut(urgency_lut, rul, importance)
                    if urgency > individual_thresholds[i]:  #<--- these are the variables being optimized
                        service_candidates.append((urgency, i, 'maintenance'))
                        engine_status[i] = 'awaiting_maintenance'

        # Loop 2: Assign available teams
        # Once service candidates are identified, assign available teams (per shift)
        available_teams = MAX_TEAMS - current_teams_used
        # Sort candidates by priority
        sorted_candidates = sorted(service_candidates, key=lambda x: x[0], reverse=True)

        # Assign teams to the highest priority candidates first
        for (priority, i, service_type) in sorted_candidates:
            if available_teams > 0 and (engine_status[i] == 'awaiting_repair' or engine_status[i] == 'awaiting_maintenance'):
                available_teams -= 1
                engine_status[i] = service_type
                shifts_in_shop[i] = 1

                # Add service cost
                if service_type == 'repair':
                    total_cost += FAILURE_COST
                elif service_type == 'maintenance':
                    total_cost += MAINTENANCE_COST

        # Loop 3: Update RUL and calculate total downtime for the shift
        for i in range(n_engines):
            # Only operational and awaiting_maintenance engines continue running (RUL decreases by CYCLES_PER_SHIFT)
            if engine_status[i] == 'operational' or engine_status[i] == 'awaiting_maintenance':
                engine_ruls[i] -= CYCLES_PER_SHIFT

            # Count downtime cost (only engines in maintenance/repair or awaiting_repair are down this shift)
            if engines_down_this_shift[i]:
                imp = int(engine_importance_data[i])
                if imp == 1:
                    total_downtime_cost += LOW_DOWNTIME_COST_PER_SHIFT
                elif imp == 2:
                    total_downtime_cost += MEDIUM_DOWNTIME_COST_PER_SHIFT
                else:
                    total_downtime_cost += HIGH_DOWNTIME_COST_PER_SHIFT

    # Add accumulated downtime cost
    total_cost += total_downtime_cost

    return total_cost, total_failures


# 5. Main Execution

if __name__ == "__main__":

    #%% 1. reading data test_processed.csv and define initial RULs and importance
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
    # Use first() to get initial (maximum) RUL, as RUL decreases over time in the data
    grouped = df.groupby(engine_col)[rul_col]
    initial_ruls = grouped.first().values

    N_ENGINES = len(initial_ruls)
    if N_ENGINES == 0:
        print("error: no engine data loaded from CSV.")
        exit()
    
    n_sample = min(100, N_ENGINES) # Sample xxx engines 
    print(f"Total engines found: {N_ENGINES}. Sampling {n_sample} for demo.")
    sampled_indices = random.sample(range(N_ENGINES), n_sample)
    initial_ruls = initial_ruls[sampled_indices]
    N_ENGINES = n_sample 

    # Dynamically create importance groups based on n_sample
    # Assume, 1/3 engines are running low important missions, 1/3 are medium, and 1/3 are high
    n_low = n_sample // 3
    n_med = n_sample // 3
    n_high = n_sample - n_low - n_med 
    
    importance_low = np.full(n_low, 1)      # low importance
    importance_medium = np.full(n_med, 2)   # medium importance
    importance_high = np.full(n_high, 3)     # high importance
    simulated_engine_importance = np.concatenate([importance_low, importance_medium, importance_high])
    np.random.shuffle(simulated_engine_importance) # randomly distribute importance to all engines
    print(f"Created simulated 'Task Importance' (1, 2, 3) for {N_ENGINES} engines.")

    data_max_rul = MAX_RUL_CONFIG 
    print(f"Finished loading initial RUL data for {N_ENGINES} engines.")
    print(f"Average *initial* RUL (at simulation start): {np.mean(initial_ruls):.2f} cycles")
    print(f"Fuzzy system MAX_RUL set to: {data_max_rul} cycles")
    print(f"Simulation config: {CYCLES_PER_SHIFT} cycle(s) per shift")
    #%% 2. Define fuzzy logic system
    # Create the Lookup Table (LUT) ONCE, to speed up evaluations
    start_main = time.time()
    urgency_lut = create_urgency_lookup_table(data_max_rul)
    print("Fuzzy logic LUT created.")

    #%% 3. Define DEAP NSGA-II components
    print(f"\n--- defining NSGA-II genetic algorithm ---")
    print(f"Pop_size: {POP_SIZE}, N_gen: {NGEN}, N_engines: {N_ENGINES}")

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_thresh", random.uniform, MIN_THRESHOLD, MAX_THRESHOLD)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_thresh, 3)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_by_importance,
                     n_engines=N_ENGINES,
                     initial_ruls_data=initial_ruls, 
                     urgency_lut=urgency_lut,
                     engine_importance_data=simulated_engine_importance)

    toolbox.register("mate", tools.cxBlend, alpha=0.2) 
    toolbox.register("mutate", tools.mutGaussian, mu=0.3, sigma=0.2, indpb=0.2)  # Increased sigma and indpb for more diversity
    toolbox.register("select", tools.selNSGA2) 

    #%% 4. Run baseline strategy (Run-to-Failure)
    print("\n--- evaluating [baseline: Run-to-Failure] strategy ---")
    # Baseline strategy: only maintain when urgency is almost max (10.0)
    r2f_thresholds = [10.0 - 1e-5] * N_ENGINES 
    r2f_cost, r2f_failures = toolbox.evaluate(r2f_thresholds)

    print(f"Total cost: ${r2f_cost:,.0f}")
    print(f"Total failures: {r2f_failures:,.0f} events")


    #%% 5. Run NSGA-II optimization
    print("\n--- running [optimization: NSGA-II + Fuzzy Policy] strategy ---")
    print("This may take a few minutes...")
    
    pop = toolbox.population(n=POP_SIZE)
    
    # Track all evaluated individuals for visualization
    all_evaluated = []
    
    # Custom evaluation wrapper to track history
    original_evaluate = toolbox.evaluate
    def evaluate_and_track(*args, **kwargs):
        result = original_evaluate(*args, **kwargs)
        all_evaluated.append(result)  # Store (cost, failures)
        return result
    toolbox.register("evaluate", evaluate_and_track)
    
    # NSGA-II main loop start
    algorithms.eaMuPlusLambda(pop, toolbox, mu=POP_SIZE, lambda_=POP_SIZE, 
                              cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, verbose=False)

    print("NSGA-II optimization completed.")
    print(f"Total evaluations: {len(all_evaluated)}")
    print(f"\n--- Total Optimization Time: {time.time() - start_main:.2f} seconds ---")

    #%% 6. Results
    print("\n--- Optimization Results (Pareto Front) ---")
    pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

    print(f"Baseline (R2F):   Cost=${r2f_cost:,.0f}, Failures={r2f_failures:,.0f}")
    print(f"NSGA-II found {len(pareto_front)} Pareto optimal solutions:")
    
    # Display all Pareto solutions
    for i, sol in enumerate(pareto_front):
        cost, failures = sol.fitness.values
        print(f"  Solution {i+1}: Cost=${cost:,.0f}, Failures={failures:,.0f}") 

    # Find the lowest cost solution for example display
    best_cost_sol = min(pareto_front, key=lambda sol: sol.fitness.values[0])
    cost, failures = best_cost_sol.fitness.values
    
    print("\n---")
    print(f"Example Selection (Lowest Cost Solution):")
    print(f"Cost: ${cost:,.0f} (Savings: ${r2f_cost - cost:,.0f})")
    print(f"Failures: {failures:,.0f} (Reduction: {r2f_failures - failures:,.0f} events)")
    print(f"Low importance (1):    {best_cost_sol[0]:.2f}")
    print(f"Medium importance (2): {best_cost_sol[1]:.2f}")
    print(f"High importance (3):   {best_cost_sol[2]:.2f}")

    #%% Save Pareto Front Results to CSV
    # Create output directory if not exists
    output_dir = os.path.join(base_dir, 'optimization_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for all Pareto solutions (no clipping at all)
    pareto_results = []
    for i, sol in enumerate(pareto_front):
        cost_val, failures_val = sol.fitness.values
        thresh_low = float(sol[0])
        thresh_med = float(sol[1])
        thresh_high = float(sol[2])
        
        pareto_results.append({
            'Solution_ID': i + 1,
            'Total_Cost': cost_val,
            'Total_Failures': failures_val,
            'Cost_Saving_vs_Baseline': r2f_cost - cost_val,
            'Failure_Reduction_vs_Baseline': r2f_failures - failures_val,
            'Threshold_Low_Importance': thresh_low,
            'Threshold_Med_Importance': thresh_med,
            'Threshold_High_Importance': thresh_high
        })
    
    # Save to CSV
    df_pareto = pd.DataFrame(pareto_results)
    csv_path = os.path.join(output_dir, 'pareto_front_solutions.csv')
    df_pareto.to_csv(csv_path, index=False)
    print(f"\n[Saved] Pareto front solutions to: {csv_path}")
    
    # Visualization
    # Extract data - use all_evaluated history instead of just final pop
    all_costs = [fit[0] for fit in all_evaluated]
    all_failures = [fit[1] for fit in all_evaluated]
    pareto_costs = df_pareto['Total_Cost'].values
    pareto_failures = df_pareto['Total_Failures'].values
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(all_failures, all_costs, c='gray', s=40, alpha=0.6,
               marker='o', label=f'Explored Solutions ({len(all_evaluated)})', zorder=1)
    pareto_sorted = sorted(zip(pareto_failures, pareto_costs))
    if len(pareto_sorted) > 0:
        pf_fail, pf_cost = zip(*pareto_sorted)
        ax.plot(pf_fail, pf_cost, color='blue', linestyle='--', linewidth=1.0, alpha=0.7, zorder=2)
        ax.scatter(pareto_failures, pareto_costs, c='blue', s=120, alpha=0.9,
                   marker='o', label=f'Pareto Front ({len(pareto_front)})', zorder=3)

    # Plot baseline and best solution using standard colors and markers
    ax.scatter([r2f_failures], [r2f_cost], c='red', s=180, marker='X',
               label='Baseline (R2F)', zorder=4)
    ax.set_xlabel('Total Failures')
    ax.set_ylabel('Total Cost ($)')
    ax.set_title('NSGA-II Optimization Results')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    