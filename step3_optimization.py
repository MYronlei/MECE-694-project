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
MAINTENANCE_COST = 30000  # scheduled maintenance cost (reduced to make trade-off)
MAINTENANCE_TIME = 4       # scheduled maintenance time (shifts)
FAILURE_COST = 80000       # failure repair cost (closer to maintenance cost)
FAILURE_TIME = 10           # failure repair time (longer)

# Project impact weights (non-monetary, for second objective)
# These represent business impact severity multipliers
IMPORTANCE_IMPACT_WEIGHTS = {
    1: 1.0,    # Low importance: baseline impact
    2: 1.5,    # Medium importance: 1.5x impact
    3: 2.0     # High importance: 2x impact (critical projects)
}
MINIMUM_COMPLETION_THRESHOLD = 60.0  # Each task type must complete at least 60%

# RUL range for simulation (after repair)
MIN_RUL_AFTER_REPAIR = 250 # min RUL for a full repair (in cycles)
MAX_RUL_CONFIG = 350     # default max RUL (in cycles)

# RUL failure threshold: when RUL drops below this, engine MUST be repaired (cannot run anymore)
RUL_FAILURE_THRESHOLD = 50  # cycles (if RUL <= this value, force repair)

# Simulation time step configuration
CYCLES_PER_SHIFT = 3  # number of cycles consumed per shift (adjustable)

# --- NSGA-II Genetic Algorithm Parameters ---
POP_SIZE = 50      # population size
NGEN = 50           # number of generations 
CXPB = 0.8          # crossover probability 
MUTPB = 0.2         # mutation probability

# Optimization range for fuzzy thresholds - expanded for more diversity
MIN_THRESHOLD = 1.0  # allow very aggressive maintenance strategies
MAX_THRESHOLD = 10.0  # allow very conservative strategies (near failure)

#%% 2. RUL Simulation (for post-repair) 

def get_repaired_rul(repair_type):
    """
    Simulate RUL after maintenance, implementing partial restoration.
    (Implements Suggestion #2)
    Returns RUL in cycles.
    """
    if repair_type == 'repair':
        # Simulate repair: partial restoration (e.g., 70% of full)
        partial_min = MIN_RUL_AFTER_REPAIR
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
    
    # Define the Fuzzy System
    rul_range = np.arange(0, max_rul_value + 1, 1)
    imp_range = np.arange(1, 4, 1) # Discrete values 1, 2, 3
    
    rul = ctrl.Antecedent(rul_range, 'rul')
    importance = ctrl.Antecedent(imp_range, 'importance')
    urgency = ctrl.Consequent(np.arange(0, 11, 1), 'urgency')

    # Membership functions designed around failure threshold for better sensitivity
    critical_rul = int(np.floor(RUL_FAILURE_THRESHOLD + (max_rul_value - RUL_FAILURE_THRESHOLD) / 3))
    safe_rul = int(np.floor(max_rul_value - RUL_FAILURE_THRESHOLD)  * 0.5 + RUL_FAILURE_THRESHOLD)
    
    # Use trapezoidal MF for 'low': flat at 1.0 until failure threshold, then decline
    rul['low'] = fuzz.trapmf(rul.universe, [0, 0, RUL_FAILURE_THRESHOLD, critical_rul])
    rul['medium'] = fuzz.trimf(rul.universe, [critical_rul * 0.5, (critical_rul + safe_rul) / 2, safe_rul * 1.2])
    rul['high'] = fuzz.trimf(rul.universe, [safe_rul * 0.8, max_rul_value, max_rul_value])

    importance['low'] = fuzz.trimf(importance.universe, [1, 1, 2])
    importance['medium'] = fuzz.trimf(importance.universe, [1, 2, 3])
    importance['high'] = fuzz.trimf(importance.universe, [2, 3, 3])

    urgency['low'] = fuzz.trapmf(urgency.universe, [0, 0, 3, 5])
    urgency['medium'] = fuzz.trapmf(urgency.universe, [3, 5, 6, 7])
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
    
    # Call evaluate_schedule which returns (cost, incompletion_rate, completion_rates, failures, maintenance_count)
    # But DEAP only needs the first two for optimization
    cost, incompletion_rate, completion_rates, failures, maintenance_count = evaluate_schedule(
        per_engine_thresh, n_engines, initial_ruls_data, urgency_lut, engine_importance_data)
    
    # Store detailed metrics in individual for later retrieval
    individual.completion_rates = completion_rates
    individual.failures = failures
    individual.maintenance_count = maintenance_count
    
    return cost, incompletion_rate

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
    - Minimize total cost (maintenance + failure)
    - Maximize weighted task completion rate
    """

    # Simulation state
    engine_ruls = initial_ruls_data.copy()
    engine_status = ['operational'] * n_engines
    shifts_in_shop = np.zeros(n_engines)

    # Objective variables
    total_cost = 0
    total_failures = 0
    maintenance_count_total = 0
    
    # Second objective: Weighted Task Completion Rate
    # Track completion by importance level (for detailed analysis)
    completion_by_importance = {1: {'target': 0.0, 'actual': 0.0},
                                 2: {'target': 0.0, 'actual': 0.0},
                                 3: {'target': 0.0, 'actual': 0.0}}
    
    # Calculate theoretical target (weighted by importance) and actual completion
    total_weighted_target_cycles = 0.0
    total_weighted_actual_cycles = 0.0
    
    # Initialize each engine's weighted target
    for i in range(n_engines):
        imp = int(engine_importance_data[i])
        impact_weight = IMPORTANCE_IMPACT_WEIGHTS[imp]
        # Each engine should ideally complete: PLANNING_HORIZON shifts × CYCLES_PER_SHIFT cycles × importance_weight
        cycles = PLANNING_HORIZON * CYCLES_PER_SHIFT * impact_weight
        total_weighted_target_cycles += cycles
        completion_by_importance[imp]['target'] += cycles

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
                        maintenance_count_total += 1

        # Loop 3: Update RUL and track weighted task completion
        for i in range(n_engines):
            imp = int(engine_importance_data[i])
            impact_weight = IMPORTANCE_IMPACT_WEIGHTS[imp]
            
            # If engine is operational or awaiting_maintenance, it's producing cycles
            if engine_status[i] == 'operational' or engine_status[i] == 'awaiting_maintenance':
                engine_ruls[i] -= CYCLES_PER_SHIFT
                # Engine completes weighted cycles this shift
                completed_cycles = CYCLES_PER_SHIFT * impact_weight
                total_weighted_actual_cycles += completed_cycles
                completion_by_importance[imp]['actual'] += completed_cycles
            
            # If engine is down (maintenance/repair/awaiting_repair), no cycles produced
            # No cycles are added to actual completion

    # Calculate completion rates by importance level
    completion_rates = {}
    for imp in [1, 2, 3]:
        if completion_by_importance[imp]['target'] > 0:
            completion_rates[imp] = (completion_by_importance[imp]['actual'] / 
                                      completion_by_importance[imp]['target']) * 100.0
        else:
            completion_rates[imp] = 0.0
    
    # Calculate weighted task completion rate (0-100%)
    # Higher is better - we want to maximize this
    weighted_completion_rate = (total_weighted_actual_cycles / total_weighted_target_cycles) * 100.0
    
    # Check minimum completion constraint (each type must achieve >= 50%)
    min_completion_rate = min(completion_rates.values())
    constraint_penalty = 0.0
    
    if min_completion_rate < MINIMUM_COMPLETION_THRESHOLD:
        # Apply penalty proportional to constraint violation
        constraint_violation = MINIMUM_COMPLETION_THRESHOLD - min_completion_rate
        constraint_penalty = constraint_violation * 5.0  # Penalty coefficient

    # For minimization objective, we return the "incompletion rate"
    # So NSGA-II minimizes incompletion (which maximizes completion)
    weighted_incompletion_rate = 100.0 - weighted_completion_rate + constraint_penalty

    # Return objectives:
    # 1. Total cost (monetary) - minimize
    # 2. Weighted incompletion rate with constraint penalty - minimize
    # Also return detailed metrics for analysis
    return total_cost, weighted_incompletion_rate, completion_rates, total_failures, maintenance_count_total


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
    r2f_thresholds = [9.0] * N_ENGINES 
    r2f_cost, r2f_incompletion_rate, r2f_completion_rates, r2f_failures, r2f_maintenance = evaluate_schedule(
        r2f_thresholds, N_ENGINES, initial_ruls, urgency_lut, simulated_engine_importance)

    print(f"Total cost (maintenance + failure only): ${r2f_cost:,.0f}")
    print(f"Weighted task incompletion rate: {r2f_incompletion_rate:.2f}% (completion: {100-r2f_incompletion_rate:.2f}%)")
    print(f"  - Low importance tasks:    {r2f_completion_rates[1]:.2f}% completion")
    print(f"  - Medium importance tasks: {r2f_completion_rates[2]:.2f}% completion")
    print(f"  - High importance tasks:   {r2f_completion_rates[3]:.2f}% completion")
    print(f"Total failures: {r2f_failures}, Total maintenance: {r2f_maintenance}")


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
        all_evaluated.append(result)  # Store (cost, project_impact)
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

    # First, collect feasible individuals in the final population (meet per-importance completion threshold)
    feasible_population = [ind for ind in pop if hasattr(ind, 'completion_rates') and
                           all(rate >= MINIMUM_COMPLETION_THRESHOLD for rate in ind.completion_rates.values())]

    if len(feasible_population) > 0:
        # Compute Pareto front among feasible individuals only (preferred)
        pareto_front = tools.sortNondominated(feasible_population, len(feasible_population), first_front_only=True)[0]
        print(f"Computed Pareto front from feasible population (size={len(feasible_population)}).")
        pareto_front_all = pareto_front
        pareto_front_feasible = pareto_front  # already feasible
        pareto_front_used = pareto_front
    else:
        # No feasible individuals in final population: fallback to computing Pareto over full population
        print("Warning: no feasible individuals found in final population; computing Pareto over full population as fallback.")
        pareto_front_all = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        # Among this Pareto, mark which ones meet threshold (may be none)
        pareto_front_feasible = [sol for sol in pareto_front_all
                                  if hasattr(sol, 'completion_rates') and
                                  all(rate >= MINIMUM_COMPLETION_THRESHOLD for rate in sol.completion_rates.values())]
        pareto_front_used = pareto_front_feasible if len(pareto_front_feasible) > 0 else pareto_front_all

    print(f"Baseline (R2F):   Cost=${r2f_cost:,.0f}, Task_Completion={100-r2f_incompletion_rate:.2f}%")
    print(f"NSGA-II found {len(pareto_front_all)} Pareto optimal solutions ({len(pareto_front_feasible)} meet the completion threshold >= {MINIMUM_COMPLETION_THRESHOLD}%).")
    
    # Display sample Pareto solutions with detailed breakdown
    sample_indices = [0, len(pareto_front_used)//2, len(pareto_front_used)-1] if len(pareto_front_used) > 2 else range(len(pareto_front_used))
    for idx in sample_indices:
        sol = pareto_front_used[idx]
        cost, incompletion_rate = sol.fitness.values
        completion_rate = 100.0 - incompletion_rate
        rates = sol.completion_rates
        min_rate = min(rates.values())
        print(f"  Solution {idx+1}: Cost=${cost:,.0f}, Overall={completion_rate:.1f}% "
              f"[Low={rates[1]:.1f}%, Med={rates[2]:.1f}%, High={rates[3]:.1f}%] Min={min_rate:.1f}%") 

    # Find the lowest cost solution for example display
    # select best-cost solution from the chosen (feasible-preferred) Pareto set
    best_cost_sol = min(pareto_front_used, key=lambda sol: sol.fitness.values[0])
    cost, incompletion_rate = best_cost_sol.fitness.values
    completion_rate = 100.0 - incompletion_rate
    
    print("\n--- Best Cost Solution ---")
    print(f"Cost: ${cost:,.0f} (Change: ${cost - r2f_cost:+,.0f})")
    print(f"Overall Task Completion: {completion_rate:.2f}% (Baseline: {100.0-r2f_incompletion_rate:.2f}%)")
    print(f"Completion by Importance:")
    print(f"  Low (1):    {best_cost_sol.completion_rates[1]:.2f}% (Baseline: {r2f_completion_rates[1]:.2f}%)")
    print(f"  Medium (2): {best_cost_sol.completion_rates[2]:.2f}% (Baseline: {r2f_completion_rates[2]:.2f}%)")
    print(f"  High (3):   {best_cost_sol.completion_rates[3]:.2f}% (Baseline: {r2f_completion_rates[3]:.2f}%)")
    print(f"Maintenance Thresholds: Low={best_cost_sol[0]:.2f}, Med={best_cost_sol[1]:.2f}, High={best_cost_sol[2]:.2f}")

    #%% Save Pareto Front Results to CSV
    # Create output directory if not exists
    output_dir = os.path.join(base_dir, 'optimization_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for all Pareto solutions with detailed metrics
    pareto_results = []
    for i, sol in enumerate(pareto_front):
        cost_val, incompletion_rate_val = sol.fitness.values
        completion_rate_val = 100.0 - incompletion_rate_val
        thresh_low = float(sol[0])
        thresh_med = float(sol[1])
        thresh_high = float(sol[2])

        # Get detailed completion rates by importance
        rates = sol.completion_rates
        min_completion = min(rates.values())
        max_completion = max(rates.values())
        completion_balance_std = np.std([rates[1], rates[2], rates[3]])

        # Calculate relative improvement metrics
        baseline_completion = 100.0 - r2f_incompletion_rate
        completion_improvement = completion_rate_val - baseline_completion
        cost_change_pct = ((cost_val - r2f_cost) / r2f_cost) * 100

        # Check if solution meets the configured minimum completion threshold for all types
        meets_constraint = all(rate >= MINIMUM_COMPLETION_THRESHOLD for rate in rates.values())

        pareto_results.append({
            'Solution_ID': i + 1,
            'Total_Cost': cost_val,
            'Cost_Change_vs_Baseline_%': cost_change_pct,
            'Weighted_Task_Completion_%': completion_rate_val,
            'Completion_Improvement_vs_Baseline_%': completion_improvement,
            'Low_Importance_Completion_%': rates[1],
            'Med_Importance_Completion_%': rates[2],
            'High_Importance_Completion_%': rates[3],
            'Min_Completion_Across_Types_%': min_completion,
            'Max_Completion_Across_Types_%': max_completion,
            'Completion_Balance_StdDev': completion_balance_std,
            'Meets_Minimum_Completion_Threshold': meets_constraint,
            'Total_Failures': sol.failures,
            'Total_Maintenance': sol.maintenance_count,
            'Threshold_Low_Importance': thresh_low,
            'Threshold_Med_Importance': thresh_med,
            'Threshold_High_Importance': thresh_high
        })
    
    # Save to CSV
    df_pareto = pd.DataFrame(pareto_results)
    csv_path = os.path.join(output_dir, 'pareto_front_solutions.csv')
    df_pareto.to_csv(csv_path, index=False)
    print(f"\n[Saved] Pareto front solutions to: {csv_path}")
    
    # Visualization with task completion rate (intuitive 0-100% scale)
    # Extract data - use all_evaluated history instead of just final pop
    all_costs = [fit[0] for fit in all_evaluated]
    all_incompletion_rates = [fit[1] for fit in all_evaluated]
    all_completion_rates = [100.0 - inc for inc in all_incompletion_rates]
    
    pareto_costs = df_pareto['Total_Cost'].values
    pareto_completion_rates = df_pareto['Weighted_Task_Completion_%'].values
    # Split Pareto CSV results into feasible vs infeasible according to the configured threshold
    feasible_mask = df_pareto['Meets_Minimum_Completion_Threshold'].astype(bool)
    pareto_feasible = df_pareto[feasible_mask]
    pareto_infeasible = df_pareto[~feasible_mask]
    baseline_completion = 100.0 - r2f_incompletion_rate
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot all explored solutions
    ax.scatter(all_completion_rates, all_costs, c='gray', s=40, alpha=0.6,
               marker='o', label=f'Explored Solutions ({len(all_evaluated)})', zorder=1)
    
    # Plot Pareto front: show infeasible Pareto points (orange) and feasible Pareto points (blue)
    # Infeasible Pareto points (if any)
    if len(pareto_infeasible) > 0:
        ax.scatter(pareto_infeasible['Weighted_Task_Completion_%'].values,
                   pareto_infeasible['Total_Cost'].values,
                   c='darkorange', s=100, alpha=0.85, marker='o', label=f'Infeasible Pareto ({len(pareto_infeasible)})', zorder=3)

    # Feasible Pareto points and connecting line (sorted)
    if len(pareto_feasible) > 0:
        pf_sorted_feas = pareto_feasible.sort_values(by='Weighted_Task_Completion_%', ascending=False)
        ax.plot(pf_sorted_feas['Weighted_Task_Completion_%'].values, pf_sorted_feas['Total_Cost'].values,
                color='blue', linestyle='--', linewidth=1.5, alpha=0.8, zorder=4)
        ax.scatter(pf_sorted_feas['Weighted_Task_Completion_%'].values, pf_sorted_feas['Total_Cost'].values,
                   c='blue', s=120, alpha=0.95, marker='o', label=f'Feasible Pareto ({len(pareto_feasible)})', zorder=5)
    else:
        # If no feasible Pareto, fallback: plot all Pareto solutions as before
        pareto_sorted = sorted(zip(pareto_completion_rates, pareto_costs), reverse=True)
        if len(pareto_sorted) > 0:
            pf_completion, pf_cost = zip(*pareto_sorted)
            ax.plot(pf_completion, pf_cost, color='blue', linestyle='--', linewidth=1.5, alpha=0.8, zorder=2)
            ax.scatter(pareto_completion_rates, pareto_costs, c='blue', s=120, alpha=0.9,
                       marker='o', label=f'Pareto Front ({len(pareto_front)})', zorder=3)

    # Plot baseline
    ax.scatter([baseline_completion], [r2f_cost], c='red', s=180, marker='X',
               label='Baseline (R2F)', zorder=4)
    
    # Enhanced axis labels with clear interpretation
    ax.set_xlabel('Weighted Task Completion Rate (%)\n(Higher = Better, Accounts for Task Importance)', fontsize=11)
    ax.set_ylabel('Total Cost (Maintenance + Failure, $)', fontsize=11)
    ax.set_title('NSGA-II Multi-Objective Optimization: Cost vs Task Completion', fontsize=12, fontweight='bold')
    
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    