# MECE-694-project
MECE 694 project

Core idea

Target: 
1. An MLP model that can predict the RUL of jet engines
2. An optimization model that can find the best urgency value defined by fuzzy logic to minimize the total cost and engine failure time
- the urgency value is linked to decision of when to do predictive maintenance
- constraint is number of maintenance teams



2. Optimization part

    Fuzzy system: 2 input 1 output:
    - inputs: 
        - Current RUL -> low、medium、high
        - Mission importance -> low、medium、high
    - rules:
        - rule1 = ctrl.Rule(rul['high'], urgency['low'])
        - rule2 = ctrl.Rule(rul['medium'] & importance['low'], urgency['low'])
        - rule3 = ctrl.Rule(rul['medium'] & importance['medium'], urgency['medium'])
        - rule4 = ctrl.Rule(rul['medium'] & importance['high'], urgency['high'])
        - rule5 = ctrl.Rule(rul['low'] & importance['low'], urgency['medium'])
        - rule6 = ctrl.Rule(rul['low'] & (importance['medium'] | importance['high']), urgency['high'])
    - output:
        - urgency value

    Optimization problem:
    - method: NSGA-II
    - problem definition: 
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

        # RUL failure threshold: when RUL drops below this, engine MUST be repaired (cannot run anymore)
        RUL_FAILURE_THRESHOLD = 50  # cycles (if RUL <= this value, force repair)

        # Simulation time step configuration
        CYCLES_PER_SHIFT = 2  # number of cycles consumed per shift (adjustable)

    - system dynamic model:
        - Status space:
            - 'operational'             : engine is running normally
            - 'awaiting_maintenance'    : maintenance has been scheduled/queued, engine can continue running
            - 'maintenance'             : maintenance is in progress (team assigned, engine down)
            - 'awaiting_repair'         : repair has been scheduled/queued, engine down
            - 'repair'                  : repair is in progress (team assigned, engine down)

        - Logic:
            - RUL_FAILURE_THRESHOLD: if RUL <= this value, engine must be repaired and can't run anymore
            - Maintenance THRESHOLD: if RUL <= this value, engine must be maintained (but can still run)
            - awaiting_maintenance: engine continues to run (RUL decreases) until:
                1. RUL hits failure threshold -> escalate to awaiting_repair
                2. Team becomes available -> start maintenance

        - Target: Find the best individual maintenance thresholds for engines that running different important missions that can:
            - Minimize total cost (maintenance + failure + downtime)
            - Minimize total failures
        
        - Parameters to be optimized:
            - Maintenance THRESHOLD for low/medium/high emergency mission engines