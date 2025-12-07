import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from matplotlib.patches import Patch
from skfuzzy import control as ctrl
import time


MEMBERSHIP_TITLE_FONTSIZE = 18


def build_fuzzy_system(max_rul_value, min_rul_value):
    """Create and return the fuzzy variables, rules and control system.

    Returns: rul, importance, urgency, rules, urgency_ctrl
    """
    rul_range = np.arange(min_rul_value, max_rul_value + 1, 1)
    imp_range = np.arange(1, 4, 1)
    # Use higher-resolution urgency universe for smoother MF plots and surface
    urgency_range = np.linspace(0, 10, 201)

    rul = ctrl.Antecedent(rul_range, 'rul')
    importance = ctrl.Antecedent(imp_range, 'importance')
    urgency = ctrl.Consequent(urgency_range, 'urgency')
    
    # Set defuzzification method to centroid for smooth continuous output
    urgency.defuzzify_method = 'centroid'

    # Absolute integer breakpoints for RUL MFs with min=33, max=170
    rul['very low']   = fuzz.trapmf(rul.universe, [33, 33, 47, 74])
    rul['low']        = fuzz.trimf(rul.universe, [60, 81, 102])
    rul['medium']     = fuzz.trimf(rul.universe, [88, 102, 115])
    rul['high']       = fuzz.trimf(rul.universe, [102, 122, 143])
    rul['very high']  = fuzz.trapmf(rul.universe, [129, 156, 170, 170])

    importance['low']    = fuzz.trimf(importance.universe, [1, 1, 2])
    importance['medium'] = fuzz.trimf(importance.universe, [1, 2, 3])
    importance['high']   = fuzz.trimf(importance.universe, [2, 3, 3])

    urgency['very low']   = fuzz.trapmf(urgency.universe, [0, 0, 1, 2.5])
    urgency['low']        = fuzz.trimf(urgency.universe, [1.5, 3, 4.5])
    urgency['medium']     = fuzz.trimf(urgency.universe, [3.5, 5, 6.5])
    urgency['high']       = fuzz.trimf(urgency.universe, [5.5, 7, 8.5])
    urgency['very high']  = fuzz.trapmf(urgency.universe, [7.5, 9, 10, 10])

    rule_specs = [
        ('very low', 'low', 'high'),
        ('very low', 'medium', 'very high'),
        ('very low', 'high', 'very high'),
        ('low', 'low', 'medium'),
        ('low', 'medium', 'high'),
        ('low', 'high', 'very high'),
        ('medium', 'low', 'low'),
        ('medium', 'medium', 'medium'),
        ('medium', 'high', 'high'),
        ('high', 'low', 'very low'),
        ('high', 'medium', 'low'),
        ('high', 'high', 'medium'),
        ('very high', 'low', 'very low'),
        ('very high', 'medium', 'very low'),
        ('very high', 'high', 'low'),
    ]
    rules = []
    for rul_term, imp_term, urgency_term in rule_specs:
        rule = ctrl.Rule(rul[rul_term] & importance[imp_term], urgency[urgency_term])
        rule.friendly_labels = (rul_term, imp_term, urgency_term)
        rules.append(rule)
    urgency_ctrl = ctrl.ControlSystem(rules)
    return rul, importance, urgency, rules, urgency_ctrl


def create_urgency_lookup_table(max_rul_value, min_rul_value, RUL_FAILURE_THRESHOLD=15):
    """Pre-compute the LUT using the fuzzy control system built by build_fuzzy_system with centroid defuzzification."""
    rul, importance, urgency, rules, urgency_ctrl = build_fuzzy_system(max_rul_value, min_rul_value)

    print("Pre-computing Fuzzy Logic Lookup Table (LUT) using centroid defuzzification...")
    start_time = time.time()

    n_rows = int(max_rul_value - min_rul_value + 1)
    lut = np.full((n_rows, 3), np.nan, dtype=float)

    for r in range(min_rul_value, max_rul_value + 1):
        idx = int(r - min_rul_value)
        for i in range(1, 4):
            sim = ctrl.ControlSystemSimulation(urgency_ctrl)
            sim.input['rul'] = float(r)
            sim.input['importance'] = float(i)
            sim.compute()
            # Output is already defuzzified using MOM method
            lut[idx, i-1] = float(sim.output['urgency'])

    elapsed = time.time() - start_time
    print(f"LUT pre-computation finished in {elapsed:.2f} seconds (using centroid defuzzification).")
    return lut, int(min_rul_value)


def plot_discrete_control_surface(lut, lut_min_rul, title='Discrete control surface (LUT resolution)'):
    """Plot the LUT as a discrete control surface aligned with RUL/importance indices."""
    rul_values = np.arange(lut_min_rul, lut_min_rul + lut.shape[0])
    importance_levels = np.arange(1, lut.shape[1] + 1)
    fig, ax = plt.subplots(figsize=(12, 3))
    extent = (rul_values[0] - 0.5, rul_values[-1] + 0.5,
              importance_levels[0] - 0.5, importance_levels[-1] + 0.5)
    im = ax.imshow(lut.T, origin='lower', aspect='auto', extent=extent,
                   cmap='viridis', interpolation='nearest')
    ax.set_xlabel('RUL (cycles)')
    ax.set_ylabel('Importance level')
    ax.set_yticks(importance_levels)
    ax.set_xticks(np.arange(lut_min_rul, lut_min_rul + lut.shape[0], 10))
    ax.set_title(title, fontsize=MEMBERSHIP_TITLE_FONTSIZE)
    # draw explicit grid lines every discrete point to emphasize the LUT cells
    x_edges = np.arange(rul_values[0] - 0.5, rul_values[-1] + 0.6, 1)
    y_edges = np.arange(importance_levels[0] - 0.5, importance_levels[-1] + 0.6, 1)
    for x_edge in x_edges:
        ax.axvline(x_edge, color='white', linewidth=0.4, alpha=0.8)
    for y_edge in y_edges:
        ax.axhline(y_edge, color='white', linewidth=0.8, alpha=0.8)
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Urgency score')
    return fig, ax


def plot_membership_functions(fuzzy_var, title):
    """Ensure the legend is located consistently on the membership plots."""
    viewer = fuzzy_var.view()
    if isinstance(viewer, tuple):
        fig, ax = viewer
    else:
        fig = getattr(viewer, 'fig', plt.gcf())
        ax = getattr(viewer, 'ax', fig.axes[0] if fig.axes else plt.gca())
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc='upper right',
            bbox_to_anchor=(1, 0.88),
            framealpha=0.9,
        )
    return fig, ax


def plot_rule_matrix(rul_var, importance_var, urgency_var, rules):
    """Draw a rule matrix showing urgency responses by RUL / importance term."""
    rul_terms = list(rul_var.terms.keys())
    importance_terms = list(importance_var.terms.keys())
    urgency_terms = list(urgency_var.terms.keys())
    rul_idx = {name: idx for idx, name in enumerate(rul_terms)}
    imp_idx = {name: idx for idx, name in enumerate(importance_terms)}
    urg_idx = {name: idx for idx, name in enumerate(urgency_terms)}

    matrix = np.full((len(rul_terms), len(importance_terms)), np.nan)
    for rule in rules:
        labels = getattr(rule, 'friendly_labels', None)
        if not labels:
            continue
        rul_label, imp_label, urg_label = labels
        matrix[rul_idx[rul_label], imp_idx[imp_label]] = urg_idx[urg_label]

    fig, ax = plt.subplots()
    cmap = plt.get_cmap('viridis', len(urgency_terms))
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=len(urgency_terms) - 1)
    ax.set_xticks(np.arange(len(importance_terms)))
    ax.set_xticklabels(importance_terms)
    ax.set_yticks(np.arange(len(rul_terms)))
    ax.set_yticklabels(rul_terms)
    ax.set_xlabel('Importance term')
    ax.set_ylabel('RUL term')
    ax.set_title('Fuzzy rule matrix')

    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            if np.isnan(matrix[r, c]):
                continue
            urgency_name = urgency_terms[int(matrix[r, c])]
            ax.text(c, r, urgency_name, ha='center', va='center', color='white', fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=range(len(urgency_terms)))
    return fig, ax


def plot_control_surface_3d(ctrl_sys, rul_var, imp_var, urgency_var, rul_vals=None, imp_vals=None, res=41):
    """Fallback 3D surface plot when the skfuzzy visualizer is unavailable."""
    ax = plt.figure().add_subplot(111, projection='3d')
    if rul_vals is None:
        rul_vals = np.linspace(rul_var.universe.min(), rul_var.universe.max(), res)
    if imp_vals is None:
        imp_vals = np.linspace(imp_var.universe.min(), imp_var.universe.max(), res)
    R, I = np.meshgrid(rul_vals, imp_vals)
    Z = np.zeros_like(R)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            sim = ctrl.ControlSystemSimulation(ctrl_sys)
            sim.input['rul'] = float(R[i, j])
            sim.input['importance'] = float(I[i, j])
            sim.compute()
            Z[i, j] = float(sim.output['urgency'])
    surf = ax.plot_surface(R, I, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('rul')
    ax.set_ylabel('importance')
    ax.set_zlabel('urgency')
    ax.set_title('Control surface (3D)')
    plt.colorbar(surf, ax=ax, shrink=0.6)


if __name__ == '__main__':
    max_rul_value = 170
    min_rul_value = 33
    rul, importance, urgency, rules, urgency_ctrl = build_fuzzy_system(max_rul_value, min_rul_value)

    # Use skfuzzy built-in viewers. They return fig, ax; plt.show() will block until windows closed.
    plot_membership_functions(rul, 'RUL membership functions')
    plot_membership_functions(importance, 'Importance membership functions')
    plot_membership_functions(urgency, 'Urgency membership functions')

    # Control system visualization
    from skfuzzy.control import visualization as vis
    vis_v = vis.ControlSystemVisualizer(urgency_ctrl)
    network_fig = getattr(vis_v, 'fig', None)
    setattr(vis_v, 'ctrl', urgency_ctrl)
    surface_plotted = False
    if hasattr(vis_v, 'view_surface'):
        vis_v.view_surface()
        surface_plotted = True
    if network_fig is not None:
        plt.close(network_fig)

    plot_rule_matrix(rul, importance, urgency, rules)

    if not surface_plotted:
        plot_control_surface_3d(urgency_ctrl, rul, importance, urgency, res=41)

    lut, lut_min = create_urgency_lookup_table(max_rul_value, min_rul_value)
    plot_discrete_control_surface(lut, lut_min)

    plt.show()
