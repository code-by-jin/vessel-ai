from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def survival_analysis(groups, ax):
    # Load and prepare data (assuming this part is done correctly as before)

    # Prepare data for log-rank test
    durations = []
    events = []
    labels = []

    for (label, group) in groups:
        T = group["DaysBXtoESRDorEGFR40_LR"]
        E = group["ESRDorEGFR40BX_LR"]
        durations.append(T)
        events.append(E)
        labels.append(label)

        if len(T) > 0:  # Ensure there is data to fit
            km = KaplanMeierFitter()
            km.fit(durations=T, event_observed=E, label=f"Severity {label}")
            km.plot_survival_function(ax=ax, show_censors=True)  # Set ci_show=False if CI not needed         

    # Perform and display log-rank test
    if len(groups) > 1:
        from itertools import combinations
        p_values = []
        for (i, j) in combinations(range(len(groups)), 2):
            result = logrank_test(durations[i], durations[j], event_observed_A=events[i], event_observed_B=events[j])
            p_values.append((f"{labels[i]} vs {labels[j]}", result.p_value))

        # title_text = f"Survival Analysis by {severity_column.replace('_', ' ')}\n"
        title_text = "Severity " + "\n".join([f"{pair[0]}: p={pair[1]:.4f}" for pair in p_values])
        ax.set_title(title_text, fontsize=18)

    # Customize plot
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel('Days from Biopsy to ESRD or EGFR < 40', fontsize=15)
    ax.legend(fontsize=15, loc="lower left")
