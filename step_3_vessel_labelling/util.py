import numpy as np  
import pandas as pd 
import matplotlib.ticker as ticker


def plot_artery_counts_by_case(artery_type, counts, ax):
    # Define the bins for the histogram - bins from 0 to 25, with an extra bin for values >25
    bins = np.arange(28)
    # Process counts to clip at 25
    processed_counts = [min(x, 26) for x in counts]  # Values greater than 25 are set to 25
    # Plot histogram with defined bins
    ax.hist(processed_counts, bins=bins, alpha=0.5, color='blue')
    # Labels for x-ticks, handling >25 as a special case
    labels = [str(i) for i in range(26)] + ['>25']
    # Set x-tick labels
    bin_width = bins[1] - bins[0]
    ax.set_xticks(np.arange(len(labels)) * bin_width + bin_width / 2)
    ax.set_xticklabels(labels, fontsize=15)  # Set font size for x-ticks
    ax.tick_params(axis='y', which='major', labelsize=15)  # Set font size for y-ticks

    # Set axis labels and title with context-specific information
    ax.set_xlabel(f'Count of {artery_type} per Whole Slide Image', fontsize=18)
    ax.set_ylabel('Frequency of Slides', fontsize=18)
    ax.set_title(f'Distribution of {artery_type} Across Slides', fontsize=20)


# Count occurrences of each severity level within each Artery Type
def barplot(counts, col, ax):
    # Plotting directly on the provided axis
    counts.plot(kind='bar', ax=ax, legend=True)
    ax.set_title(f'Distribution of {col} by Artery Type', fontsize=20)
    ax.set_ylabel('Count', fontsize=15)
    ax.set_xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, fontsize=18)
    ax.tick_params(axis='y', labelsize=18)

    # Legend configuration
    ax.legend(title=col, fontsize=15, title_fontsize=15)

    # Annotating bars with their heights
    max_height = max(counts.max())
    ax.set_ylim(0, max_height * 1.1)  # Scale y-axis to fit annotations
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10)
        

def distribution_analysis(df, col, ax, title, color):
    # Check if the data is continuous or discrete
    # Assuming data is discrete if unique values are few and all are integers
    data = df[col].dropna()
    unique_values = np.sort(data.unique())
    is_all_integers = all(value.is_integer() for value in unique_values)

    is_continuous = len(unique_values) > 7 and not is_all_integers
    
    if is_continuous:
        # Handle zeros separately if present
        zero_count = (data == 0).sum()
        non_zero_data = data[data != 0]
        max_val = non_zero_data.max()
        
        bins = np.linspace(0, max_val, 6)  # Create bins between 0 and max_val
        bins = np.insert(bins, 0, -np.finfo(float).eps)  # Start bins from zero

        # Bin the data
        data_binned = pd.cut(data, bins=bins, include_lowest=True, right=True)
        counts = data_binned.value_counts().sort_index()

        # Create labels for bins
        labels = ['0'] if zero_count > 0 else []  # Label for zero
        labels += [f"({bins[i]:.2f}, {bins[i+1]:.2f}]" for i in range(1, len(bins)-1)]

        # Plot the counts
        counts.plot(kind='bar', ax=ax, color=color, alpha=0.75)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45)  # Rotate labels for better visibility
    else:
        if is_all_integers:
            unique_values = unique_values.astype(int)
        # Use bar chart for discrete data
        counts = data.value_counts().sort_index()
        counts.plot(kind='bar', ax=ax, color=color, alpha=0.75)
        ax.set_xticks(range(len(unique_values)))
        ax.set_xticklabels(unique_values, rotation=0)

    
    # Setting the labels and titles
    ax.set_xlabel('Severity Score', fontsize=15)
    ax.set_title(title, fontsize=18)
    ax.set_ylabel('Count', fontsize=15)
    ax.tick_params(axis='y', which='major', labelsize=15)

    # Annotate bars with counts
    for p in ax.patches:
        ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
