import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_lines(x_list, mean1, ci1, mean2, ci2, label1, label2, xlabel, ylabel, title, save_path, ylims=None, legend_loc='upper right'):
    # Plot data
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("colorblind")

    ax.plot(x_list, mean1, label=label1, color=colors[0], linestyle='-', linewidth=2)
    ax.fill_between(x_list, np.array(mean1) - np.array(ci1), np.array(mean1) + np.array(ci1), color=colors[0], alpha=0.3)

    if mean2 is not None:
        ax.plot(x_list, mean2, label=label2, color=colors[1], linestyle='--', linewidth=2)
        ax.fill_between(x_list, np.array(mean2) - np.array(ci2), np.array(mean2) + np.array(ci2), color=colors[1], alpha=0.3)

    if ylims is not None:
        ax.set_ylim(ylims)
    ax.autoscale(enable=True, axis='y')  # Disable auto-scaling for y-axis
    plt.tight_layout()

    # Customize the grid, legend, and labels
    ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.5, color='lightgrey')
    ax.grid(False, axis='x')
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)

    # Customize the legend
    legend = ax.legend(loc=legend_loc, frameon=True, framealpha=0.9, fontsize=12)
    frame = legend.get_frame()
    frame.set_color('white')

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(True)

    # set x-axis ticks
    ax.set_xticks(x_list)

    # Set white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # save the plot
    plt.savefig(save_path, bbox_inches='tight')



def plot_lines2(x_list_a, mean1_a, ci1_a, mean2_a, ci2_a, label1_a, label2_a,
               x_list_b, mean1_b, ci1_b, mean2_b, ci2_b, label1_b, label2_b,
               xlabel, ylabel, title, save_path, legend_loc='upper right'):
    # Plot data
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use the same color palette
    colors = sns.color_palette("colorblind")

    # Plot first set of data
    ax.plot(x_list_a, mean1_a, label=f"{label1_a}", color=colors[0], linestyle='-', linewidth=2)
    ax.fill_between(x_list_a, np.array(mean1_a) - np.array(ci1_a), np.array(mean1_a) + np.array(ci1_a), color=colors[0], alpha=0.3)

    if mean2_a is not None:
        ax.plot(x_list_a, mean2_a, label=f"{label2_a}", color=colors[1], linestyle='-', linewidth=2)
        ax.fill_between(x_list_a, np.array(mean2_a) - np.array(ci2_a), np.array(mean2_a) + np.array(ci2_a), color=colors[1], alpha=0.3)

    # Plot second set of data
    ax.plot(x_list_b, mean1_b, label=f"{label1_b}", color=colors[0], linestyle='--', linewidth=2)
    ax.fill_between(x_list_b, np.array(mean1_b) - np.array(ci1_b), np.array(mean1_b) + np.array(ci1_b), color=colors[0], alpha=0.1)

    if mean2_b is not None:
        ax.plot(x_list_b, mean2_b, label=f"{label2_b}", color=colors[1], linestyle='--', linewidth=2)
        ax.fill_between(x_list_b, np.array(mean2_b) - np.array(ci2_b), np.array(mean2_b) + np.array(ci2_b), color=colors[1], alpha=0.1)

    # Customize the grid, legend, and labels
    ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.5, color='lightgrey')
    ax.grid(False, axis='x')
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)

    # Customize the legend
    legend = ax.legend(loc=legend_loc, frameon=True, framealpha=0.6, fontsize=12)
    frame = legend.get_frame()
    frame.set_color('white')

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(True)

    # set x-axis ticks
    ax.set_xticks(sorted(set(x_list_a + x_list_b)))

    # Set white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # save the plot
    plt.savefig(save_path, bbox_inches='tight')

def plot_bar(
    x_list, means1, ci1, label1, means2, ci2, label2, xlabel, ylabel, title, save_path, ylims,
    width=0.35, xfontsize=15, legend_fs=20): # Increased default width
    """Generates and saves a bar plot with one or two sets of data.

    Args:
        x_list (list): List of x-axis tick labels (e.g., environment names).
        means1 (list): List of mean values for the first data set.
        ci1 (list): List of confidence interval values for the first data set.
        label1 (str): Label for the first data set.
        means2 (list or None): List of mean values for the second data set. None if only one data set.
        ci2 (list or None): List of confidence interval values for the second data set. None if only one data set.
        label2 (str or None): Label for the second data set. None if only one data set.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        save_path (str): Path to save the generated plot image.
        ylims (tuple): Tuple containing the lower and upper limits for the y-axis (e.g., (0, 100)).
        width (float, optional): Width of the bars. Adjust this to change spacing. Defaults to 0.35.
    """
    
    # Apply a predefined formatting style (e.g., for a specific conference)
    # formatter = get_format("NeurIPS") # options: ICLR, ICML, NeurIPS, InfThesis
    
    # Create a figure and an axes object
    # The figsize argument controls the size of the figure in inches
    # Increased figure width to give more space for x-tick labels
    # set font to times new roman
    plt.rc('font', family='Times New Roman')

    fig, ax = plt.subplots(figsize=(7, 6)) # Changed figsize
    
    colors = sns.color_palette("colorblind", n_colors=2 if means2 is not None else 1) # Changed color palette
    
    # Remove the top, right, and left spines (lines) of the plot for a cleaner look
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Keep bottom spine for x-axis reference
    ax.spines["bottom"].set_color('gray') # Optionally style the bottom spine
    ax.spines["bottom"].set_linewidth(0.5)


    # Remove y-axis grid lines (can be re-enabled later if desired)
    ax.yaxis.grid(False) # Initially turn off y-grid
    
    # Add a subtle horizontal grid for the y-axis
    ax.grid(axis='y', color='gray', alpha=0.4, linestyle='--') # Made grid lighter and dashed
    
    # Remove y-tick marks, customize x-tick marks
    ax.tick_params(axis='y', which='both', length=0)
    # Add padding to x-tick labels and customize their appearance
    ax.tick_params(axis='x', which='both', length=0, pad=5) # Added pad=10 for spacing

    # Set the y-axis limits
    ax.set_ylim(ylims)

    # Generate an array of x positions for the bars
    x = np.arange(len(x_list))
    
    if means2 is not None:
        # Plotting two sets of data (grouped bar chart)
        # Plot bars for the first data set, shifted to the left
        ax.bar(x - width/2, means1, width, yerr=ci1, label=label1, color=colors[0], alpha=0.85, edgecolor='gray') # Added alpha and edgecolor

        # Plot bars for the second data set, shifted to the right
        ax.bar(x + width/2, means2, width, yerr=ci2, label=label2, color=colors[1], alpha=0.85, edgecolor='gray') # Added alpha and edgecolor
    else:
        # Plotting a single set of data
        ax.bar(x, means1, width, yerr=ci1, label=label1, color=colors[0], alpha=0.85, edgecolor='gray') # Added alpha and edgecolor

    # Customize the grid: major y-axis grid lines (already handled above)
    # ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.5, color='lightgrey')
    # Ensure no grid lines for the x-axis
    ax.grid(False, axis='x')
    
    # Set labels and title with specified font sizes
    ax.set_xlabel(xlabel, fontsize=25, labelpad=15) # Added labelpad for x-axis label
    ax.set_ylabel(ylabel, fontsize=25, labelpad=15) # Added labelpad for y-axis label
    ax.set_title(title, fontsize=22, pad=20) # Increased title fontsize and padding

    # Customize the legend if there are two data sets
    if means2 is not None:
        legend = ax.legend(loc='upper left', frameon=False, fontsize=legend_fs) # Removed legend frame for a cleaner look
        # frame = legend.get_frame()
        # frame.set_color('white') # Set legend background color (not needed if frameon=False)

    # Set x-axis tick positions
    ax.set_xticks(x)
    # Set x-axis tick labels with a larger font size and rotation
    ax.set_xticklabels(x_list, fontsize=xfontsize,) # Changed fontsize, added rotation and horizontal alignment

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout() 

    # Save the plot to the specified path
    # bbox_inches='tight' ensures that all elements of the plot are included in the saved image
    plt.savefig(save_path, bbox_inches='tight', dpi=300) # Added dpi for higher resolution
#     plt.close(fig) # Close the figure to free up memory


if __name__ == '__main__':
    # Example usage
    # box = False
    # exp_list = [0, 1, 3, 5, 7, 10] if not box else [0, 5, 10]
    # mean_score = [1, 2, 3, 4, 5, 6]
    # ci_95 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # mean_score_no_prior = [2, 3, 4, 5, 6, 7]
    # ci_95_no_prior = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    # env = 'env'
    # goal = 'goal'
    # model = 'model'
    # exp = 'exp'
    # save_path = os.path.join('./plots', f"{env}_{goal}_{model}_{exp}_oed_error.png")

    # plot_lines(exp_list, mean_score, ci_95, mean_score_no_prior, ci_95_no_prior, "With Prior", "Without Prior", "Number of Observations", "Error", "Error with and without Prior", save_path)
    # Example usage
    # box = False
    # exp_list_a = [0, 1, 3, 5, 7, 10] if not box else [0, 5, 10]
    # exp_list_b = [0, 2, 4, 6, 8, 10]
    # mean_score_a = [1, 2, 3, 4, 5, 6]
    # ci_95_a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # mean_score_no_prior_a = [2, 3, 4, 5, 6, 7]
    # ci_95_no_prior_a = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    # mean_score_b = [2, 2.5, 3.5, 4.5, 5.5, 6.5]
    # ci_95_b = [0.1, 0.15, 0.25, 0.35, 0.45, 0.55]
    # mean_score_no_prior_b = [3, 3.5, 4.5, 5.5, 6.5, 7.5]
    # ci_95_no_prior_b = [0.15, 0.2, 0.3, 0.4, 0.5, 0.6]

    # env = 'env'
    # goal = 'goal'
    # model = 'model'
    # exp = 'exp'
    # save_path = os.path.join('./plots', f"{env}_{goal}_{model}_{exp}_oed_error.png")

    # plot_lines2(exp_list_a, mean_score_a, ci_95_a, mean_score_no_prior_a, ci_95_no_prior_a, 
    #         "With Prior A", "Without Prior A", 
    #         exp_list_b, mean_score_b, ci_95_b, mean_score_no_prior_b, ci_95_no_prior_b, 
    #         "With Prior B", "Without Prior B", 
    #         "Number of Observations", "Error", "Error with and without Prior", save_path)

    x_labels = ['env1', 'env2', 'env3', 'env4', 'env5', 'env6']
    means_prior = [1, 2, 3, 4, 5, 6]
    ci_prior = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    means_no_prior = [2, 3, 4, 5, 6, 7]
    ci_no_prior = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    plot_bar(x_labels, means_prior, ci_prior, "Prior", means_no_prior, ci_no_prior, "No Prior", "Environment", "Error", "Environment Difficulty Comparison", "environment_difficulty_comparison.png", (0.1, 10), width=0.25)