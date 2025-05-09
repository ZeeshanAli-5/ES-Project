import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
import math

# Load the dataset
df = pd.read_csv('Sales Dataset.csv')

# Part 1: Find the average and variance of the dataset (Amount column)
def calculate_stats(data):
    mean = np.mean(data)
    variance = np.var(data)
    return mean, variance

amount_mean, amount_variance = calculate_stats(df['Amount'])
print(f"Amount Mean: {amount_mean}")
print(f"Amount Variance: {amount_variance}")

# Part 2: Process data for histogram and pie chart
def create_frequency_distribution(data, bins=10):
    hist, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    freq_dist = pd.DataFrame({
        'bin_center': bin_centers,
        'frequency': hist,
        'bin_start': bin_edges[:-1],
        'bin_end': bin_edges[1:]
    })
    return freq_dist

# Create frequency distribution
freq_dist = create_frequency_distribution(df['Amount'])
print("Frequency Distribution:")
print(freq_dist)

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(df['Amount'], bins=10, edgecolor='black')
plt.title('Histogram of Amount')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()

# Create pie chart for Category
category_counts = df['Category'].value_counts()
plt.figure(figsize=(10, 8))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution by Category')
plt.axis('equal')
plt.show()

# Part 3: Calculate mean and variance using frequency distribution
def calc_stats_from_freq_dist(freq_dist):
    total_freq = freq_dist['frequency'].sum()
    mean = sum(freq_dist['bin_center'] * freq_dist['frequency']) / total_freq
    
    variance = sum(freq_dist['frequency'] * (freq_dist['bin_center'] - mean)**2) / total_freq
    return mean, variance

freq_mean, freq_variance = calc_stats_from_freq_dist(freq_dist)
print(f"Mean from frequency distribution: {freq_mean}")
print(f"Variance from frequency distribution: {freq_variance}")

# Part 4: Confidence interval, tolerance interval, and validation
# Split data into 80% training and 20% validation
random.seed(42)
train_size = int(0.8 * len(df))
train_indices = random.sample(range(len(df)), train_size)
train_df = df.iloc[train_indices]
validation_df = df.drop(train_indices)

# Calculate 95% confidence interval for mean using training data
def confidence_interval_mean(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - h, mean + h

# Calculate 95% confidence interval for variance using training data
def confidence_interval_variance(data, confidence=0.95):
    n = len(data)
    var = np.var(data)
    chi2_lower = stats.chi2.ppf((1 - confidence) / 2, n - 1)
    chi2_upper = stats.chi2.ppf((1 + confidence) / 2, n - 1)
    var_lower = (n - 1) * var / chi2_upper
    var_upper = (n - 1) * var / chi2_lower
    return var_lower, var_upper

# Calculate 95% tolerance interval for training data
def tolerance_interval(data, confidence=0.95, proportion=0.95):
    n = len(data)
    mean = np.mean(data)
    stdev = np.std(data)
    
    # Find k factor for two-sided tolerance interval
    df_value = n - 1
    chi_squared = stats.chi2.ppf(1 - confidence, df_value)
    k = np.sqrt((df_value * (1 + 1/n) * stats.f.ppf(proportion, 1, df_value)) / chi_squared)
    
    lower_bound = mean - k * stdev
    upper_bound = mean + k * stdev
    return lower_bound, upper_bound

# Calculate intervals for Amount using training data
train_amount = train_df['Amount']
mean_ci_lower, mean_ci_upper = confidence_interval_mean(train_amount)
var_ci_lower, var_ci_upper = confidence_interval_variance(train_amount)
tol_lower, tol_upper = tolerance_interval(train_amount)

print(f"95% Confidence interval for mean: ({mean_ci_lower}, {mean_ci_upper})")
print(f"95% Confidence interval for variance: ({var_ci_lower}, {var_ci_upper})")
print(f"95% Tolerance interval: ({tol_lower}, {tol_upper})")

# Validate with the 20% validation set
validation_amount = validation_df['Amount']
validation_mean = np.mean(validation_amount)
validation_var = np.var(validation_amount)
percentage_in_tol = sum((validation_amount >= tol_lower) & (validation_amount <= tol_upper)) / len(validation_amount) * 100

print(f"Validation set mean: {validation_mean}")
print(f"Validation set variance: {validation_var}")
print(f"Is validation mean within CI?: {mean_ci_lower <= validation_mean <= mean_ci_upper}")
print(f"Is validation variance within CI?: {var_ci_lower <= validation_var <= var_ci_upper}")
print(f"Percentage of validation data points within tolerance interval: {percentage_in_tol}%")

# Part 5: Hypothesis testing
# Hypothesis: The mean Amount for different Categories is the same
def hypothesis_test():
    # Group data by Category
    category_groups = train_df.groupby('Category')['Amount']
    
    # Perform one-way ANOVA test
    categories = []
    for category, group in category_groups:
        categories.append(group.values)
    
    f_stat, p_value = stats.f_oneway(*categories)
    
    print("\nHypothesis Test Results:")
    print(f"Null Hypothesis: The mean Amount is the same across all Categories")
    print(f"F-statistic: {f_stat}")
    print(f"P-value: {p_value}")
    
    alpha = 0.05
    if p_value < alpha:
        print(f"Reject null hypothesis (p < {alpha}): The mean Amount differs significantly across Categories")
    else:
        print(f"Fail to reject null hypothesis (p > {alpha}): No significant difference in mean Amount across Categories")
    
    # Create box plot to visualize differences
    plt.figure(figsize=(12, 8))
    train_df.boxplot(column='Amount', by='Category')
    plt.title('Amount Distribution by Category')
    plt.suptitle('')  # Remove default title
    plt.ylabel('Amount')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

hypothesis_test()