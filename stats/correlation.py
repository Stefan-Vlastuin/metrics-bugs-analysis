import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def show_correlation(df):
    pd.set_option('display.max_columns', None)  # Show all columns

    standardized_df = (df - df.mean()) / df.std()
    correlation_matrix = standardized_df.corr()  # Uses Pearson correlation coefficient

    plt.figure(figsize=(40, 30))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.show()
