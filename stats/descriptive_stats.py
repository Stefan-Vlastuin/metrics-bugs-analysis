import pandas as pd


def show_descriptive_stats(df):
    pd.set_option('display.max_columns', None)  # Show all columns
    stats = df.describe()
    print("DESCRIPTIVE STATISTICS")
    print(stats)
    print()
    print("DESCRIPTIVE STATISTICS (LATEX)")
    print(stats.transpose().to_latex(float_format="%.2f",
                                     columns=['mean', 'std', 'min', 'max', '25%', '50%', '75%', 'max']))
