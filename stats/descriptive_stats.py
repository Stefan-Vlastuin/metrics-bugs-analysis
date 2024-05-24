import pandas as pd


def show_descriptive_stats(df):
    pd.set_option('display.max_columns', None)  # Show all columns
    stats = df.describe()
    print(stats)



