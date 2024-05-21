import pandas as pd

from read_data.read_data import get_dataframe


def show_descriptive_stats(df):
    pd.set_option('display.max_columns', None)  # Show all columns
    stats = df.describe()
    print(stats)


show_descriptive_stats(get_dataframe())
