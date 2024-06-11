from matplotlib import pyplot as plt
import seaborn as sns


def show_correlation(df):
    correlation_matrix = df.corr()  # Uses Pearson correlation coefficient

    plt.figure(figsize=(23, 25))
    sns.set(font_scale=1.4)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True, cbar=True,
                cbar_kws={'orientation': 'horizontal', 'location': 'top', 'shrink': 0.75, 'pad': 0.03},
                annot_kws={'size': 10})
    plt.savefig('output/correlation_matrix.png')
    plt.show()
