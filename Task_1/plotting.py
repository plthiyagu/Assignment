import matplotlib.pyplot as plt
import seaborn as sns

def plot_and_save_embeddings_no_section(df,
                                        x_col='pca_0',
                                        y_col='pca_1',
                                        cluster_col='cluster',
                                        outlier_col='is_outlier',
                                        output_prefix='embedding_plot'):
    plt.figure(figsize=(8, 6))

    # Plot embeddings colored by cluster
    plt.clf()
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=cluster_col, palette='tab10', s=50, alpha=0.8)
    plt.title('Embeddings colored by Cluster')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_clusters.jpg', dpi=300)
    plt.show()

    # Plot embeddings colored by outlier flag
    plt.clf()
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=outlier_col, palette={False: 'blue', True: 'red'}, s=50, alpha=0.8)
    plt.title('Embeddings colored by Outlier Flag')
    plt.legend(title='Is Outlier', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_outliers.jpg', dpi=300)
    plt.show()