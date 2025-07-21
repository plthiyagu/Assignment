import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

def get_sentence_embeddings(texts, model_name='all-MiniLM-L6-v2', batch_size=32):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return embeddings

def standard_scale_embeddings(df, embedding_col='embedding', scaled_col='scaled_embedding'):
    embedding_array = np.vstack(df[embedding_col].values)
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embedding_array)
    df[scaled_col] = list(scaled_embeddings)
    return df

def reduce_dimensions(df, embedding_col='scaled_embedding', method='pca', n_components=2, col_prefix='dimred'):
    embeddings = np.vstack(df[embedding_col].values)

    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
    elif method.lower() == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'pca', 'tsne', or 'umap'.")

    components = reducer.fit_transform(embeddings)

    for i in range(n_components):
        df[f'{col_prefix}_{i}'] = components[:, i]
    return df