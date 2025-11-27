import os
import numpy as np
from joblib import dump, load
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
import torch

def extract_features_with_pca_scaler(unet_lightning_model, dataloader: DataLoader, device, n_components=32, models_dir="models", result_dir="result", pca_model_path=None, scaler_model_path=None):
    """
    Extract bottleneck features from UNetLightning, reduce with PCA and normalize with MinMaxScaler.
    Saves features and indices to a .npz and returns its path.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    model = unet_lightning_model.model.to(device)
    model.eval()

    # PCA
    if pca_model_path and os.path.exists(pca_model_path):
        pca = load(pca_model_path)
        print(f"Loaded PCA from: {pca_model_path}")
        pca_fitted = True
    else:
        pca = PCA(n_components=n_components)
        pca_fitted = False

    # Scaler
    if scaler_model_path and os.path.exists(scaler_model_path):
        scaler = load(scaler_model_path)
        print(f"Loaded Scaler from: {scaler_model_path}")
        scaler_fitted = True
    else:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler_fitted = False

    features_list, indices_list = [], []

    with torch.no_grad():
        for images, _, indices in dataloader:
            images = images.to(device)
            feats = model.extract_features(images).cpu().numpy()
            feats = feats.reshape(feats.shape[0], -1)  # flatten spatial dims

            if not pca_fitted:
                pca.fit(feats)
                dump(pca, os.path.join(models_dir, 'pca_model.joblib'))
                print(f"Saved PCA -> {os.path.join(models_dir, 'pca_model.joblib')}")
                pca_fitted = True

            reduced = pca.transform(feats)
            features_list.extend(reduced)
            indices_list.extend(indices.cpu().numpy())

    features_array = np.array(features_list)

    if not scaler_fitted:
        scaler.fit(features_array)
        dump(scaler, os.path.join(models_dir, 'scaler_model.joblib'))
        print(f"Saved Scaler -> {os.path.join(models_dir, 'scaler_model.joblib')}")

    normalized = scaler.transform(features_array)

    save_npz = os.path.join(result_dir, 'features_and_indices.npz')
    np.savez_compressed(save_npz, features=normalized, indices=np.array(indices_list))
    print(f"Saved features -> {save_npz}")
    return save_npz

def clusterize_kmeans(features_npz_path, n_clusters, models_dir="models", output_path=None):
    """
    Run KMeans on normalized features, save model and cluster labels.
    """
    data = np.load(features_npz_path)
    features = data['features']
    indices = data['indices']

    kmeans = KMeans(n_clusters=n_clusters, random_state=1234)
    labels = kmeans.fit_predict(features)

    os.makedirs(models_dir, exist_ok=True)
    km_path = os.path.join(models_dir, 'kmeans_model.pkl')
    dump(kmeans, km_path)
    print(f"Saved KMeans -> {km_path}")

    if output_path is None:
        output_path = os.path.join(os.path.dirname(features_npz_path), 'clusters.npz')
    np.savez_compressed(output_path, cluster_labels=labels, indices=indices)
    print(f"Saved clusters -> {output_path}")
    return output_path