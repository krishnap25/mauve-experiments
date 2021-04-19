import numpy as np
import sys
import time
import sklearn.metrics

import src.utils as utils

try:
    import faiss
    from sklearn.preprocessing import normalize
    from sklearn.decomposition import PCA
    FOUND_FAISS = True
except (ImportError, ModuleNotFoundError):
    print('faiss or sklearn not found', file=sys.stderr)
    FOUND_FAISS = False

try:
    sys.path.append('library/spreadingvectors')
    from train_spv import train_spv_and_quantize
    FOUND_SPV = True
except ImportError:
    print('SpreadingVectors not found', file=sys.stderr)
    FOUND_SPV = False

try:
    sys.path.append('library')
    from DRMM import train_drmm_and_quantize
    FOUND_DRMM = True
except (ImportError, ModuleNotFoundError):
    print('DRMM or TensorFlow not found', file=sys.stderr)
    FOUND_DRMM = False


# PR metrics
def compute_mauve_metrics(p_feats, q_feats, discretization_algo='kmeans_l1',
                          kmeans_num_clusters=100, kmeans_explained_var=0.99,
                          device=utils.CPU_DEVICE, spv_num_epochs=160,
                          drmm_num_epochs=4, drmm_n_layer=3, drmm_n_comp_per_layer=10,
                          seed=25041993):
    """
    p_feats, q_feats are torch.Tensor
    """
    t1 = time.time()
    if discretization_algo == 'kmeans_l1':
        if not FOUND_FAISS:
            print('Faiss or sklearn not found. Exiting')
            sys.exit(-1)
        p, q = cluster_feats(p_feats.detach().cpu().numpy(),
                             q_feats.detach().cpu().numpy(),
                             num_clusters=kmeans_num_clusters,
                             norm='l1', whiten=True, min_var=kmeans_explained_var,
                            seed=seed)
    elif discretization_algo == 'kmeans_l2':
        if not FOUND_FAISS:
            print('Faiss or sklearn not found. Exiting')
            sys.exit(-1)
        p, q = cluster_feats(p_feats.detach().cpu().numpy(),
                             q_feats.detach().cpu().numpy(),
                             num_clusters=kmeans_num_clusters,
                             norm='l2', whiten=False, min_var=kmeans_explained_var,
                             seed=seed)
    elif discretization_algo in ['spv', 'spreadingvectors', 'lattice']:
        if not FOUND_SPV:
            print('SpreadingVectors not found. Exiting')
            sys.exit(-1)
        num_epochs = (spv_num_epochs // 4) * 4 # make it divisible by 4
        p, q = train_spv_and_quantize(p_feats, q_feats,
                                      device=device,
                                      epochs=num_epochs, seed=seed)
        # p, q: (744,)
    elif discretization_algo == 'drmm':
        if not FOUND_DRMM:
            print('DRMM or tensorflow not found. Exiting')
            sys.exit(-1)
        p, q = train_drmm_and_quantize(
            p_feats.detach().cpu().numpy(), q_feats.detach().cpu().numpy(), seed=seed,
            nEpoch=drmm_num_epochs, nComponentsPerLayer=drmm_n_comp_per_layer, nLayers=drmm_n_layer,
        )
        # p, q: at most (drmm_n_comp_per_layer ** drmm_n_layer,)
    else:
        raise ValueError('Unknown discretization algo: ', discretization_algo)
    t2 = time.time()
    print('discretization time:', round(t2-t1, 2))
    metrics = get_mauve_score(p, q)
    return p, q, metrics


# PR metrics
def get_discretization_algo_name(
        discretization_algo='kmeans_l1', kmeans_num_clusters=100, kmeans_explained_var=0.99,
        device=utils.CPU_DEVICE, spv_num_epochs=160, seed=25041993,
        drmm_num_epochs=4, drmm_n_layer=3, drmm_n_comp_per_layer=10
):
    assert 0 < kmeans_explained_var < 1
    kmeans_args = f'{kmeans_num_clusters}_{kmeans_explained_var}' if kmeans_explained_var != 0.99 else kmeans_num_clusters
    if discretization_algo == 'kmeans_l1':
        name = f'kmeans_l1_{kmeans_args}'
    elif discretization_algo == 'kmeans_l2':
        name = f'kmeans_l2_{kmeans_args}'
    elif discretization_algo in ['spv', 'spreadingvectors', 'lattice']:
        name = 'spv'
    elif discretization_algo == 'drmm':
        name = f'drmm_{drmm_n_layer}_{drmm_n_comp_per_layer}'
    else:
        raise ValueError('Unknown discretization algo: ', discretization_algo)
    return name

##################
# Helper functions
##################
def cluster_feats(p, q, num_clusters,
                  norm='none', whiten=True, min_var=0.99,
                  niter=500, seed=0):
    """ p, q are numpy arrays"""
    assert 0 < min_var < 1
    print(f'seed = {seed}')
    assert norm in ['none', 'l2', 'l1', None]
    data1 = np.vstack([q, p])
    if norm in ['l2', 'l1']:
        data1 = normalize(data1, norm=norm, axis=1)
    pca = PCA(n_components=None, whiten=whiten, random_state=seed+1)
    pca.fit(data1)
    s = np.cumsum(pca.explained_variance_ratio_)
    idx = np.argmax(s >= min_var)  # last index to consider
    print(f'lower dimensionality = {idx}')
    data1 = pca.transform(data1)[:, :idx+1]
    # Cluster
    data1 = data1.astype(np.float32)
    d = data1.shape[1]
    t1 = time.time()
    kmeans = faiss.Kmeans(data1.shape[1], num_clusters, niter=niter, verbose=True,
                         nredo=5, update_index=True, seed=seed+2)
    kmeans.train(data1)
    _, labels = kmeans.index.search(data1, 1)
    labels = labels.reshape(-1)
    t2 = time.time()
    print('kmeans time:', round(t2-t1, 2))

    q_labels = labels[:len(q)]
    p_labels = labels[len(q):]

    q_bins = np.histogram(q_labels, bins=num_clusters,
                           range=[0, num_clusters], density=True)[0]
    p_bins = np.histogram(p_labels, bins=num_clusters,
                          range=[0, num_clusters], density=True)[0]
    return p_bins / p_bins.sum(), q_bins / q_bins.sum()


def kl_multinomial(p, q):
    assert p.shape == q.shape
    if np.logical_and(p != 0, q == 0).any():
        return np.inf
    else:
        idxs = np.logical_and(p != 0, q != 0)
        return np.sum(p[idxs] * np.log(p[idxs] / q[idxs]))

def get_mauve_score(p, q, mixture_weights=np.linspace(0, 1, 100), scaling_factor=5):
    angles = np.linspace(1e-6, np.pi / 2 - 1e-6, 25)
    mixture_weights = np.cos(angles)  # on an angular grid
    divergence_curve = [[0, np.inf]]  # extreme point
    for w in np.sort(mixture_weights):
        r = w * p + (1 - w) * q
        divergence_curve.append([kl_multinomial(q, r), kl_multinomial(p, r)])
    divergence_curve.append([np.inf, 0])  # other extreme point
    divergence_curve = np.exp(-scaling_factor * np.asarray(divergence_curve))
    mauve = sklearn.metrics.auc(divergence_curve[:, 0], divergence_curve[:, 1])
    return mauve
