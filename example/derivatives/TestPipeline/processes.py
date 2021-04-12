from nltools import Design_Matrix
import nltools
# from nltools.stats import zscore
import pandas as pd
import sklearn
import networkx as nx


def denoise(pipeline, sub, data, global_spike_cutoff=3, diff_spike_cutoff=3):

    print(f"...Denoising sub {sub}")
    # csf = zscore(pd.DataFrame(data.extract_roi(mask=self.masks["csf"]).T, columns=['csf']))
    csf = pd.DataFrame(data.extract_roi(
        mask=pipeline.masks["csf"]).T, columns=['csf'])

    spikes = data.find_spikes(global_spike_cutoff, diff_spike_cutoff)
    covariates = pd.read_csv(pipeline.layout._get_unique(
        subject=sub, scope='derivatives', extension='.tsv').path, sep='\t')
    mc = covariates[['trans_x', 'trans_y',
                     'trans_z', 'rot_x', 'rot_y', 'rot_z']]
    mc_cov = pipeline.make_motion_covariates(mc, pipeline.tr)
    dm = Design_Matrix(pd.concat([csf, mc_cov, spikes.drop(
        labels='TR', axis=1)], axis=1), sampling_freq=1/pipeline.tr)
    dm = dm.add_poly(order=2, include_lower=True)

    data.X = dm
    stats = data.regress()

    return stats['residual']


def smooth(pipeline, sub, data, fwhm=6):
    return data.smooth(fwhm)


def EVcentrality(pipeline, sub, data, adjacency_treshold=0.6, mask="rois"):
    rois = pipeline.load_data(sub, denoise={}).extract_roi(
        mask=pipeline.masks[mask])
    roi_corr = 1 - \
        sklearn.metrics.pairwise_distances(rois, metric='correlation')
    a = nltools.data.Adjacency(roi_corr, matrix_type='similarity', labels=[
                               x for x in range(rois.shape[0])])
    a_thresholded = a.threshold(upper=adjacency_treshold, binarize=True)
    G = a_thresholded.to_graph()
    return nltools.mask.roi_to_brain(pd.Series(nx.eigenvector_centrality(G)), nltools.mask.expand_mask(pipeline.masks[mask]))
