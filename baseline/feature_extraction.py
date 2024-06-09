import utilities as ut
import numpy as np
import os
import pandas as pd
import plotly
import plotly.graph_objs as go

from pyAudioAnalysis.MidTermFeatures import directory_feature_extraction as dW
from sklearn.metrics import (
    accuracy_score, auc,
    confusion_matrix, f1_score,
    matthews_corrcoef, precision_score,
    recall_score, roc_curve
)
from sklearn.svm import SVC

# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
CLASS_FOLDER = f'{CURRENT_DIR}/../data/musical_regions'

name_1, name_2 = 'spectral_entropy_std', 'chroma_std_std'
layout = go.Layout(
    title='GR Folk Music Classification',
    xaxis=dict(title=name_1,), yaxis=dict(title=name_2,))

if __name__ == '__main__':
    df = pd.read_csv(f'{CURRENT_DIR}/../data_exploration/preprocessed_dataset.csv')
    regions = np.sort(df['region'].dropna().unique())
    regions = regions[-2:]
    print(regions)

    features = []
    fMeanStd = []
    fns = []
    plots = []

    # get features from folders (all classes)
    for region in regions:
        print(f'{CLASS_FOLDER}/{region}')
        f, _, fn = dW(f'{CLASS_FOLDER}/{region}', 2, 1, 0.1, 0.1)
        features.append(f)
        fns.append(fn)

        foo = np.array([f[:, fn.index(name_1)], f[:, fn.index(name_2)]]).T
        fMeanStd.append(foo)

    # plot histograms for each feature and normalize
    ut.plot_feature_histograms(
        features,
        fns[0],
        regions,
    )

    f = np.concatenate(fMeanStd, axis=0)

    mean, std = f.mean(axis=0), np.std(f, axis=0)

    arr_2_concat = []
    for idx, val in enumerate(fMeanStd):
        print(val)
        fMeanStd[idx] = (val - mean) / std
        plots.append(go.Scatter(
            x=val[:, 0],
            y=val[:, 1],
            mode='markers',
            name=region,
            marker=dict(
                size=10,
                color='rgba(255, 182, 193, .9)',)
            )
        )
        if idx != 0:
            arr_2_concat.append(np.ones(val.shape[0]))
        else:
            arr_2_concat.append(np.zeros(val.shape[0]))

    f = (f - mean) / std

    # get classification decisions for grid
    y = np.concatenate(arr_2_concat)

    cl = SVC(kernel='rbf', C=1)
    cl.fit(f, y)

    x_ = np.arange(f[:, 0].min(), f[:, 0].max(), 0.01)
    y_ = np.arange(f[:, 1].min(), f[:, 1].max(), 0.01)
    xx, yy = np.meshgrid(x_, y_)
    Z = cl.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape) / 2
    cs = go.Heatmap(
        x=x_,
        y=y_,
        z=Z,
        showscale=False,
        colorscale=[
            [0, 'rgba(255, 182, 193, .3)'],
            [0.5, 'rgba(100, 182, 150, .3)'],
            [1, 'rgba(100, 100, 220, .3)']
        ]
    )

    plotly.offline.plot(go.Figure(data=plots, layout=layout),
                        filename='temp2.html', auto_open=True)

    # METRICS
    accuracy = accuracy_score(yy, Z)
    precision = precision_score(yy, Z, average='macro')
    recall = recall_score(yy, Z, average='macro')
    f1 = f1_score(yy, Z, average='macro')
    matthews = matthews_corrcoef(yy, Z)
