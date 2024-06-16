# import utilities as ut
import numpy as np
import os
import pandas as pd
import pickle
# import plotly
# import plotly.graph_objs as go

from pyAudioAnalysis.MidTermFeatures import directory_feature_extraction as dW

# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
CLASS_FOLDER = f'{CURRENT_DIR}/../data/musical_regions'

# name_1, name_2 = 'spectral_entropy_std', 'chroma_std_std'
# layout = go.Layout(
#     title='GR Folk Music Classification',
#     xaxis=dict(title=name_1,), yaxis=dict(title=name_2,))

if __name__ == '__main__':
    dataset = f'{CURRENT_DIR}/../data_exploration/preprocessed_dataset.csv'
    df = pd.read_csv(dataset)

    regions = np.sort(df['region'].dropna().unique())
    # regions = regions[-1:]
    print(regions)

    features = []
    # fMeanStd = []
    # plots = []
    fn = None

    # get features from folders (all classes)
    for region in regions:
        print(f'{CLASS_FOLDER}/{region}')
        f, _, feat = dW(f'{CLASS_FOLDER}/{region}', 2, 1, 0.1, 0.1)
        features.append(f)

        if fn is None:
            fn = feat

        # foo = np.array([f[:, feat.index(name_1)], f[:, feat.index(name_2)]]).T
        # fMeanStd.append(foo)

    # Save the array
    with open('features.pkl', 'wb') as f:
        pickle.dump(features, f)

    # Flatten the nested structure
    data = [list(sample) + [regions[class_idx]] for class_idx, samples in enumerate(features) for sample in samples]

    # Create column names
    num_features = len(features[0][0])
    column_names = fn + ['label']

    # Create the features dataframe
    df = pd.DataFrame(data, columns=column_names)
    df.to_csv(f'{CURRENT_DIR}/train.csv')

    # plot histograms for each feature and normalize
    # ut.plot_feature_histograms(
    #     features,
    #     fn,
    #     regions,
    # )

    # f = np.concatenate(fMeanStd, axis=0)

    # mean, std = f.mean(axis=0), np.std(f, axis=0)

    # arr_2_concat = []
    # for idx, val in enumerate(fMeanStd):
    #     print(val)
    #     fMeanStd[idx] = (val - mean) / std
    #     plots.append(go.Scatter(
    #         x=val[:, 0],
    #         y=val[:, 1],
    #         mode='markers',
    #         name=region,
    #         marker=dict(
    #             size=10,
    #             color='rgba(255, 182, 193, .9)',)
    #         )
    #     )
    #     if idx != 0:
    #         arr_2_concat.append(np.ones(val.shape[0]))
    #     else:
    #         arr_2_concat.append(np.zeros(val.shape[0]))

    # f = (f - mean) / std

    # get classification decisions for grid
    # y = np.concatenate(arr_2_concat)

    # cl = SVC(kernel='rbf', C=1)
    # cl.fit(f, y)

    # x_ = np.arange(f[:, 0].min(), f[:, 0].max(), 0.01)
    # y_ = np.arange(f[:, 1].min(), f[:, 1].max(), 0.01)
    # xx, yy = np.meshgrid(x_, y_)
    # Z = cl.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape) / 2
    # cs = go.Heatmap(
    #     x=x_,
    #     y=y_,
    #     z=Z,
    #     showscale=False,
    #     colorscale=[
    #         [0, 'rgba(255, 182, 193, .3)'],
    #         [0.5, 'rgba(100, 182, 150, .3)'],
    #         [1, 'rgba(100, 100, 220, .3)']
    #     ]
    # )

    # plotly.offline.plot(go.Figure(data=plots, layout=layout),
    #                     filename='temp2.html', auto_open=True)
