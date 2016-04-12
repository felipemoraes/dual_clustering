import numpy as np
from scipy.spatial.distance import euclidean
from dual_optics import optics
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as data

sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}

def distance_in_out(vec1,vec2):
    return euclidean(vec1[0],vec2[1])


moons, _ = data.make_moons(n_samples=50, noise=0.05)
blobs, _ = data.make_blobs(n_samples=50, centers=[(-0.75,2.25), (1.0, 2.0)], cluster_std=0.25)
test_data = np.vstack([moons, blobs])
sample = [[data,data] for data in test_data]

#plt.scatter(test_data.T[0], test_data.T[1], color='b', **plot_kwds)

eps = 0.6
minpts = 10


model = optics(sample, eps, minpts, distance_in_out)
model.process()


print model.get_labels()
