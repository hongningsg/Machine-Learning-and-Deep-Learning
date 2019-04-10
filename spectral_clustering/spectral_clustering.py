import numpy as np
import nltk
import random

class spectral_clustering:
    def __init__(self, num_means=2, distance=nltk.cluster.util.euclidean_distance, repeats=50, normalise=True,rng=random.Random(10)):
        self.num_means = num_means
        self.distance = distance
        self.repeats = repeats
        self.normalise = normalise
        self.rng = rng

    def _kmeans(eigenvectors):
        kmeans_ = nltk.cluster.KMeansClusterer(self.num_means,
                                               self.distance,
                                               self.repeats,
                                               self.normalise,
                                               self.rng)
        clusters = kmeans_.cluster(eigenvectors, assign_clusters=True)
        c1 = []
        c2 = []
        index = 0
        for i in clusters:
            if i == 0:
                c1.append(one_to_one[index])
            else:
                c2.append(one_to_one[index])
            index += 1
        clusters = [c1,c2]
        eigenvectors = np.matrix(eigenvectors)
        return eigenvectors, clusters

    def _get_vector(L):
        w, v = np.linalg.eig(L)
        col1 = np.array(v[:,[1,2]])
        col1 = np.insert(col1,2,w,axis=1)
        return np.array(col1[:,[0,1]])

    def _diag_matrix(G):
        matrix_size = len(G)
        #build matrix
        mat = []
        for i in range(matrix_size):
            row = [0]*matrix_size
            mat.append(row)
        D = np.array(mat)
        W = np.array(mat)

        #convert graph
        node_Name = []
        degree = {}
        for edge in G.edges:
            node_Name.append(edge[0])
            node_Name.append(edge[1])
            try:
                degree[edge[0]] += 1
            except:
                degree[edge[0]] = 1
            try:
                degree[edge[1]] += 1
            except:
                degree[edge[1]] = 1
        node_Name = list(set(node_Name))
        node_Name.sort()
        one_to_one = {}
        reverse_find = {} 
        index = 0
        for node in node_Name:
            one_to_one[index] = node
            reverse_find[node] = index
            index += 1

        #BUILD DEGREE MATRIX
        for i in one_to_one:
            D[i][i] = degree[one_to_one[i]]
            
        #BUILD Adjacency matrix
        for edge in G.edges:
            W[reverse_find[edge[0]]][reverse_find[edge[1]]] = 1
            W[reverse_find[edge[1]]][reverse_find[edge[0]]] = 1

        return D - W

    def fit(X):
        L = self._diag_matrix(X)
        vector = self._get_vector(L)
        self.weights, self.clusters = self._kmeans(vector)

    def components():
        return self.clusters

    def weights():
        return self.weights
    
    
