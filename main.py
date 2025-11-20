import random

import numpy as np
from sklearn.manifold import TSNE , SpectralEmbedding
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import tensorflow as tf
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.sparse import csr_matrix

def compute_similarities(data):
    std = np.std(data)
    # S = np.zeros((data.shape[0],data.shape[0]))
    # for i in range(S.shape[0]):
    #     for j in range(S.shape[1]):
    #         d = data[i] - data[j]
    #         d2 = d.T.dot(d)
    #         S[i,j] = np.exp(-d2/(2*(std**2)))
    dd = np.asarray([[[x for _ in range(data.shape[0])] for x in y] for y in data])
    diff = np.asarray([x.T - data for x in dd])
    ddd = np.asarray([[d.T.dot(d) for d in d1] for d1 in diff])
    S = np.exp(-ddd / (2 * (std ** 2)))
    return S


def compute_label_adjency_matrix(labels):
    S = np.zeros((len(labels), len(labels)))
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if labels[i] == labels[j]:
                S[i, j] = 1
    return S

def MySpectralClustering(data,n_clusters,labels=None,only_use_labels = False):
    std = np.std(data)
    print("Generating Adjency Matrix")
    if labels is None:
        D = compute_similarities(data)
    else:
        if not only_use_labels:
            S1 = compute_similarities(data)
            S2 = compute_label_adjency_matrix(labels)
            D = 0.5*S1 + 0.5*S2
        if only_use_labels:
            D = compute_label_adjency_matrix(labels)

    A = np.zeros((data.shape[0],data.shape[0]))
    for i in range(D.shape[0]):
        A[i,i] = sum(D[:][i]/D.shape[0])
    L = A - D
    print("Eigen analysis of Laplacian Matrix")
    eigvals , eigvectors = np.linalg.eig(L)
    eigvectors = np.asmatrix(eigvectors,dtype="float32")

    #plt.plot(range(len(eigvals)),eigvals)
    plt.scatter(range(len(eigvals)),eigvals)
    if labels is None:
        filename = "eigenvalues_not_use_labels"
    elif only_use_labels:
        filename = "eigenvalues_only_use_labels"
    else:
        filename = "eigenvalues_use_labels"
    plt.savefig(f"figs\\{dataset_name}\\{filename}")
    plt.close()


    k_eigvect = eigvectors.T
    V = k_eigvect[1:n_clusters+1].T
    clustering = KMeans(n_clusters=n_clusters,random_state=0).fit(V)
    return clustering , V

def clustering_statistics(clustering):
    plt.show()
    labels_dist = {}
    for i in np.unique(clustering.labels_):
        labels_dist[i] = list(np.zeros(n_clusters,dtype="int8"))

    i = 0
    for img in train_img[:size]:
        labels_dist[clustering.labels_[i]][train_label[i]] += 1
        i += 1

    cluster_class_purity = {}
    for l in labels_dist.keys():
        cluster_class_purity[l] = max(labels_dist[l]) / sum(labels_dist[l])


    cluster_max_class = {}
    for l in labels_dist.keys():
        cluster_max_class[l] = np.argmax(labels_dist[l])


    class_dist = {}
    for i in np.unique(train_label):
         class_dist[i] = list(np.zeros(n_clusters, dtype="int8"))
    i = 0
    for img in train_img[:size]:
        class_dist[train_label[i]][clustering.labels_[i]] += 1
        i += 1

    class_purity = {}
    for l in class_dist.keys():
        class_purity[l] = max(class_dist[l]) / sum(class_dist[l])

    max_class = {}
    for l in class_dist.keys():
         max_class[l] = np.argmax(class_dist[l])

    max_class_number = {}
    for l in class_dist.keys():
        max_class_number[l] = class_dist[l][np.argmax(class_dist[l])]

    return labels_dist,cluster_class_purity,cluster_max_class,class_dist,class_purity,max_class,max_class_number

def assign_class_to_clusters(labels_dist,cluster_max_class):
    cluster_label = list(range(10))
    for cl in list(range(10)):
        cluster_label[cl] = cluster_max_class[cl]
    for cl in list(range(10)):
        cluster_dist = list(labels_dist[cl])
        while cluster_label[cl] in cluster_label[:cl]:
            cluster_dist[cluster_label[cl]] = -1
            cluster_label[cl] = np.argmax(cluster_dist)
    return cluster_label

def assign_class_to_clusters_perfect_matching(label_dist,side="row"):
    adjency_matrix = np.zeros((len(label_dist),len(label_dist)))
    for cluster in label_dist:
        adjency_matrix[cluster] = label_dist[cluster]
    graph = csr_matrix(adjency_matrix)
    return maximum_bipartite_matching(graph, perm_type=side)



def evaluate_clustering_with_classes(data,labels,centers,cluster_labels):
    lab = []
    for d in data:
        d=np.resize(d,(10))
        dists = [(d-c).T.dot(d-c) for c in centers]
        mindist = np.argmin(dists)
        lab.append(cluster_labels[mindist])
    correct = 0
    for i in range(len(labels)):
        if lab[i] == labels[i]:
            correct+=1
    return correct / len(labels)

for dataset,linear_dims,dataset_name in [(tf.keras.datasets.mnist,28*28,"mnist"),(tf.keras.datasets.cifar10,32*32*3,"cifar")]:
    print("Dataset : ",dataset_name)
    (train_img_original, train_label), (test_img_original, test_label) = dataset.load_data()

    size = 5000

    train_img = np.resize(train_img_original,(60000,linear_dims))/255
    test_img = np.resize(test_img_original,(10000,linear_dims))/255
    train_label = np.asarray(train_label,dtype="int8")
    test_label = np.asarray(test_label,dtype="int8")

    if dataset == tf.keras.datasets.cifar10:
        if dataset==tf.keras.datasets.cifar10:
            label_indexed = []
            for label in train_label[:size]:
                label_indexed.append(label[0])
        train_label = label_indexed

    print("Embedding data")
    embedded_mnist = SpectralEmbedding(n_components=2).fit_transform(train_img[:size])
    #print("Plotting")

    colors = [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(0.5,0,0),(0.5,1,0),(0.5,1,1),(0.5,0.5,1)]


    i = 0
    for point in embedded_mnist:
        color = colors[train_label[i]]
        plt.scatter(point[0],point[1],color=color)
        i+=1
    plt.savefig(f"figs\\{dataset_name}\\embedded2D")
    plt.close()

    for use_labels in ("not_use_labels","only_use_labels","use_labels"):
        print("Using Spectral Analysis",use_labels)
        n_clusters = 10
        if use_labels == "use_labels":
            clustering , embedded_tsne_mnist = MySpectralClustering(embedded_mnist,n_clusters=10,labels=train_label[:size])
        elif use_labels == "not_use_labels":
            clustering, embedded_tsne_mnist = MySpectralClustering(embedded_mnist, n_clusters=10)
        elif use_labels == "only_use_labels":
            clustering, embedded_tsne_mnist = MySpectralClustering(embedded_mnist, n_clusters=10 ,labels=train_label[:size],only_use_labels=True)

        colors = []
        for j in range(n_clusters):
            while True:
                r,g,b = random.random() , random.random() , random.random()
                if (r,g,b) not in colors:
                    colors.append((r,g,b))
                    break

        i=0
        for point in embedded_mnist:
            color = colors[clustering.labels_[i]]
            plt.scatter(point[0],point[1],color=color)
            i+=1

        plt.title(f"{dataset_name} classified embedded {use_labels}")
        plt.savefig(f"figs\\{dataset_name}\\classified_embedded_{use_labels}")
        plt.close()


        labels_dist,cluster_class_purity,cluster_max_class,class_dist,class_purity,max_class,max_class_number= clustering_statistics(clustering)
        print("Cluster class distribution : ", labels_dist)
        print("Cluster class purity : ", cluster_class_purity)
        print("Cluster max class : ", cluster_max_class)
        print("Class distribution : ", class_dist)
        print("Class purity : ", class_purity)
        print("Max class : ", max_class)
        print("Max class number : ", max_class_number)

        cluster_labels = assign_class_to_clusters(labels_dist, cluster_max_class)
        print("Cluster labels : ",cluster_labels)

        accuracy = evaluate_clustering_with_classes(embedded_tsne_mnist, train_label[:size], clustering.cluster_centers_, cluster_labels)
        print("Accuracy :",accuracy)

        with open("statistics.txt","a") as file:
            file.write(f"{dataset_name} {use_labels}\n")
            file.write(f"Cluster class distribution :  {labels_dist}\n")
            file.write(f"Cluster class purity : {cluster_class_purity}\n")
            file.write(f"Cluster max class : {cluster_max_class}\n")
            file.write(f"Class distribution : {class_dist}\n")
            file.write(f"Class purity :  {class_purity}\n")
            file.write(f"Max class : {max_class}\n")
            file.write(f"Max class number : {max_class_number}\n")
            file.write(f"Cluster labels : {cluster_labels}\n")
            file.write(f"Accuracy : {accuracy}\n")