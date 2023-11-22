
# KNN - K-Near Neighbour algorithm which group data to clusters
# 1 step: init first position of centroids
# 2 step: assign each date point to nearest centroid
# 3 step: calcultate average position by using assigned data and move centroids to this positions
# Repeate algorithm from 2nd step

import numpy as np

def knn_mean_run(X:np.array, K:int, iterations:int = 10):
    randIdxs = np.random.permutation(X.shape[0])
    centroids = X[randIdxs[:K]]
    cIndexes = np.zeros(X.shape[0], dtype="int")
    minDist = np.zeros(X.shape[0])
    for i in range(iterations):
        # Cendroid-based Idx = min((X[idx]-self.centroids)**2)
        for j in range(len(cIndexes)):
            dists = np.linalg.norm(X[j]-centroids, axis=1)
            minDist[j] = np.min(dists)
            cIndexes[j] = np.argmin(dists)
        # Calculate k-mean centroids of this subset
        for i in range(K):
            Kn = X[cIndexes == i]
            centroids[i] = np.mean(Kn, axis=0)
    costVal = np.mean(minDist)
    return centroids, cIndexes, costVal

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt

    src_img = plt.imread("MathML/img/KNN_s_bird.png")
    print("Shape of image: ", src_img.shape)
    input_data = src_img.reshape((-1, 3))
    print("Shape of input data: ", input_data.shape)

    K = 16
    saveCntrds = None
    saveIdxs = None
    saveCntrds, saveIdxs, saveCost = knn_mean_run(input_data, K)
    for i in range(5):
        cntrds, idxs, cost = knn_mean_run(input_data, K)
        print("Cost value: ", cost)
        if cost < saveCost:
            saveCntrds = cntrds
            saveIdxs = idxs
            saveCost = cost
    print("Best centroids: ", saveCntrds)
    print("Best cost: ", saveCost)
    X_compressed = saveCntrds[saveIdxs,:]
    X_compressed = X_compressed.reshape(src_img.shape)

    fig, ax = plt.subplots(1,2, figsize=(16,16))
    plt.axis('off')

    ax[0].imshow(src_img)
    ax[0].set_title('Original')
    ax[0].set_axis_off()


    # Display compressed image
    ax[1].imshow(X_compressed)
    ax[1].set_title('Compressed with %d colours'%K)
    ax[1].set_axis_off()
    plt.show()
    


