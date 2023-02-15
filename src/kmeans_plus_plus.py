from functools import partial
import random
import numpy as np
import torch

FLOAT_MAX = 1e10

def getDist(X, Y):
    Z = torch.sum((X - Y) ** 2)
    return Z

def getNearestCenter(X, xID, centerIDs):
    minDistance = FLOAT_MAX
    for centerID in centerIDs:
        dis = getDist(X[xID], X[centerID])
        minDistance = min(minDistance, dis)
    return minDistance

def initialize(X, num_clusters):
    """
    <<<<<< Modified to K-means++ >>>>>>
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    clusterCenterGroup = [random.randint(0, X.size(0) - 1)]
    distanceGroup = torch.zeros(X.size(0))
    sum = 0.0
    for index in range(1, num_clusters):
        # maxx, maxID = 0, 0
        for i in range(X.size(0)):
            distanceGroup[i] = getNearestCenter(X, i, clusterCenterGroup[:index])
            # if distanceGroup[i] > maxx:
            #     maxx = distanceGroup[i]
            #     maxID = i
            sum += distanceGroup[i]
        # clusterCenterGroup.append(maxID)
        sum *= random.random()
        for i in range(X.size(0)):
            sum -= distanceGroup[i]
            if sum < 0 and not i in clusterCenterGroup:
                clusterCenterGroup.append(i)
                break
    if len(clusterCenterGroup) < num_clusters:
        indices = np.random.choice(X.size(0), num_clusters, replace=False)
    else:
        indices = np.array(clusterCenterGroup)
    initial_state = X[indices]
    return initial_state


def kmeans(
        X,
        num_clusters,
        cluster_centers=[],
        tol=1e-5,
        iter_limit=100,
        device=torch.device('cpu'),
):
    
    pairwise_distance_function = partial(pairwise_distance, device=device)
    
    # convert to float
    X = X.float()

    # initialize
    if type(cluster_centers) == list: 
        initial_state = initialize(X, num_clusters)
    else:
        initial_state = cluster_centers
        dis = pairwise_distance_function(X, initial_state)
        choice_points = torch.argmin(dis, dim=0)
        initial_state = X[choice_points]
        initial_state = initial_state.to(device)
    
    assert initial_state.size(0) == num_clusters
    
    X = X.to(device)
    initial_state = initial_state.to(device)
    iteration = 0
    
    while True:

        dis = pairwise_distance_function(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)

            if selected.shape[0] == 0:
                selected = X[torch.randint(len(X), (1,))]

            initial_state[index] = selected.mean(dim=0)

        # Calculate Loss
        # one_hot = torch.zeros(X.size(0), num_clusters).to(device)
        # one_hot.scatter_(1, choice_cluster.unsqueeze(1), 1)
        # loss = torch.sum(one_hot * dis)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        if center_shift ** 2 < tol:
            break
        if iter_limit != 0 and iteration >= iter_limit:
            break
    return choice_cluster.cpu(), initial_state.cpu()


def pairwise_distance(data1, data2, device=torch.device('cpu')):
    data1, data2 = data1.to(device), data2.to(device)
    dis = torch.cdist(data1, data2, p=2)
    return dis


