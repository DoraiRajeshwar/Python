import numpy as np

#Defining class matrix 
class matrix:
    
#Function that allows loading of file
    def __init__(self, filename):
        self.array_2d = np.empty((0, 0))
        self.load_from_csv(filename)
        self.standardise()
        self.rows = self.array_2d.shape[0]
        self.columns = self.array_2d.shape[1]

    def load_from_csv(self, filename):
        self.array_2d = np.loadtxt(filename, delimiter=',')
        return self.array_2d
    
#Standerdising the array
    def standardise(self):
        for x in range(len(self.array_2d)):
            for y in range(len(self.array_2d[x])):
                avg = np.average(self.array_2d, axis=0)
                maxi = np.max(self.array_2d, axis=0)
                mini = np.min(self.array_2d, axis=0)
                self.array_2d[x, y] = (self.array_2d[x, y]-avg[y])/(maxi[y]-mini[y])
        return self.array_2d
    
#Calculating Weighted Euclidean distance
    def get_distance(self, other_matrix, w, beta):
        distance = []
        for row in self.centroids:
            rowval = 0
            for i in range(len(row)):
                rowval += w[i]**beta * ((other_matrix[i] - row[i]))**2
            distance.append(rowval)
        return np.array(distance)
    
#Getting the counts of elements
    def get_count_frequency(self):
        unique, frequency = np.unique(self.S, return_counts=True)
        return dict(zip(unique, frequency))

#Initiating random weights
def get_initial_weights(column):
    w = np.random.random(column)
    w /= np.sum(w)
    return w

#Updating the centroid with nearby matrix by taking mean of it
def get_centroids(datamatrix, matrix_s, K):
    centroid = []
    for cluster in range(K):
        rowid = []
        for i in range(len(matrix_s)):
            if int(matrix_s[i][0]) == cluster:
                rowid.append(datamatrix[i, :])
        centroid.append(np.mean(np.array(rowid), axis=0))
    return np.array(centroid)

#Creating cluster with weights and centroid
def get_groups(m, cluster, betaval):
    K = cluster
    beta = betaval
    save_s = []
    m.w = get_initial_weights(m.columns)
    m.S = np.zeros((m.array_2d.shape[0], 1))
    m.centroids = np.random.permutation(m.array_2d)[:K]

    while True:
        save_s = m.S.copy()
        count = 0
        for onerow in m.array_2d:
            dist = m.get_distance(onerow, m.w, beta)
            m.S[count] = (np.where(dist == min(dist))[0][0])
            count = count+1
            
        if not(save_s == m.S).all():
            #Updating Weights and Centroid
            m.centroids = get_centroids(m.array_2d, m.S, K)
            m.weights = get_new_weight(m.array_2d, m.centroids, m.S, beta)
        else:
            return m

#Calculating and updating Weights
def get_new_weight(datamatrix, centroids, matrix_s, beta):

    n = datamatrix.shape[0]
    m = datamatrix.shape[1]
    w = np.zeros((1, m))
    K = centroids.shape[0]
    delta= np.zeros((1, m))
    for k in range(K):
        for i in range(n):
            if(matrix_s[i, :] == k):
                delta += (datamatrix[i, :] - centroids[k, :])**2
    for j in range(m):
        if delta[0, j] == 0:
            w[0, j] = 0
        else:
            value = 0
            for t in range(m):
                value += (delta[0, j]/delta[0, t])**(1/(beta-1))
            w[0, j] = value
    return w

def run_test():
    m = matrix('Data.csv')
    for k in range(2, 5):
        for beta in range(11, 25):
            S = get_groups(m, k, beta/10)
            print(str(k)+'-'+str(beta)+'='+str(S.get_count_frequency()))

#Main function
if __name__ == '__main__':

    run_test()


