import networkx as nx
import pandas as pd
import math

class LinkRecommendation():
    trainG = None
    testG = None
    sim = None  # similarity matrix
    maxl=2
    beta = 0.1 
    local_mds = ["CN","JA","AA","PA"]
    
    def __init__(self, dataPath):
        df = pd.read_csv(dataPath, sep=' ', header=None)
        test_df = df.sample(frac=0.2, random_state=10)
        train_df = df.drop(test_df.index)

        self.trainG = nx.Graph()
        self.testG = nx.Graph()

        train_data = train_df.values
        test_data = test_df.values

        for i in range(len(train_data)):
            self.trainG.add_edge(train_data[i][0], train_data[i][1])

        for i in range(len(test_data)):
            self.testG.add_edge(test_data[i][0], test_data[i][1])

    def katz(self):
        train_n = len(self.trainG.nodes)
        # nodes = list(self.trainG.nodes)

        sim = [[0 for i in range(train_n)] for j in range(train_n)]
        for i in range(train_n):

            if i not in self.trainG.nodes:
                continue

            for j in range(i+1, train_n):
                if j in self.trainG.nodes:		# TODO: check if we need this
                    sim[i][j] = sim[j][i] = self.katz_similarity(i,j)
        print('finished')
        self.sim = sim
        self.precision_and_recall()


    def local_methods(self, method):
        train_n = len(self.trainG.nodes)
        nodes = list(self.trainG.nodes)

        sim = [[0 for i in range(train_n)] for j in range(train_n)]
        for i in range(train_n):
            for j in range(train_n):
                if i != j and i in self.trainG.nodes and j in self.trainG.nodes:
                    sim[i][j] = self.local_methods_calSimilarity(nodes[i], nodes[j], method)

        self.sim = sim
        self.precision_and_recall()

    def local_methods_calSimilarity(self, i, j, method):
        if method == "CN":
            return len(set(self.trainG.neighbors(i)).intersection(set(self.trainG.neighbors(j))))

        elif method == "JA":
            num_JA=len(set(self.trainG.neighbors(i)).intersection(set(self.trainG.neighbors(j))))/float(len(set(self.trainG.neighbors(i)).union(set(self.trainG.neighbors(j)))))
            return num_JA

        elif method == "AA":
            num_AA= sum([1.0/math.log(self.trainG.degree(v)) for v in set(self.trainG.neighbors(i)).intersection(set(self.trainG.neighbors(j)))])
            return num_AA

        elif method == "PA":
            num_PA=(self.trainG.degree(i) * self.trainG.degree(j))
            return num_PA

    def precision_and_recall(self,k=5):
        precision = recall = c = 0
        for person in self.testG.nodes:
            if person in self.trainG.nodes:
                testNeighbors = [n for n in self.testG.neighbors(person)]
                if len(testNeighbors) < k:
                    k = len(testNeighbors)
                top_k = set(self.top_k_rec(person,k))
                precision += len(top_k.intersection(testNeighbors)) / float(k)
                recall += len(top_k.intersection(testNeighbors)) / float(len(testNeighbors))
                c += 1
        print("Precision is : " + str(precision / c))
        print("Recall is : " + str(recall / c))


    def top_k_rec(self, person,k):
        nodes = list(self.trainG.nodes)
        indexPerson = nodes.index(person)
        indexRecs = sorted(filter(lambda indexX: indexPerson!=indexX and not self.trainG.has_edge(person,nodes[indexX]),
                              range(len(self.sim[indexPerson]))),
                       key=lambda indexX: self.sim[indexPerson][indexX],reverse=True)[0:k]
        recFriends = [nodes[i] for i in indexRecs]
        return recFriends


    def build_model(self, method):
        if method in local_mds:
            self.local_methods(method)
        elif method == "katz":
            self.katz()
        else:
            print("invalid input")


    def katz_similarity(self,i,j):
        l = 1
        neighbors_i = [n for n in self.trainG.neighbors(i)]
        score = 0
        
        while l <= self.maxl:
            numberOfPaths = neighbors_i.count(j)
            if numberOfPaths > 0:
                score += (self.beta**l)*numberOfPaths
            neighborsForNextLoop = []
            for k in neighbors_i:
                neighbors_k = [n for n in self.trainG.neighbors(k)]
                neighborsForNextLoop += neighbors_k
            neighbors_i = neighborsForNextLoop
            l += 1
    
        return score
        

# Local methods: CN(common_neighbors) JA(jaccard) AA(adamic_adar) PA(preferential_attachment)
# katz
# xgboost

#dataPath = 'data/test.txt'
dataPath = 'data/facebook_combined.txt'
LP = LinkRecommendation(dataPath)
LP.build_model('CN')
print(LP.top_k_rec(0,5))