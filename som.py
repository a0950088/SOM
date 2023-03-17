import numpy as np
from math import exp

class SOM:
    def __init__(self, data, epoch): # 初始化SOM所需參數
        self.inputdata = self.normalized(data[:,:-1])
        self.eoutputdata = data[:,-1:]
        self.w_row = 10
        self.w_col = 10
        self.w = np.random.uniform(low=-1.0, high=1.0, size=(self.w_row,self.w_col,self.inputdata.shape[1]))
        self.epoch = epoch
        self.lr = 0.9
        self.radius = 10
    
    def normalized(self, data): # 將輸入超過二維以上的資料正規化
        if data.shape[1] > 2:
            for d in range(data.shape[0]):
                maxd = max(data[d])
                mind = min(data[d])
                if maxd == mind:
                    data[d] = data[d]
                else:
                    data[d] = (data[d]-mind)/(maxd-mind)
        return data
        
    def distance(self, x1, x2): # 計算歐基里德距離
        return sum(abs(x1-x2)**2)**0.5
    
    def updatelr(self, t): # 隨著迭代次數更新學習率
        return self.lr*exp(-(t/self.epoch))
    
    def updateradius(self, t): # 隨著迭代次數更新鄰近區域半徑
        return self.radius*exp(-(t/self.epoch))
    
    def updatew(self, x, winner, new_lr, new_radius): # 更新鍵結值
        k = np.zeros((self.w_row,self.w_col,1))
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                k[i][j][0] = exp(-((abs(winner[0]-i)**2+abs(winner[1]-j)**2)/(2*(new_radius**2))))
        self.w = self.w + new_lr*k*(x-self.w)
        
    def find_winner(self, x): # 比較輸入與每個鍵結值的距離尋找得勝的神經元
        min_distance = -1.0
        win_index = np.zeros(2)
        for i in range(self.w_row):
            for j in range(self.w_col):
                new_distance = self.distance(x,self.w[i][j])
                if min_distance==-1.0 or min_distance>new_distance:
                    min_distance = new_distance
                    win_index[0] = i
                    win_index[1] = j
        return win_index
    
    def train(self): # 訓練SOM神經網路
        for t in range(self.epoch):
            print("epoch:", t)
            random_index = np.random.choice(self.inputdata.shape[0],self.inputdata.shape[0],replace=False)
            new_lr = self.updatelr(t)
            new_radius = self.updateradius(t)
            for r in random_index:
                winner = np.copy(self.find_winner(self.inputdata[r]))
                self.updatew(self.inputdata[r], winner, new_lr, new_radius)
            print("learnrate: ",new_lr," radius: ",new_radius)
        print(self.w)
        classifier = []
        for i in range(self.w_row):
            for j in range(self.w_col):
                mind = []
                for d in self.inputdata:
                    mind.append(self.distance(d,self.w[i][j]))
                classifier.append(self.eoutputdata[mind.index(min(mind))][0])
        return classifier