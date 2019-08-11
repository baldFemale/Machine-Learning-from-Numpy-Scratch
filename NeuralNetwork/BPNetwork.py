import numpy as np
import matplotlib.pyplot as plt


class BPNetwork:

    def __init__(self):
        self.weight = {}
        self.theta = {}
        self.weight_layer = []
        self.theta_layer = []
        self.hidden_value = {}
        pass

    def activate_func(self, X):
        return 1.0/(1+np.exp(-X))

    def forward(self, X):
        for i in range(len(self.weight_layer)):
            self.hidden_value[self.weight_layer[i]] = np.mean(X,axis=0).reshape(1,-1)
            # print(X)
            X = X @ self.weight[self.weight_layer[i]]
            X = X - self.theta[self.theta_layer[i]]
            # print(X)
            X = self.activate_func(X)
            # print(X)
        return X

    def backward(self, Y, Y_hat, learning_rate):
        # print(self.theta[self.theta_layer[0]])
        # print(self.weight[self.weight_layer[0]])
        # print(self.weight[self.weight_layer[0]].shape)
        # print(Y_hat)
        # print(Y)
        g = (Y_hat @ (1-Y_hat).T) @ (Y-Y_hat)
        # print(g)
        self.theta[self.theta_layer[-1]] -= learning_rate*g
        self.weight[self.weight_layer[-1]] += learning_rate*(self.hidden_value[self.weight_layer[-1]].T @ g)

        for i in range(len(self.weight_layer)-2,-1,-1):
            g = self.hidden_value[self.weight_layer[i+1]] @ (1-self.hidden_value[self.weight_layer[i+1]]).T @ \
                g @ self.weight[self.weight_layer[i+1]].T
            self.theta[self.theta_layer[i]] -= learning_rate*g
            self.weight[self.weight_layer[i]] += learning_rate*(self.hidden_value[self.weight_layer[i]].T @ g)

    def fit(self, X, Y, layers, hidden_number, iteration, batch_size, learing_rate):
        n_sample, n_feature = X.shape
        xavier = 1/np.sqrt(n_feature)
        unique_value = np.unique(Y)
        out_put_size = len(unique_value)

        self.weight["input_hidden_1"] = np.random.uniform(-xavier,xavier,[n_feature,hidden_number])
        self.weight_layer.append("input_hidden_1")
        for i in range(layers-1):
            self.weight["hidden_"+str(i+1)+"_hidden_"+str(i+2)] = np.random.uniform(-xavier,xavier,
                                                                                    [hidden_number,hidden_number])
            self.weight_layer.append("hidden_"+str(i+1)+"_hidden_"+str(i+2))
        self.weight["hidden_"+str(layers)+"_output"] = np.random.uniform(-xavier,xavier,[hidden_number,out_put_size])
        self.weight_layer.append("hidden_"+str(layers)+"_output")

        for i in range(1,layers+1):
            self.theta["hidden_"+str(i)] = np.zeros(shape=(1,hidden_number))
            self.theta_layer.append("hidden_"+str(i))
        self.theta["output"] = np.zeros(shape=(1,out_put_size))
        self.theta_layer.append("output")

        figure = plt.figure()
        for epoch in range(iteration):
            if epoch%500==0:
                print("in the epoch {}".format(epoch))
            for offset in range(0,n_sample,batch_size):
                # print(offset)
                x = X[offset:offset+batch_size]
                y = Y[offset:offset+batch_size]
                accumulate_y = np.zeros(shape=(1, out_put_size))

                for v in np.unique(y):
                    accumulate_y[0][int(v)] = len(y[y == v])
                res = np.mean(self.forward(x), axis=0).reshape(1,-1)
                accumulate_y /= len(y)
                self.backward(accumulate_y,res,learing_rate)
            plt.scatter(epoch,np.sum(res-accumulate_y),)
        plt.show()

    def predict(self, X, Y):
        res = [np.argmax(self.forward(sample),axis=1) for sample in X]
        print(sum(res[i]==Y[i] for i in range(len(res)))/len(res))
        pass
