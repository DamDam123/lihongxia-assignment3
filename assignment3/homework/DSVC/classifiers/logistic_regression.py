import numpy as np
import random
import math

class LogisticRegression(object):

    def __init__(self):
        self.w = None
        self.k = None


    def sigmoid(self,X,theta):
        z = X.dot(theta.T)
        y_pre = 1./(1+np.exp(-z))
        return y_pre


    def loss(self, X_batch, y_batch):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
        data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """

        #########################################################################
        # TODO:                                                                 #
        # calculate the loss and the derivative                                 #
        #########################################################################
        J = (-1) * (np.sum(y_batch.dot(np.log(self.sigmoid(X_batch,self.w))) + (1-y_batch).dot(np.log(1 - self.sigmoid(X_batch,self.w)))))/ X_batch.shape[0]
        der = X_batch.T.dot(self.sigmoid(X_batch,self.w) - y_batch)/ X_batch.shape[0]
        loss = tuple([J,der])
        return loss

        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def train(self, X, y, learning_rate=1e-3, num_iters=100,
            batch_size=200, verbose=True):

        """
        Train this linear classifier using stochastic gradient descent（随机梯度下降）.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing（循环次数）
        - batch_size: (integer) number of training examples to use at each step.(每次循环的数据集大小)
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.（输出每次的损失函数）
        """
        num_train, dim = X.shape[0],X.shape[1]

        if self.w is None:
            self.w = 0.001 * np.random.randn(dim)#从标准正态分布中返回多个维度为D的theta

        loss_history = []#存放每次损失函数的值的列表
        for it in range(num_iters):
            #存放每次训练的数据集
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            mask = np.random.choice(num_train,batch_size)
            X_batch = X [mask,:]
            y_batch = y[mask]
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # evaluate loss and gradient(计算损失和梯度)
            loss, grad = self.loss(X_batch, y_batch)
            loss_history.append(loss)
            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            self.w = self.w - learning_rate * grad
            # print("+++++")
            # print((y_batch - self.sigmoid(X_batch, self.w)).shape)
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 100 == 0:
                print ('iteration %d / %d: loss %f' % (it, num_iters, loss))
        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: N x D array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
        array of length N, and each element is an integer giving the predicted
        class.
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        h_value = self.sigmoid(X,self.w)
        print("+++")
        print(h_value)
        print(h_value.shape)
        for i in range(len(y_pred)):
            if h_value[i] >=0.5:
                y_pred[i] = 1
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    #多分类的损失函数
    def mult_loss(self,X_batch,y_batch,theta):
        J = (-1)*(np.sum(y_batch.T.dot(np.log(self.sigmoid(X_batch,theta)))+(1-y_batch).T.dot(np.log(1-self.sigmoid(X_batch,theta)))))/X_batch.shape[0]
        der = X_batch.T.dot(self.sigmoid(X_batch,theta)-y_batch)/X_batch.shape[0]
        loss = tuple([J,der])
        return loss

    # 多分类训练函数
    def mult_train(self, X, y, learning_rate=1e-3, num_iters =100, batch_size=200, verbose = True):
        num_train, dim = X.shape[0], X.shape[1]
        for i in range(10):
            if self.k is None:
                self.k = 0.001*np.random.randn(10,dim)
        muli_loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None
            mask = np.random.choice(num_train,batch_size)
            X_batch = X[mask,:]
            y_batch = y[mask]
            y_bat = np.zeros((batch_size,10))
            # 将标签值改为one-code编码
            for i in range(batch_size):
                y_bat[i,y_batch[i]] = 1
            loss,grad = self.mult_loss(X_batch,y_bat,self.k)
            muli_loss_history.append(loss)
            self.k = self.k - learning_rate*grad.T
            # if verbose and it % 100 == 0:
            #     print('iteration %d / %d: loss %f' % (it, num_iters, loss))




    def one_vs_all(self, X, y, learning_rate = 1e-3, num_iters = 100,
            batch_size=200, verbose = True):
        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        """
        dim = X.shape[1]
        self.k = np.zeros((10,dim))
        # 训练10个二分类器
        for i in range(10):
           self.mult_train(X, y, learning_rate, num_iters, batch_size, verbose)
    # 多分类预测，预测值10个二分类中最大概率的索引值即为预测分类值
    def mult_predict(self,X):
        h_value = self.sigmoid(X,self.k)
        y_pre = np.zeros(X.shape[0])
        for i in range(len(y_pre)):
            y_pre[i]= np.argmax(h_value[i,:])
        return y_pre




