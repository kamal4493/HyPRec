#!/usr/bin/env python
"""
Module that provides the main functionalities of LTR approach to CBF.
"""
import time
import numpy
from sklearn import svm
import numpy.ma as ma
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

numpy.set_printoptions(threshold=numpy.inf)
        

class LTRRecommender(object):
    """
      A class that takes in the Labeled rating matrix and theta(the output of LDA)
    """
    def __init__(self,n_users,n_docs,theta,strategy,sorted_sim_indices , ratings):
      """
      """
      self.n_users = n_users
      self.n_docs = n_docs
      self.user_model = [None for user in range(n_users)]
      self.theta = theta
      self.split_strategy = strategy
      self.sorted_sim_indices = sorted_sim_indices  
      self.test_indices = [None for user in range(n_users)]
      self.ratings = ratings
      #using random neg paper indice to experiment in predict
      self.random_neg_indices= [None for user in range(n_users)]
      
    def build_pairs(self,labeled_r,theta ,user_id):
      """
        This function builds the pairs from the labeled rating matrix such that each pair is a positive and negative entry
        :param matrix labeled_r : labeled rating matrix consisting of 1,-1 and 0
        :param theta : document distribution matrix
        :returns : pair matrix of all positive and negative ratings and labels
          
      """
      pos_ids = numpy.where( labeled_r[user_id,:] == 1)[0]
      neg_ids = numpy.where( labeled_r[user_id,:] == -1)[0]
      #shuffling them to destroy any order
      numpy.random.shuffle(pos_ids)
      numpy.random.shuffle(neg_ids)

      self.random_neg_indices[user_id] = numpy.random.choice(neg_ids)
      
      pair_matrix = numpy.zeros(( len(pos_ids) * len(neg_ids) * 2 , theta.shape[1] ))
      labels_ones = numpy.ones( len(pos_ids) * len(neg_ids) )
      labels_zeros = numpy.zeros(len(pos_ids) * len(neg_ids) )
      labels = numpy.append(labels_ones,labels_zeros)
      counter = 0
      for pos_id in pos_ids:
        for neg_id in neg_ids:
          pair_matrix[counter] = theta[pos_id] - theta[neg_id]
          counter = counter + 1
      for pos_id in pos_ids:
        for neg_id in neg_ids:
          pair_matrix[counter] = theta[neg_id] - theta[pos_id]
          counter = counter + 1
      return pair_matrix,labels

    def train(self, test_mask):
      """
      Train the LTR Recommender.
      """
      labeled_r = None
      train_data = self.ratings.copy()
      #zero values are also considered as a mask
      train_data[train_data == 0] = 2
      train_data = ma.masked_array(train_data , test_mask)
      if (self.split_strategy == "random"):
        labeled_r = self.put_random_negatives(train_data)
      elif (self.split_strategy == "pairwise"):
        labeled_r = self.put_pairwise_negatives(train_data)

      theta = self.theta
      for user in range(self.n_users):
        #builds a pair matrix for every user according to the documents he rated
        pair_matrix,labels = self.build_pairs(labeled_r,theta,user)
        #fit Model that is SVM
        self.user_model[user] = svm.SVC(kernel="linear", probability = True)
        self.user_model[user].fit(pair_matrix, labels)

    def put_random_negatives(self,rating_matrix):
      """
      input:
          rating matrix
      :returns:
          randomly put negative ratings and return the rating_matrix
      :rtype: int[][]
      """
      for user in range(self.n_users):
        zero_indices = numpy.where(rating_matrix[user] == 2 )[0]
        num_positive_ids = numpy.count_nonzero(rating_matrix[user] == 1)
        random_negatives = numpy.random.choice(zero_indices,num_positive_ids,replace=False)
        for random_id in random_negatives:
          rating_matrix[user,random_id] = -1          
      return rating_matrix

    def put_pairwise_negatives(self,rating_matrix):
      """
      input:
          rating matrix
      :returns:
          randomly put negative ratings and return the rating_matrix
      :rtype: int[][]
      """
      for user in range(self.n_users):
        ones_indices = numpy.where(rating_matrix[user] == 1)[0]
        numpy.random.shuffle(ones_indices)
        num_positive_ids = numpy.count_nonzero(rating_matrix[user] == 1)
        neg_ids = []
        for indice in ones_indices:
          for document_id in self.sorted_sim_indices[indice]:
            # and not ma.is_masked(rating_matrix[user,document_id]):
            if (document_id not in neg_ids) and rating_matrix[user,document_id] == 2 :
              neg_ids.append(document_id)
              break
        for neg_id in neg_ids:
          rating_matrix[user,neg_id] = -1          
      return rating_matrix

    def predict(self,test_mask):
      """
      Predict ratings for every user and item.

      :returns: A (user, document) matrix of predictions
      :rtype: ndarray
      """

      predictions = numpy.zeros((self.n_users,self.n_docs))
      prediction_scores = numpy.zeros((self.n_users,self.n_docs))
      for user in range(self.n_users):

        """
        #TODO: vectorize this
        for index,indice in enumerate(self.test_indices[user]):
          doc_to_pred = self.theta[indice]
          print(doc_to_pred.shape)
          #experimenting to get a difference of test document and a potential negative rated document and then predicting
          #doc_negative = self.random_neg_indices[user]
          #doc_to_pred = doc_to_pred - self.theta[doc_negative]
          doc_to_pred.shape = (1,doc_to_pred.shape[0])
          predictions[user][indice] = self.user_model[user].predict(doc_to_pred)
          prediction_scores[user][indice] = self.user_model[user].predict_proba(doc_to_pred)[0, 1]"""
        #half vectorized version (still outer loop there) :(

        #docs_to_test = self.theta[ self.test_indices[user] ]
        test_indices_user = numpy.where(test_mask[user])[0]
        docs_to_test = self.theta[test_indices_user ]
        predictions[user , test_indices_user ] = self.user_model[user].predict(docs_to_test)
        prediction_scores[user,test_indices_user  ] = self.user_model[user].predict_proba(docs_to_test)[:, 1]

      return predictions,prediction_scores