#!/usr/bin/env python
"""
A module that provides functionalities for calculating error metrics
and evaluates the given recommender.
"""
import numpy
import math
from sklearn.decomposition import LatentDirichletAllocation
from util.top_recommendations import TopRecommendations

class LTR_Evaluator(object):
    """
    A class for computing evaluation metrics and splitting the input data.
    """
    def __init__(self,n_users,ratings):
      # stores recommended indices for each user.
      self.recommendation_indices = [[] for i in range(n_users)]
      self.ratings = ratings
     
        
    def generate_k_fold_matrices(self,rating_matrix,k):
      """
      Generate k training and test matrices and returns them
      """
      n_users,n_docs = rating_matrix.shape
      n_folds = k
      # initialising train and test matrices
      test_mask = numpy.zeros((n_folds,n_users,n_docs) , dtype = bool)
      #train_data = numpy.zeros((n_folds,n_users,n_docs))
      #for x in range(n_folds):
        #TODO: memory effic code
       # train_data[x] = rating_matrix.copy()
      #setting the train and test matrices
      for user in range(n_users):
        rated_item_indices = rating_matrix[user].nonzero()[0]
        non_rated_indices = numpy.where(rating_matrix[user] == 0)[0]
        numpy.random.shuffle(rated_item_indices)
        numpy.random.shuffle(non_rated_indices)
        
        count = math.floor((1.0 / n_folds) * len(rated_item_indices))
        if (count == 0):
           count = 1
           
        count2 = math.floor((1.0 / n_folds) * len(non_rated_indices))
        #TODO: comment
        for k in range(n_folds):
            if (k+1) == n_folds:
              #test indices for rated items
              test_indices = rated_item_indices[k*count :  ]
              #test indices for non rated items
              test_indices_zeros = non_rated_indices[k*count2 :  ]
            else:
              test_indices = rated_item_indices[k*count : (k+1)*count ]
              test_indices_zeros = non_rated_indices[k*count2 : (k+1)*count2 ]
            #train_indices = numpy.append(rated_item_indices[(k+1)*count :],rated_item_indices[:(k)*count] ) # deprecated
            #train_data[k,user,test_indices] = 2
            #train_data[k,user,test_indices_zeros] = 2

            test_mask[k,user,test_indices] = True
            test_mask[k,user,test_indices_zeros] = True
      #return train_data,test_data
      return test_mask
      
    def calculate_mrr(self, n_recommendations, predictions, prediction_scores ,test_mask):
      """
      The method calculates the mean reciprocal rank for all users
      by only looking at the top n_recommendations.

      :param int n_recommendations: number of recommendations to look at, sorted by relevance.
      :param float[][] predictions: calculated predictions of the recommender
      :returns: mrr at n_recommendations
      :rtype: float
      """
      self.recommendation_indices =self.load_top_recommendations(n_recommendations, prediction_scores,test_mask)
      mrr_list = []
      
      for user in range(self.ratings.shape[0]):
        mrr = 0
        for mrr_index, index in enumerate(self.recommendation_indices[user]):
          score = self.ratings[user][index] * predictions[user][index]
          if score == 1:
            mrr = score / (mrr_index + 1)
            break
          if mrr_index + 1 == n_recommendations:
            break
        mrr_list.append(mrr)

      return numpy.mean(mrr_list, dtype=numpy.float16)

      
    def load_top_recommendations(self, n_recommendations, predictions ,test_mask):
      """
      This method loads the top n recommendations into a local variable.

      :param int n_recommendations: number of recommendations to be generated.
      :param int[][] predictions: predictions matrix (only 0s or 1s)
      :returns: A matrix of top recommendations for each user.
      :rtype: int[][]
      """
      for user in range(self.ratings.shape[0]):
        test_indices = numpy.where(test_mask[user])[0]
        top_recommendations = TopRecommendations(n_recommendations)
        for index in test_indices:
          top_recommendations.insert(index, predictions[user][index])
        self.recommendation_indices[user] = list(reversed(top_recommendations.get_indices()))
        top_recommendations = None

      return self.recommendation_indices
  
    def calculate_ndcg(self, n_recommendations, rounded_predictions):
        """
        The method calculates the normalized Discounted Cumulative Gain of all users
        by only looking at the top n_recommendations.
        :param int n_recommendations: number of recommendations to look at, sorted by relevance.
        :param float[][] predictions: calculated predictions of the recommender
        :returns: nDCG for n_recommendations
        :rtype: float
        """
        ndcgs = []
        for user in range(self.ratings.shape[0]):
            dcg = 0
            idcg = 0
            for pos_index, index in enumerate( self.recommendation_indices[user] ):
                dcg += (self.ratings[user, index] * rounded_predictions[user][index]) / numpy.log2(pos_index + 2)
                idcg += 1 / numpy.log2(pos_index + 2)
                if pos_index + 1 == n_recommendations:
                    break
            if idcg != 0:
                ndcgs.append(dcg / idcg)
        return numpy.mean(ndcgs, dtype=numpy.float16)  