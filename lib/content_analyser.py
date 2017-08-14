"""
A module that provides functionalities for analysing document information
"""
import numpy
import math
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

class Content_Analyser(object):
    """
    A class for content analysis of documennts
    """

    def get_document_distribution(self,term_freq):
      """
        This function calculates the document distribution matrix and returns it
      """
      lda = LatentDirichletAllocation(n_topics=20, max_iter=5,
                                      learning_method='online',
                                      learning_offset=50., random_state=0,
                                      verbose=0)
      document_distribution = lda.fit_transform(term_freq)
      return  document_distribution

    def get_sorted_cosine_sim(self,theta):
      """
        This function calcualte the cosine similarity between the theta matrix and return them in sorted manner
      """
      pairwise_cosine_sim = cosine_similarity(theta)
      sorted_sim_indices = numpy.argsort(pairwise_cosine_sim)
      return sorted_sim_indices  