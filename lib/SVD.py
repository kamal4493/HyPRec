#!/usr/bin/env python
"""
Module that provides the main functionalities of SVD
"""
import time
import numpy
from numpy.linalg import solve
from overrides import overrides
from lib.abstract_recommender import AbstractRecommender
from lib.linear_regression import LinearRegression
from scipy import linalg

class SVD(AbstractRecommender):
    """
    A class that takes in the rating matrix and outputs user and item
    representation in latent space.
    """
    def __init__(self, initializer, evaluator, hyperparameters, options,
                 verbose=False, load_matrices=True, dump_matrices=True, train_more=True,
                 is_hybrid=False, update_with_items=False, init_with_content=True):
        """
        Train a matrix factorization model to predict empty
        entries in a matrix. The terminology assumes a ratings matrix which is ~ user x item

        :param ModelInitializer initializer: A model initializer.
        :param Evaluator evaluator: Evaluator of the recommender and holder of the input data.
        :param dict hyperparameters: hyperparameters of the recommender, contains _lambda and n_factors
        :param dict options: Dictionary of the run options, contains n_iterations and k_folds
        :param boolean verbose: A flag if True, tracing will be printed
        :param boolean load_matrices: A flag for reinitializing the matrices.
        :param boolean dump_matrices: A flag for saving the matrices.
        :param boolean train_more: train_more the collaborative filtering after loading matrices.
        :param boolean is_hybrid: A flag indicating whether the recommender is hybrid or not.
        :param boolean update_with_items: A flag the decides if we will use the items matrix in the update rule.
        """
        # setting input
        self.initializer = initializer
        self.evaluator = evaluator
        self.ratings = evaluator.get_ratings()
        self.n_users, self.n_items = self.ratings.shape
        self.k_folds = None
        self.prediction_fold = -1

        # setting flags
        self._verbose = verbose
        self._load_matrices = load_matrices
        self._dump_matrices = dump_matrices
        self._train_more = train_more
        self._is_hybrid = is_hybrid
        self._update_with_items = update_with_items
        self._split_type = 'user'
        self._init_with_content = init_with_content
        self.hyperparameters = None
        self.set_options(options)

    def compute_svd_matrices(self):
        """
        The function computes the SVD marix factorization of the rating matrix
        """
        U, s, Vh = linalg.svd(self.ratings)
        return U,s,Vh
    @overrides
    def train(self, item_vecs=None):
        """
        Train the SVD collaborative filtering algorithm.

        :param ndarray item_vecs: optional initalization for the item_vecs matrix.
        """
        if item_vecs is not None:
            self.document_distribution = item_vecs.copy()
        else:
            self.document_distribution = None
        if self.splitting_method == 'naive':
            self.set_data(*self.evaluator.naive_split(self._split_type))
            return self.train_one_fold(item_vecs)
        else:
            self.fold_test_indices = self.evaluator.get_kfold_indices()
            return self.train_k_fold(item_vecs)

    @overrides
    def train_k_fold(self, item_vecs=None):
        """
        Trains k folds of sVD collaborative filtering.

        :returns: List of error metrics.
        :rtype: list[float]
        """
        all_errors = []
        for current_k in range(self.k_folds):
            self.set_data(*self.evaluator.get_fold(current_k, self.fold_test_indices))
            current_error = self.train_one_fold(item_vecs)
            all_errors.append(current_error)
            self.predictions = None
        return numpy.mean(all_errors, axis=0)

    @overrides
    def train_one_fold(self, item_vecs=None):
        """
        Train one fold for n_iter iterations from scratch.

        :returns: List of error metrics.
        :rtype: list[float]
        """
        matrices_found = False
        if self._load_matrices is False:
            self.user_vecs = numpy.random.random((self.n_users, self.n_factors))
            if (item_vecs is None or not self._init_with_content
                    or not item_vecs.shape == (self.n_items, self.n_factors)):
                self.item_vecs = numpy.random.random((self.n_items, self.n_factors))
            else:
                self.item_vecs = item_vecs
        else:
            users_found, self.user_vecs = self.initializer.load_matrix(self.hyperparameters,
                                                                       'user_vecs' + self._get_options_suffix(),
                                                                       (self.n_users, self.n_factors))
            if self._verbose and users_found:
                print("User distributions files were found.")
            items_found, self.item_vecs = self.initializer.load_matrix(self.hyperparameters,
                                                                       'item_vecs' + self._get_options_suffix(),
                                                                       (self.n_items, self.n_factors))
            if self._verbose and items_found:
                print("Document distributions files were found.")
            if not items_found and item_vecs is not None:
                items_found = True
                self.item_vecs = item_vecs
            matrices_found = users_found and items_found
        if not matrices_found:
            if self._verbose and self._load_matrices:
                print("User and Document distributions files were not found, will train collaborative.")
            self.partial_train()
        else:
            if self._train_more:
                if self._verbose and self._load_matrices:
                    print("User and Document distributions files found, will train model further.")
                self.partial_train()
            else:
                if self._verbose and self._load_matrices:
                    print("User and Document distributions files found, will not train the model further.")

        if self._dump_matrices:
            self.initializer.set_config(self.hyperparameters, self.n_iter)
            self.initializer.save_matrix(self.user_vecs, 'user_vecs' + self._get_options_suffix())
            self.initializer.save_matrix(self.item_vecs, 'item_vecs' + self._get_options_suffix())

        return self.get_evaluation_report()

    def _get_options_suffix(self):
        suffix = ''
        if self._init_with_content:
            suffix += 'i'
        if self._update_with_items:
            suffix += 'u'
        if suffix:
            suffix = '_' + suffix
        return suffix

    def partial_train(self):
        """
        Train model for n_iter iterations. Can be called multiple times for further training.
        """
        error = numpy.inf
        if 'fold' in self.hyperparameters:
            current_fold = self.hyperparameters['fold'] + 1
        else:
            current_fold = 0
        if self._verbose:
            error = self.evaluator.get_rmse(self.user_vecs.dot(self.item_vecs.T), self.train_data)
            if current_fold == 0:
                print('Epoch:{epoch:02d} Loss:{loss:1.4e} Time:{time:.3f}s'.format(**dict(epoch=0, loss=error,
                                                                                          time=0)))
            else:
                print('Fold:{fold:02d} Epoch:{epoch:02d} Loss:{loss:1.4e} '
                      'Time:{time:.3f}s'.format(**dict(fold=current_fold, epoch=0, loss=error, time=0)))
        for epoch in range(1, self.n_iter + 1):
            t0 = time.time()
            old_error = error
            self.user_vecs = self.als_step(self.user_vecs, self.item_vecs, self.train_data, self._lambda, type='user')
            self.item_vecs = self.als_step(self.item_vecs, self.user_vecs, self.train_data, self._lambda, type='item')
            t1 = time.time()
            error = self.evaluator.get_rmse(self.user_vecs.dot(self.item_vecs.T), self.train_data)
            if self._verbose:
                if current_fold == 0:
                    print('Epoch:{epoch:02d} Loss:{loss:1.4e} Time:{time:.3f}s'.format(**dict(epoch=epoch, loss=error,
                                                                                              time=(t1 - t0))))
                else:
                    print('Fold:{fold:02d} Epoch:{epoch:02d} Loss:{loss:1.4e} '
                          'Time:{time:.3f}s'.format(**dict(fold=current_fold, epoch=epoch, loss=error,
                                                           time=(t1 - t0))))
            if error >= old_error:
                if self._verbose:
                    print("Local Optimum was found in the last iteration, breaking.")
                break

    @overrides
    def get_predictions(self):
        """
        Predict ratings for every user and item.

        :returns: A (user, document) matrix of predictions
        :rtype: ndarray
        """
        if self.predictions is None or not self.prediction_fold == self.hyperparameters['fold']:
            collaborative_predictions = self.user_vecs.dot(self.item_vecs.T)
            if self._is_hybrid:
                self.item_based_recommender.set_data(self.train_data, self.test_data)
                # Train Linear Regression
                regr = LinearRegression(self.train_data, self.test_data, self.item_based_recommender.get_predictions(),
                                        collaborative_predictions)
                self.predictions = regr.train()
                self.prediction_fold = self.hyperparameters['fold']
                if self._verbose:
                    print("returned linear regression ratings")
            else:
                self.prediction_fold = self.hyperparameters['fold']
                self.predictions = collaborative_predictions

        return self.predictions

    @overrides
    def predict(self, user, item):
        """
        Single user and item prediction.

        :returns: prediction score
        :rtype: float
        """
        return self.user_vecs[user, :].dot(self.item_vecs[item, :].T)

    def set_item_based_recommender(self, recommender):
        """
        Set the item_based recommender, in order to use it as a hybrid recommender.

        :param ContentBased recommender: The content based recommender
        """
        self.item_based_recommender = recommender
