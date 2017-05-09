Requirements
============
* pep8
* python-coverage
* mysql-client
* mysql-server
* mysql-python
* json
* sklearn
* numpy
* sphinx
* scipy
* overrides
* tensorflow
* keras
* pandas
* lda2vec
* spaCy
* chainer
* python-dev

Setup 
==========

1. Importing the dataset: 

This project is based on the citeulike dataset from: http://www.wanghao.in/data/ctrsr_datasets.rar
Download the datasets citeulike-a and citeulike-t from the tonga server and add them into the folder data/.

2. Initializing the database:

We assume that the data files are in the directory data/, and the database configuration from config/config.json

To build the database, we run the command: ::

      make rebuild-database

We execute this statement in MySQL console before importing the data: ::

      set autocommit = 0;

How to use
==========

#. Reading recommender's specifications from config/recommender.json, we run our recommender is made through the runnables.py script, using the command: ::

     make run

#. Reading experiment's specifications from config/runs.json, we run a full experiment is made through the runnables.py script, using the command: ::

     make experiment

#. The running options are available using the command ::

     python3 runnables.py -h

Testing
=======
#. Running the runtests.py script, will run the tests in tests.tests: ::

      make test

#. Running the runtests.py script, with arguments that contain the modules of the unittests, will run the tests in all the provided modules. ::

      python runtests.py test1 test2 test3 # ...


Documentation
=============
To generate the documentation we can use the command: ::

      make docs

Linting
=======
We are following flake8 code conventions, we can verify it using the command: ::

      make lint-flake8


Issues and TODOs
================
https://github.com/anasalzogbi/HyPRec/issues

License
=======
HyPRec is licensed under the Apache License 2.0 

Copyright © 2017 by Albert-Ludwigs-Universität Freiburg, Institut für Informatik 
