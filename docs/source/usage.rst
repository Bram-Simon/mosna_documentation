Usage
=====

.. _installation:

Installation
------------

To install MOSNA for use with GPU compatible libraries:

.. code-block:: console

   (.venv) $ conda create --solver=libmamba -n mosna-gpu -c rapidsai -c conda-forge -c nvidia -c pytorch rapids=23.04.01 python=3.10 cuda-version=11.2 pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 scanpy
           $ conda activate mosna-gpu

Alternatively without GPU:

.. code-block:: console
   (.venv) $ conda create --solver=libmamba -n mosna -c conda-forge python=3.10 scanpy
           $ conda activate mosna

Install some essential dependencies:

.. code-block:: console
   (.venv) $ pip install ipykernel ipywidgets
           $ pip install tysserand
           $ cd /path/to/mosna_benchmark/
           $ pip install -e .
           $ pip install scipy==1.13


Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']

