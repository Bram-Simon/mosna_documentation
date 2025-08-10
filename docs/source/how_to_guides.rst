How to Guides
=============





Tysserand
---------

Tysserand generates computational networks that can subsequently be analyzed with mosna.
To generate such networks with tysserand, we need to provide it with the spatial coordinates of the nodes, which can either be individual cells
or spots (such as in Visium 10X genomics). The nodes should be provided as a numpy array of shape ``(n_nodes, 2)``, where the first column contains the
x-coordinates and the second column contains the y-coordinates \footnote{It is also possible to provide 3D coordinates}.
Assuming we have a pandas dataframe ``group`` with columns ``X_position`` and ``Y_position`` that specify the spatial coordinates of the nodes, 
we can use the following code to generate the network:

.. code-block:: console

  df_nodes = group[['X_position', 'Y_position']]
  df_nodes.columns = ['X_position', 'Y_position']
  np_array_nodes = df_nodes.values
  np_array_edges = ty.build_delaunay(np_array_nodes)

The function ``build_delaunay`` calculates the edges of the network based on the Delaunay triangulation of the nodes.
Now we are ready to plot the network using tysserand's built-in plotting functionality:

.. code-block:: python

  # By calculating the distances, we can use the distance as a color-mapper.
  distances = ty.distance_neighbors(np_array_nodes, np_array_edges)

  ty.plot_network_distances(
        np_array_nodes, 
        np_array_edges, 
        distances, 
        labels=df_cluster_id, 
        figsize=(100,100), 
        legend_opt={'fontsize': 52, 'bbox_to_anchor': (0.96, 1), 'loc': 'upper left'},
        size_nodes=60,
        color_mapper=color_mapper,
        cmap_nodes=cmap_nodes,
        ax=ax  # Ensure you pass the axis here
    )


.. image:: images/img1_tysserand_network.png
   :alt: Example result
   :width: 94%
   :align: center


.. raw:: html

   <br><br>
   <br><br>
   <br><br>



Calculating Assortativity with mosna
------------------------------------

Assortativity analysis in mosna allows you to quantify the tendency of nodes with similar
attributes to connect to each other in spatial networks.

Creating Mixing Matrices
------------------------

In mosna, you can generate mixing matrices using the ``mixing_matrix()`` and ``count_edges_directed()`` functions.

The ``mixing_matrix()`` function requires three main arguments:

- **nodes**: A pandas DataFrame containing one-hot-encoded attributes for each node in the network
- **edges**: A pandas DataFrame containing edge information with two columns named 'source' (node 1) and 'target' (node 2)
- **attributes**: A list containing all unique attributes (e.g., cell phenotypes, cluster labels) to analyze

.. code-block:: python

    # Example usage of mixing_matrix function
    mixing_matrix_result = mosna.mixing_matrix(
        nodes=nodes_df,
        edges=edges_df,
        attributes=phenotype_list
    )

**Important**: The edges DataFrame must contain exactly two columns named 'source' and 'target'. These column names are hardcoded in the mosna implementation and cannot be changed.

How Mixing Matrices Work
------------------------

The mixing matrix calculation process works as follows:

1. **Matrix Initialization**: A square matrix is created with dimensions equal to the number of unique attributes
2. **Edge Counting**: For each position (i, j) in the matrix, the function counts the number of edges between nodes with attributes i and j
3. **Undirected Analysis**: The ``count_edges_undirected()`` function is used to ensure that edges are counted regardless of direction

We can now populate the mixing matrix as follows:

.. code-block:: python

    # For each attribute combination (i, j)
    mixmat[i, j] = count_edges_undirected(
        nodes, 
        edges, 
        attributes=[attributes[i], attributes[j]]
    )

**Calculate Edges**

The function uses logical operations to identify valid edge pairs.
For each edge, it checks if the source and target nodes have the specified attributes:

.. code-block:: python

    # Example of the logical operations used internally
    pairs = np.logical_or(
        np.logical_and(
            nodes.loc[edges['source'], attributes[0]].values,
            nodes.loc[edges['target'], attributes[1]].values
        ),
        np.logical_and(
            nodes.loc[edges['target'], attributes[0]].values,
            nodes.loc[edges['source'], attributes[1]].values
        )
    )


**Data Requirements**

- **One-hot encoding**: Node attributes must be one-hot encoded in the nodes DataFrame
- **Consistent indexing**: The node indices in the edges DataFrame must correspond to the row indices in the nodes DataFrame
- **Unique attributes**: The attributes list should contain all unique phenotypes or cluster labels you want to analyze

This mixing matrix approach provides a robust foundation for calculating various assortativity metrics and understanding spatial organization patterns in your single-cell data.


