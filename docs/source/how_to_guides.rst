How to Guides
=============


Mosna proposes a pipeline to explore increasingly complex features in relation with clinical data.
It can be used to extract and visualize descriptive statistics, and to identify features that are most
predictive of clinical variables, by training machine learning models.
In particular, the following features are explored, in order of increasing complexity:

- Fractional abundance of cellular phenotypes
- Preferential interractions between different phenotypes (quantified with assortativity z-scores)
- Cellular niches


Mosna leverages the tysserand library to discover patterns of cellular interaction that are potentially clinically relevant.
In this how-to guide we discuss how to:

- generate and visualize spatial networks with tysserand
- calculate assortativity scores that quantify preferential interactions with mosna
- generate mixing matrices with mosna
- identify and visualize cellular niches with mosna (to do)


Generate and Visualize Spatial Networks
---------------------------------------

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

The function ``build_delaunay`` calculates the edges of the network based on the physical distance of the nodes using the Delaunay triangulation.



**Use Trimming to Improve Network Construction**

Next, we will refine the network by removing long-distance connections, which are unlikely to represent real cellular interactions.

.. code-block:: python

  pairs = ty.build_delaunay(
        coords, 
        node_adaptive_trimming=True, 
        n_edges=3, 
        trim_dist_ratio=2,
        min_dist=0, 
        trim_dist=150,
    )

- ``node_adaptive_trimming=True`` enables the removal of edges based on distance
- ``n_edges=3`` ensures that each node has at least 3 connections
- ``trim_dist`` defines the maximum allowed edge length, in this case 150
- ``trim_dist_ratio=2`` sets distance ratio to help define which edges need to be removed


**Color Mapping**

Given a set of unique attributes (e.g. phenotypes) ``uniq``, we can generate a color mapping as follows.

.. code-block:: python

  clusters_cmap = mosna.make_cluster_cmap(uniq)
  celltypes_color_mapper = {x: clusters_cmap[i % n_colors] for i, x in enumerate(uniq)}


When visualizing the network, this color mapping will be used to give each node a color that corresponds to its attribute (e.g. phenotype)



**Handling Isolated Cells**

Solitary nodes can be removed as follows:

.. code-block:: python

  pairs = ty.link_solitaries(np_array_nodes, np_array_edges, method='delaunay', min_neighbors=3)




**Visualization of Network**

Now we are ready to plot the network using tysserand's built-in plotting functionality:

.. code-block:: python

  # By calculating the distances, we can use the distance as a color-mapper.
  distances = ty.distance_neighbors(np_array_nodes, np_array_edges)

  ty.plot_network_distances(
        np_array_nodes, 
        np_array_edges, 
        distances, 
        labels=df_cluster_id, 
        figsize=(100,100)     # The resolution of the resulting image depends on this. Notice that (100, 100) will generate a very detailed network, 
                              # but may require significant computational time for generating the network.
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






Data Transformation and Batch Correction
----------------------------------------

To normalize marker expression data, we can apply centered log-ratio (CLR) transformation:

.. code-block:: python

    obj_transfo = mosna.transform_data(
    data=obj, 
    groups=sample_col,
    use_cols=marker_cols,
    method='clr')


- ``groups=sample_col`` creates groups to ensure that the transformations are applied to each sample separately
- ``use_cols=marker_cols`` specifies which columns contain marker expression data (as only these need to be normalized)



**Visualization for Quality Control**

Next, we generate a simple histogram for quality control

.. code-block:: python

  obj_transfo[marker_cols].hist(bins=50, figsize=(20, 20));



**Network Node Transformation and aggregation**

We apply the same correction to the network node data. Then we aggregate the nodes

.. code-block:: python

  nodes_dir = mosna.transform_nodes(
      nodes_dir=nodes_dir,
      id_level_1='patient',
      id_level_2='sample', 
      use_cols=marker_cols,
      method='clr',
      save_dir='auto',
  )
  nodes_agg = mosna.aggregate_nodes(
      nodes_dir=nodes_dir,
      use_cols=marker_cols,
  )

This combines all the nodes in the transformed network into a single data set. We can then assess and correct batch effects.


**Dimensionality reduction**

We create a UMAP for visual assessment of the batch effects, before correcting them.

.. code-block:: python

  embed_viz, _ = mosna.get_reducer(nodes_agg[marker_cols], nodes_dir)
  fig, ax, color_mapper = mosna.plot_clusters(
      embed_viz, 
      cluster_labels=nodes_agg['patient'], 
      save_dir=None,
      return_cmap=True,
      show_id=False,
  )

  fig, ax, color_mapper = mosna.plot_clusters(
      embed_viz, 
      cluster_labels=nodes_agg['sample'], 
      save_dir=None,
      return_cmap=True,
      show_id=False,
  )


**Batch Effect Correction**

Now we can apply the batch effect correction. In this step, the systematic differences between patients/samples are removed,
while preserving the present biological variation.

.. code-block:: python

  nodes_dir, nodes_corr = mosna.batch_correct_nodes(
      nodes_dir=nodes_dir,
      use_cols=marker_cols,
      batch_key='patient',
      return_nodes=True,
  )




Comparing Response Groups and Survival analysis
-----------------------------------------------




**Differential Analysis between Response Groups**

First, we will investigate how differences in fractional abundance of cell-types are associated to differences in response:

.. code-block:: python

  pvals = mosna.find_DE_markers(prop_types, group_ref=1, group_tgt=2, group_var=group_col)

Now that we have calculated the p-values, which are corrected for the false discovery rate (FDR), we can visualize the differences between different patient groups.

.. code-block:: python

  fig, ax = mosna.plot_distrib_groups(
      prop_types, 
      group_var=group_col,
      groups=[1, 2], 
      pval_data=pvals, 
      pval_col='pval', 
      max_cols=-1, 
      multi_ind_to_col=True,
      group_names=group_names,
      )
  fig.suptitle("Cell type proportions per response group", y=1.0);

An example result is shown in the image below:

.. image:: images/img3_responder_non_responder_example.png
   :alt: Example result
   :width: 94%
   :align: center







Assortativity and Mixing Matrices
---------------------------------

After looking at the fractional cell abundances, we move towards the next step of complexity: patterns of preferential interactions between cell-types.
Assortativity analysis in mosna allows you to quantify preferential interactions between nodes with different attributes (e.g. cell types).
Moreover, z-scores can be calculated to show the statistical significance of these preferential interactions.
These assortativity z-scores can be ordered in a mixing matrix.
An example is provided in the figure below, where we have used cell phenotypes as attributes.




.. image:: images/img2_mixmat_example.png
   :alt: Example result
   :width: 94%
   :align: center


In a mixing matrix, the attributes (phenotypes) are placed on both the x- and the y-axis.
Each cell in the matrix represents the assortativity z-score between the corresponding attributes.
In our example above, for example, neutrophils are preferentially interacting amongst themselves (top left cell),
whereas neutrophils and regulatory T-cells show avoidant behavior (bottom left cell).



To generate these mixing matrices, mosna makes use of the functions ``mixing_matrix()`` and ``count_edges_directed()``.
The ``mixing_matrix()`` function initializes the mixing matrix, and requires three main arguments:

- **nodes**: A pandas DataFrame containing one-hot-encoded attributes for each node in the network
- **edges**: A pandas DataFrame containing edge information with two columns named 'source' (node 1) and 'target' (node 2)
- **attributes**: A list containing all unique attributes (e.g., cell phenotypes, cluster labels) to analyze

.. code-block:: python

    # Example usage of mixing_matrix function
    mixmat = mosna.mixing_matrix(
        nodes=nodes_df,
        edges=edges_df,
        attributes=phenotype_list
    )

**Important**: The edges DataFrame must contain exactly two columns named 'source' and 'target'. The ``mixing_matrix()`` function uses these names internally, so they cannot be changed.

Furthermore, it is important to keep the following requirements on the input data in mind:

- **One-hot encoding**: Node attributes must be one-hot encoded in the nodes DataFrame
- **Consistent indexing**: The node indices in the edges DataFrame must correspond to the row indices in the nodes DataFrame
- **Unique attributes**: The attributes list should contain all unique phenotypes or cluster labels you want to analyze


Subsequently, we can populate the mixing matrix as follows:

.. code-block:: python

    # For each attribute combination (i, j)
    mixmat[i, j] = count_edges_undirected(
        nodes, 
        edges, 
        attributes=[attributes[i], attributes[j]]
    )






