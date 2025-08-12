Theoretical Concepts
====================

Here, we provide some background on concepts such as assortativity and Delaunay triangulation.
For practical examples of how to implement mosna, skip ahead to the next section.

.. _assortativity:

Assortativity
-------------

Assortativity can be defined as a tendency of links to exist between nodes with similar attributes [1]_.
It is a general measure of preferential interactions between nodes that share the same attributes, expressed as a single scalar value.
These attributes can, for example, be cell type labels or marker positiveness.
In mosna, assortativity is used to quantify preferential interactions between different cell types or spots (attributes),
where attributes that are often found together have a positive assortativity value those that
show avoidant behavior a negative assortativity value.

Interactions between neighboring cells
are known to underpin many physiological processes, including immune responses [2]_.
Hence, assortativity can potentially provide valuable insights into which interactions are
occurring in resected tissue that is analyzed with any type of spatial imaging technique.
Since assortativity can be calculated for all attribute pairs
in a cellular network, it is possible to obtain a large number of features and make cross-sample
comparisons. Adding clinical data enables us to investigate which of these features have
predictive power to predict target features in the clinical dataset.
This could enable biomarker discovery.

**Network Attribute Randomization**

However, calculating assortativity is not a straightforward task. The relative proportion of cell types in the network affects the apparent assortativity: if there are many cells
of the same type in a network, most of the edges in that network will be between cells of
the same type [3]_. As a result the network will appear very assortative [3]_. To correct
for this, MOSNA performs network attribute randomization, shuffling the assignment of values of each attribute to the cells [3]_.
With this method, the number of cells that are positive for each attribute and the links between the cells are preserved.

In MOSNA, the 'N_shuffle' parameter is used for this purpose, by specifying the number of randomizations. This process is then repeated N times. It is important to choose the
number of randomizations high enough so that all phenotypes in a sample get satisfactory coverage. Especially when several cell types occur far fewer in a sample than others,
interactions between these types are more rare. A higher number of randomizations is
then required.

Delaunay Triangulation
----------------------

Which nodes in a network are considered to interact is determined by mosna using the physical distance between them.
For this purpose, multiple distance metrics can be used. In our examples, we will use
Delaunay triangulation, which is widely used in computational geometry, both in two and three
dimensional space [4]_. It divides a set of points into a triangle mesh (a set of triangles
connected by their common edges). A max-min angle criterion is then imposed [4]_. This
requires that the diagonal of every convex quadrilateral — a four sided polygon that has
interior angles smaller than 180 degrees each — is "chosen well" [4]_. It is "chosen well",
if replacing the diagonal by an alternative one does not increase the minimum of the six
angles in the two triangles that make up the quadrilateral [4]_. Hence, the Delaunay
triangulation of a set of points (in a plane) will maximize the minimum angle in any
triangle [4]_.



References
----------

.. [1] Liu, X., Murata, T., & Wakita, K. (2014). Detecting network communities beyond assortativity-related attributes. Physical Review E, 90(1), 012806.

.. [2] Sellmyer, M. A., Bronsart, L., Imoto, H., Contag, C. H., Wandless, T. J., & Prescher, J. A. (2013). Visualizing cellular interactions with a generalized proximity reporter. Proceedings of the National Academy of Sciences, 110(21), 8567-8572.

.. [3] Coullomb, A., & Pancaldi, V. (2023). mosnareveals different types of cellular interactions predictive of response to immunotherapies in cancer.

.. [4] Musin, O. R. (1997, August). Properties of the Delaunay triangulation. In Proceedings of the thirteenth annual symposium on Computational geometry (pp. 424-426).