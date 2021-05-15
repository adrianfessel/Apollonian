# Apollonian
 Method to generate the 'apollonian network' [1] for an arbitrary binary shape. Binary shapes can be reduced to a network-like structure referred to as the skeleton via morphological operations. However, this trditional approach has a number of deficiencies. First, the skeleton is very sensitive to distortions of the shapes boundary and is prone to forming random structures in extended shapes. Second, from a graph-theoretical perspective it is problematic to find a definition for nodes of degree k=2 (nodes with two neighbors) based on the skeleton. For certain scenarios, the apollonian network can provide a way to fix these shortcomings.
 
 The detailed procedure is described in [1]. The figure below illustrates these steps for the binary representation of a Physarum polycephalum network. Briefly, nodes are defined as the centers of the non-overlapping disks that form a dense circle-packing of the binary shape. Therefore each node has a position and a defined radius, thus fixing the number of k=2 nodes between junctions. Nodes are placed by iteratively computing the euclidean distance transform and placing nodes centered around the maximum of the edt, with the maximum value giving the radius of the node. To avoid unnecessary computations from repeatedly computing the edt for the full image, the edt is calculated in in a window around the node.
 
 The set of edges is determined by comparing with the traditional skeleton. Skeleton nodes that fall within an apollonian circle are combined and apollonian nodes are connected as prescribed by the skeleton. In detail, the method projects the voronoi tesselation of apollonian circle centers onto the skeleton and establishes edges between voronoi basins touching on the skeleton.
 
![alt text](https://github.com/adrianfessel/Apollonian/blob/main/method.png?raw=true)

# Reference
 [1] : http://nbn-resolving.de/urn:nbn:de:gbv:46-00107082-12
