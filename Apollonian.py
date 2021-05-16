import numpy as np
import matplotlib.pyplot as plt
import cv2
import networkx as nx
import scipy.spatial as spatial
from scipy.ndimage.morphology import distance_transform_edt as edt
import skimage.morphology as morphology


def apollonian(B):
    """
    Function to generate apollonian graph for an arbitrary binary shape. Circle
    packing is performed according to 'apollonian_circle_packing'. Edges are
    established by projecting the voronoi tesselation of circle centers onto
    the morphological skeleton of the binary shape and determining regions
    that are adjacent on the skeleton but associated with different circle centers.
    See [1] for a detailed explanation of the procedure.

    Parameters
    ----------
    B : numpy array
        image containing the binary shape

    Returns
    -------
    G : networkx graph
        apollonian graph with node attributes
        'r' : radius
        'x' : x-coordinate
        'y' : y-coordinate
        
    References
    ----------
    [1] : http://nbn-resolving.de/urn:nbn:de:gbv:46-00107082-12

    """
    
    R, X, Y = apollonian_circle_packing(B.copy(), 4)
    
    G = nx.Graph()
    for ID, (r, x, y) in enumerate(zip(R, X, Y)):
        G.add_node(ID, r=r, x=x, y=y)
        
    S = morphology.skeletonize(B>0)
    XS, YS = np.where(S)
    
    # V = get_voronoi_map(X, Y, B.shape)
    V = get_voronoi_skeleton(X, Y, XS, YS, S.shape)
    
    D = spatial.distance.pdist(np.transpose(np.stack([XS, YS])))
    
    for k in np.argwhere(D<1.1*np.sqrt(2)):
                
        i, j = condensed_to_square(k, len(YS))

        if V[XS[i], YS[i]] - V[XS[j], YS[j]] != 0:
            G.add_edge(V[XS[i], YS[i]], V[XS[j], YS[j]])
    
    VS = np.unique(V*S)
    
    to_remove = []
    
    for ID in G:
        if ID not in VS:
            to_remove.append(ID)
            
    G.remove_nodes_from(to_remove)
    
    return G


def apollonian_circle_packing(B, minR):
    """
    Function for apollonian circle packing. Packs an arbitrary binary shape with
    circles by iteratively placing the largest possible circle that does not
    overlap with the border of the shape or with any other circle. Terminates
    when a minimal radius is reached.
    To avoid frequent recalculation of the euclidean distance transform
    for the full shape, the distance transform is recalculated only in a
    window surrounding the last circle that has been placed.
    See [1] for a detailed explanation of the procedure.

    Parameters
    ----------
    B : numpy array
        binary image
    minR : float
        minimal radius

    Returns
    -------
    R : numpy array
        radii of all circles
    X : numpy array
        x-coordinates of all circles
    Y : numpy array
        y-coordinates of all cirvles

    References
    ----------
    [1] : http://nbn-resolving.de/urn:nbn:de:gbv:46-00107082-12

    """
    
    B[:,0] = 0
    B[:,-1] = 0
    B[0,:] = 0
    B[-1,:] = 0
    
    Dist = edt(B)
    
    R = list()
    X = list()
    Y = list()
    
    Height, Width = B.shape
    mX, mY = np.meshgrid(range(Width),range(Height))
    
    r = (Height+Width)/2
    
    while r > minR:
        Ind = np.argmax(Dist)
        y, x = np.unravel_index(Ind,Dist.shape)
        
        r = Dist[y,x]

        ymin = 0 if int(y-2*r) < 0 else int(y-2*r)
        ymax = Height-1 if int(y+2*r) >= Height else int(y+2*r) 
        xmin = 0 if int(x-2*r) < 0 else int(x-2*r)
        xmax = Width-1 if int(x+2*r) >= Height else int(x+2*r) 
        
        B[np.sqrt((x-mX)**2+(y-mY)**2)<=r] = 0

        # recalculate edt in a window, fuse with full edt with gaussian weights
        
        Dist_Window = edt(B[ymin:ymax,xmin:xmax])
        xWm, yWm = np.meshgrid(range(np.size(Dist_Window,axis=1)),range(np.size(Dist_Window,axis=0)))
    
        Map = np.sqrt((np.mean(xWm)-xWm)**2+(np.mean(yWm)-yWm)**2)
        Map = Map-np.min(Map)
        Map = Map/np.max(Map)
        
        Dist[ymin:ymax,xmin:xmax] = (1-Map**2)*Dist_Window + Map**2*Dist[ymin:ymax,xmin:xmax]
        
        R.append(r)
        X.append(x)
        Y.append(y)
                       
    return R, X, Y


def plot_apollonian(G):
    """
    Function for displaying the apollonian graph. Nodes are colorcoded 
    according to their degree.

    Parameters
    ----------
    G : networkx graph
        apollonian graph; each node is required to have the attributes
        'r' : radius
        'x' : x-coordinate
        'y' : y-coordinate

    Returns
    -------
    None.

    """

    IDs = [ID for ID in G.nodes()]
    Degrees = {ID:(G.degree(ID) if G.degree(ID) < 6 else 5) for ID in G}
    
    Colors = ['red','lime','gray','blue','yellow','magenta']
    
    for ID in G.nodes():
        c = plt.Circle((G.nodes[ID]['x'], G.nodes[ID]['y']), radius=G.nodes[ID]['r'], edgecolor=Colors[Degrees[ID]], facecolor="none", linewidth=1)
        plt.gca().add_patch(c)
    
    pos = {ID:[G.nodes[ID]['x'],G.nodes[ID]['y']] for ID in IDs}
    nx.draw_networkx_edges(G, pos, width=1)

    plt.xlim([0, np.max([pos[i][0] for i in pos])])
    plt.ylim([0, np.max([pos[i][1] for i in pos])])
    
    plt.axis('equal')
    plt.axis('off')


def get_voronoi_map(x, y, size):
    """
    Function to compute the voronoi tesselation for the set of points (x, y)
    in an array with shape identical to B.

    Parameters
    ----------
    x : numpy array
        x-coordinates
    y : numpy array
        y-coordinates
    size : (int, int)
        desired output shape

    Returns
    -------
    V : numpy array
        voronoi tesselation with shape size

    """

    pos = np.asarray([[xi, yi] for xi, yi in zip(x, y)])
    
    X, Y = np.meshgrid(np.arange(size[1]),np.arange(size[0]))
    XY = np.c_[X.ravel(), Y.ravel()]
    
    Tree = spatial.cKDTree(pos)
    V = np.float32(Tree.query(XY)[1].reshape(size[0], size[1]))

    return V


def get_voronoi_skeleton(x, y, X, Y, size):
    """
    Function to compute the voronoi tesselation for the set of points (x, y),
    sampled only at the points (Y, X).

    Parameters
    ----------
    x : numpy array
        x-coordinates centers
    y : numpy array
        y-coordinates centers
    X : numpy array
        x-coordinates samples
    Y : numpy array
        y-coordinates samples
    size : TYPE
        DESCRIPTION.

    Returns
    -------
    V : numpy array
        sampled voronoi tesselation with shape size

    """
    
    
    pos = np.asarray([[xi, yi] for xi, yi in zip(x, y)])
    XY = np.c_[Y.ravel(), X.ravel()]
    
    Tree = spatial.cKDTree(pos)
    V = np.float32(Tree.query(XY)[1])
    
    VS = np.zeros(size)
    
    for x, y, v in zip(X, Y, V):
        VS[x, y] = v

    return VS

#------------------------------------------------------------------------------
# Functions for converting indices between condensed distance matrices and
# square distance matrices. Source:
# https://stackoverflow.com/questions/13079563/how-does-condensed-distance-matrix-work-pdist

def calc_row_idx(k, n):
    return int(np.ceil((1/2.) * (- (-8*k + 4 *n**2 -4*n - 7)**0.5 + 2*n -1) - 1))

def elem_in_i_rows(i, n):
    return i * (n - 1 - i) + (i*(i + 1))//2

def calc_col_idx(k, i, n):
    return int(n - elem_in_i_rows(i + 1, n) + k)

def condensed_to_square(k, n):
    i = calc_row_idx(k, n)
    j = calc_col_idx(k, i, n)
    return i, j
#-----------------------------------------------------------------------------


if __name__ == '__main__':
    
    import time
    
    start = time.time()
    
    File = './example.jpeg'
    B = cv2.imread(File, cv2.IMREAD_GRAYSCALE)
    
    G = apollonian(B)
    
    print(time.time()-start)
    
    plt.figure(figsize=2*plt.figaspect(1))
    plt.imshow(B, cmap='gray')
    plot_apollonian(G)