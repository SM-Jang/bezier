import matplotlib.pyplot as plt
import numpy as np
import bezier

from sklearn.preprocessing import minmax_scale
from bezier.hazmat.curve_helpers import evaluate_hodograph, get_curvature


def curvature(data, N, norm=False):
    # N: The number of sampling curvature
    
    curve = bezier.Curve.from_nodes(data)
    
    kappa = []
    Number = N 


    for s in range(Number):
        t = s / Number
        tangent_vec = curve.evaluate_hodograph(t)
        kappa.append(get_curvature(data.T, tangent_vec, t))

    if norm: kappa = minmax_scale(kappa)

    return kappa, curve

def plot_curvature(curvature):
    plt.plot(curvature, color='black')
    plt.scatter(range(len(curvature)), curvature, s = 10)
    plt.title("Curvature Plot")
    plt.show()
    plt.close()

def plot_2D(data, N):
    
    curve = bezier.Curve.from_nodes(data.T)
    print(curve)
    
    curve.plot(num_pts=N, color=None, alpha=None, ax=None)
    plt.title("Bezier Curve")
    plt.show()
    plt.close()
    
def plot_3D(data, N):
    
    curve = bezier.Curve.from_nodes(data.T)
    print(curve)
    
    fig = plt.figure(figsize=(10,10))
    fig.suptitle("Bezier Curve")

    ax = fig.add_subplot(projection='3d')

    ## Data Points ##
    nodes = curve.nodes
    ax.scatter(
        nodes[0, :],    # z-coordinates.
        nodes[1, :],    # z-coordinates.
        nodes[2, :],    # z-coordinates.
        color='black'
    )
#     for i in range(len(nodes[0])):
#         ax.text(
#             nodes[0, i],    # z-coordinates.
#             nodes[1, i],    # z-coordinates.
#             nodes[2, i],    # z-coordinates.
#             '{}'.format((i)+1)
#         )
        
    ## Bezier Points ##
    p=100
    ts = np.arange(0, 1, 1/p)
    curve_points = []
    for t in ts:
        curve_points.append(curve.evaluate(t).squeeze())


    curve_points = np.stack(curve_points).T

    ax.plot(
        curve_points[0, :],    # z-coordinates.
        curve_points[1, :],    # z-coordinates.
        curve_points[2, :],    # z-coordinates.
    )

    plt.show()
    plt.close()


if __name__ == '__main__':
    sample2 = np.random.rand(10,2)
    curve  = bezier.Curve.from_nodes(sample2)
    k, curve = curvature(sample2, 30)
    plot_curvature(k)
    plot_2D(sample2, 30)
    
    sample3 = np.random.rand(10,3)
    curve  = bezier.Curve.from_nodes(sample3)
    k, curve = curvature(sample3, 30)
    plot_curvature(k)
    plot_3D(sample3, 30)


