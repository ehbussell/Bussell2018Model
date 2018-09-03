"""Generate new network and save to file."""

import numpy as np
import matplotlib.pyplot as plt

def make_new_network():
    """Generate new network with nodes randomly positioned around 3 centres.

    Nodes cannot be closer than min_node_distance
    Numbers of high/low risk group hosts are chosen by binomial trial
    """
    centres_x = np.array([3, 5, 7])
    centres_y = np.array([7, 5, 3])

    n_nodes_region = [20, 15, 20]
    min_node_distance = 0.2
    node_pop = 30
    high_prop = 0.1
    region_names = ["A", "B", "C"]

    # Initialise structure
    points_x = [np.full(n_nodes_region[i], -100.0) for i in range(len(region_names))]
    points_y = [np.full(n_nodes_region[i], -100.0) for i in range(len(region_names))]

    for region in range(len(centres_x)):
        for node in range(n_nodes_region[region]):
            valid_point = False
            while not valid_point:
                radius = np.random.rand()
                theta = 2 * np.pi * np.random.rand()
                x = np.round(centres_x[region] + radius*np.cos(theta), 2)
                y = np.round(centres_y[region] + radius*np.sin(theta), 2)
                min_dist = min([
                    np.amin(np.sqrt(np.square(points_x[i] - x) + np.square(points_y[i] - y)))
                    for i in range(len(region_names))])
                if min_dist > min_node_distance:
                    valid_point = True
                    points_x[region][node] = x
                    points_y[region][node] = y

    # Sort region A L->R, region B top->bottom, and region C L->R for later coupling
    a_sort = np.argsort(points_x[0])
    b_sort = np.argsort(points_y[1])[::-1]
    c_sort = np.argsort(points_x[2])

    points_x[0] = points_x[0][a_sort]
    points_x[1] = points_x[1][b_sort]
    points_x[2] = points_x[2][c_sort]

    points_y[0] = points_y[0][a_sort]
    points_y[1] = points_y[1][b_sort]
    points_y[2] = points_y[2][c_sort]

    # Save to node file
    with open("node_file.txt", "w") as outfile:
        count = 0
        for region in range(len(centres_x)):
            for node in range(n_nodes_region[region]):
                pop_high = np.random.binomial(node_pop, high_prop)
                pop_low = node_pop - pop_high
                outfile.write("{0} {1} {2} {3} {4} 0 0 0 {5} 0 0 0\n".format(
                    count, points_x[region][node], points_y[region][node], region_names[region],
                    pop_high, pop_low))
                count += 1

    # Plot network for reference
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(centres_x, centres_y, "rx")

    for i in range(len(region_names)):
        ax.plot(points_x[i], points_y[i], "gx")

    plt.show()

if __name__ == "__main__":
    make_new_network()
