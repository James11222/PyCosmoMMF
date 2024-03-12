import numpy as np

def test_print():
    """
    A simple function to test the installation of the package.
    """
    print("Hello World! This is an installation test. If you see this, the installation was successful.")

def shrink(data, new_size):
    """
    A simple function to shrink a 3D array by summing over blocks of new_size.
    """
    xs = new_size
    ys = new_size
    zs = new_size
    return data.reshape(xs, data.shape[0]//xs, 
                        ys, data.shape[1]//ys, 
                        zs, data.shape[2]//zs).sum(axis=1).sum(axis=2).sum(axis=3)