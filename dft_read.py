#%%
from unicodedata import name
import struct
import xml.etree.ElementTree as ET
import numpy as np

def dft_read(dft_filename, verbose=False):
    """Simple reader for BrainSuite's .dft files.
    This reads a .dft file and outputs a the vertices corresponding to the
    curves and the color of each curve.
    Note: if using Python 2, you must do
    "from __future__ import print_function" before calling this function.
    EXAMPLE USAGE
        list_curves, list_colors = dft_read("subj1_curves.dft")
    INPUT:
        > dft_filename: string of the file to be read.
        > verbose: Boolean, set to True if you want verbose output
    OUTPUT:
        > list_curves: list with each element representing one curve using a
            (N x 3) NumPy array where each row is a vertex.
        > list_color: list where n-th element is the a 3-element list of ints
            in the form [R,G,B] corresponding to color of the n-th curve"""

    # list of curves to return,
    list_curves = []

    # if verbose is set, we'll time how long it takes to finish
    if verbose:
        import time
        tic = time.time()

    # open the file to be read in binary mode
    with open(dft_filename, "rb") as fo:

        # for now, I just discard the first 8 bytes which are the text label
        # for the file version
        _ = fo.read(8)[0]
        # discard the next 4 bytes corresponding to the version code.
        _ = fo.read(4)[0]
        # read in as an integer the header size  (4 bytes)
        hdrsize = struct.unpack('i', fo.read(4))[0]
        # start of data of the curve vertices
        dataStart = struct.unpack('i', fo.read(4))[0]
        # start of XML data which gives the color of each curve
        mdoffset = struct.unpack('i', fo.read(4))[0]
        # Discard the next 4 bytes ("pdoffset") since I'm not sure what they do
        _ = struct.unpack('i', fo.read(4))[0]
        # Number of curves (read in as an unsigned int32)
        nContours = struct.unpack('I', fo.read(4))[0]

        if verbose:
            print("the number of curves is: ", nContours)

        # move current file reading position to start of xml block
        fo.seek(mdoffset)

        # calculate size of the XML block
        xml_block_size = dataStart - mdoffset
        # read in XML block
        xml_block = fo.read(xml_block_size)

        # get root element of XML block
        root = ET.fromstring(xml_block)
        # list of all the colors.  Each color is represented by a
        # list of 3 elements corresponding to RGB.
        list_colors = []

        for child in root:
            temp_color = child.attrib['color']
            # convert to float
            temp_color = [float(x) for x in temp_color.split(" ")]
            list_colors.append(temp_color)

        # move current file reading position to start of curve vertex data
        fo.seek(dataStart)

        # loop through every curve
        for curve in range(nContours):
            # number of points in current curve
            num_points = struct.unpack('i', fo.read(4))[0]

            if verbose:
                print("Number of points in current curve:", num_points)

            # in order to read off all the points of a curve in one fell swoop,
            # we need to know the number of (4 byte) floats to read off
            num_floats = num_points*3
            points = struct.unpack('f'*num_floats, fo.read(4*num_floats))

            # make NumPy array from the points.  We reshape it to be a
            # Nx3 array.
            points_arr = np.array(points).reshape((-1, 3))

            # add to the list of curves
            list_curves.append(points_arr)

        if verbose:
            toc = time.time()
            print("Finished processing", nContours,
                  "curves in", toc-tic, "seconds.")

        return list_curves, list_colors


def get_xy_angle(list_curves):
    """input: a list of curves as Nx3 lists of coordinates
        output: list of Nx1 orientations as angles from the positive x direction in the xy plane"""

    list_angles = []
    for i in range(len(list_curves)):
        vectors = []
        for j in range(list_curves[i].shape[0] - 1):
            vectors.append(list_curves[i][j+1,:] - list_curves[i][j,:])
        list_angles.append(np.arctan(vectors[:,1] / vectors[:,0]))
    
    return list_angles

if __name__ == "__main__":
    tensors_fname = '/home/brysongray/data/BrainSuiteTutorialDWI/2523412.dwi.RAS.correct.T1_coord.eig.nii'
    fibers_fname = '/home/brysongray/data/BrainSuiteTutorialDWI/2523412_fiber_tracks'
#%%
    list_curves, list_colors = dft_read(fibers_fname)
# %%
list_angles = get_xy_angle(list_curves)
# %%
