#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 13:04:01 2021

@author: julianna
"""

# Visualize 3D SHAP values
#
# PartitionShap, among other programs, may assign SHAP values
# to each (x, y, z) cell in a 3D model input.
# For example, an image classification model may have RGB inputs
# and we are interested in the SHAP contribution of superpixels
# within each color channel
from sys import exit
import numpy as np
import pyvista as pv
from optparse import OptionParser
from matplotlib.colors import ListedColormap

def main():
    # Replace with appropriate file paths for each type
    #   SHAP pickle (.pickle) files
    file_Path1 = "shap_eurosat_13band_multiband_blur-100x100.pickle"
    #   Numpy (.npz) files
    file_Path2 = "shap_eurosat_13band_multiband_blur-100x100.npz"
    
    print("Are you loading a .pickle file?: ")
    file_type = input("Yes or No (Y/N): ")

    if file_type == 'Y' or file_type == 'y':
        loadPickle(file_Path1)
    else:    
        loadNPZ(file_Path2)

    return 0

# Load .npz file
def loadNPZ(file_Path):
    parser = OptionParser()
    parser.add_option("-f", "--file",
            help="Path to 3D SHAP values (.npz)",
            default= file_Path)
    parser.add_option("-d", "--data_name",
            help="Name of SHAP values in the input SHAP values (.npz) file.",
            default="array_0")
    parser.add_option("-e", "--show_edges",
            help="Show edges of grid elements",
            default=False, action="store_true")
    options, args = parser.parse_args()
    inFile = options.file
    dataName = options.data_name
    showEdges = options.show_edges

    inNPZ = None  # Numpy archive
    values = None # SHAP values


    # Check: can open Numpy file?
    try:
        inNPZ = np.load(inFile)
    except:
        print("Could not load input SHAP values file {}. Ensure valid numpy '.npz'".format(
            inFile))
        exit(1)

    # Check: can read data from name?
    try:
        values = inNPZ[dataName]
    except:
        print("{} is not a file in the numpy (.npz) archive".format(dataName))
        exit(1)

    # Check: is data 3D?
    if (len(values.shape) != 3):
        print("Only supports 3 dimensions. Detected shape of {}".format(values.shape))
        exit(1)

    # Calc min, max
    minValue = np.min(values)
    maxValue = np.max(values)

    print("")
    print("XAI 3D viewer")
    print("--------------")
    print("values file: {}".format(inFile))
    print("  data name: {}".format(dataName))
    print("      shape: {}".format(values.shape))
    print("      range: ({:.4f}, {:.4f})".format(minValue, maxValue))
    print("")

    percent = 0.5

    # Create grid
    grid = buildGrid(values, "Values")

    tgrid = grid.threshold_percent([0.4, 0.6], invert = True)

    p = pv.Plotter()
    
    addMesh(p,grid,"seismic")
    p.show()

# Load .pickle file with data view 
def loadPickle(file_Path):
    import pickle
    import shap
    
    def loadPickle(pickleFile, instanceIdx, classIdx):
        try: 
            shap_values = pickle.load(open(pickleFile, "rb"))
        except pickle.UnpicklingError:
            print("\nSHAP pickle file not found")
            exit(1)
        else:    
            return (shap_values[instanceIdx, :, :, :, classIdx].values, shap_values[instanceIdx, :, :, :, :].base_values, shap_values.output_names, shap_values[instanceIdx, :, :, :, :].data)

    def printShapInfo(shap_values):
        labels = shap_values.output_names
        print(labels)
    
    def toggle_data_view():
        from pyvista import examples
    
        mesh = examples.download_dragon()
        mesh['scalars'] = mesh.points[:, 1]
        mesh.plot(background='white', cpos='xy', cmap='plasma', show_scalar_bar=False)
        return mesh
        
    parser = OptionParser()
    parser.add_option("-p", "--pickle_file",
                      help="Path to pickled SHAP values.",
                      default = file_Path)
    parser.add_option("-i", "--instance_index",
                      help="Index of instance to visualize.",
                      default = 0, type = "int")
    parser.add_option("-c", "--class_index",
                      help="Index of class to visualize.",
                      default = 0, type = "int")
    options, args = parser.parse_args()

    infile = options.pickle_file
    instanceIdx = options.instance_index
    classIdx = options.class_index

    inNPZ = None  # Numpy archive
    values = None # SHAP values

    # Check: can read data?
    #try:
    shap_values, base_values, class_labels, data_values = loadPickle(infile, instanceIdx, classIdx)
    #except:
    #    print("Could not read {} as pickled SHAP values.".format(infile))
    #    exit(1)
    
    # Check: is data 3D?
    if (len(shap_values.shape) != 3):
        print("Only supports 3 dimensions. Detected shape of {}".format(shap_values.shape))
        exit(1)

    # Calc min, max
    minValue = np.min(shap_values)
    maxValue = np.max(shap_values)
    
     # Calc min, max (data)
    minDatValue = np.min(data_values)
    maxDatValue = np.max(data_values)

    print("")
    print("SHAP 3D viewer")
    print("--------------")
    print("values file: {}".format(infile))
    print("      shape: {}".format(shap_values.shape))
    print(" shap range: ({:.4f}, {:.4f})".format(minValue, maxValue))
    print(" data range: ({:.4f}, {:.4f})".format(minDatValue, maxDatValue))
    print("Prediction class: {}".format(class_labels[classIdx]))
    print("           value: {}".format(base_values[classIdx]))
    print("")
 
    #Define/Create cmap color
    from matplotlib.colors import LinearSegmentedColormap
    colors = []
    for l in np.linspace(1, 0, 100):
        colors.append((30./255, 136./255, 229./255,l))
    for l in np.linspace(0, 1, 100):
        colors.append((255./255, 13./255, 87./255,l))
    red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)

    blue_rgb = np.array([0.0, 0.0, 76.0/255])
    red_rgb = np.array([127.0/255, 0.0, 0.0])
    white_rgb = np.array([1.,1.,1.])

    colors = []
    for alpha in np.linspace(1, 0, 100):
        c = blue_rgb * alpha + (1 - alpha) * white_rgb
        colors.append(c)
    for alpha in np.linspace(0, 1, 100):
        c = red_rgb * alpha + (1 - alpha) * white_rgb
        colors.append(c)
    red_white_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)
    
    # Create Plotter
    p = pv.Plotter(shape=(1,2))
    
    p.subplot(0,0)
    # Create grid for SHAP values
    grid1 = buildGrid(shap_values, "SHAP Values")
    tgrid = grid1.threshold_percent([0.4, 0.6], invert = True)
    
    addMesh(p,grid1,red_white_blue)
    
    p.subplot(0,1)
    # Create grid for Data Values
    grid2 = buildGrid(data_values, "Data Values")
    tgrid = grid2.threshold_percent([0.4, 0.6], invert = True)
    addMesh(p,grid2,"gray")
    
    # Sync view
    p.link_views()
    
    #Show Plot    
    p.show()

# Build grid to display values
def buildGrid(values, valType, origin=(0, 0, 0), spacing=(10, 10, 10)):
    # Spatial reference
    grid = pv.UniformGrid()

    # Grid dimensions (shape + 1)
    grid.dimensions = np.array(values.shape) + 1

    # Spatial reference params
    grid.origin = origin
    
    grid.spacing = spacing

    # Grid data
    grid.cell_arrays[valType] = values.flatten(order="F")
    return grid

# Add mesh and threshold sliders to plot
def addMesh(plot, grid, color):
    # Very faint grid mesh
    plot.add_mesh(grid,
                style="wireframe",
                opacity=0.075,
                cmap=color,
                )

    plot.add_mesh_threshold(grid,
                          invert=True,
                          pointa=(0.1, 0.9),
                          pointb=(0.45, 0.9),
                          title = "Lower threshold",
                          cmap=color)

    plot.add_mesh_threshold(grid,
                          pointa=(0.55, 0.9),
                          pointb=(0.9, 0.9),
                          invert=False,
                          title = "Higher threshold",
                          cmap=color)

if __name__ == "__main__":
    main()