#!/usr/bin/env python3
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


### dictionary to match purpose to CSV header
h_dict = {
        "prec" : "Precision",
        "GOPS": "[GOPs/s]", # Will change to '[GOP/s]
        "BW": "BW [GB/s]",
        "time": "time [ms]",
        "comps": "computations",
        "size": "data [Bytes]",
        "oiters": "Outer Its",
        "iiters": "Inner Its",
        "citers": "Comp Its",
        "#elms": "# Elements"
        # "OP_pb": "Operations per Byte [OP / Byte]"
        }

def read_csv(path=None):
    """
    Opens the CSV file in 'path' and returns 2 dictionaries:
    1. The key is the precision it was performed in, the value is the list of a list
       of column entries of the csv file (the lines are sorted according to number
       of computations)
    2. The key is the same as in h_dict, the value is the index of the row
       array / list for the correesponding key
    """
    if path == None:
        path = "../20201125_A100_roofline_d3.csv"
    with open(path, 'r') as f:
        print("The csv file is opened")
        csv_f = csv.reader(f, delimiter=';', skipinitialspace=True)
        header = next(csv_f)
        print("CSV header: {}".format(header))
        
        i_dict = {}
        for key, val in h_dict.items():
            for i in range(len(header)):
                if header[i] == val:
                    i_dict[key] = i
        print("Resulting index dictionary: {}".format(i_dict))

        data = []

        for r in csv_f:
            data.append(r)

    prec_set = set()
    for line in data:
        prec_set.add(line[i_dict["prec"]])

    # Filter input data and put it into a dictionary
    data_dict = {}
    for line in data:
        if not line[i_dict["prec"]] in data_dict:
            data_dict[line[i_dict["prec"]]] = [line]
        else:
            data_dict[line[i_dict["prec"]]].append(line)
    for key, value in data_dict.items():
        # Sort data by number of computations
        value.sort(key=lambda x:int(x[i_dict["comps"]]))
    return data_dict, i_dict

def filter_data(data, predicate):
    # Filter input data and put it into a dictionary
    filtered_dict = {}
    for key, lines in data.items():
        filtered_dict[key] = []
        for line in lines:
            if predicate(line):
                filtered_dict[key].append(line)
    return filtered_dict




############################### Actual Plotting ###############################
### Color definition
myblue    = (0, 0.4470, 0.7410);
myorange  = (0.8500, 0.3250, 0.0980);
myyellow  = (0.9290, 0.6940, 0.1250);
mymagenta = (0.4940, 0.1840, 0.5560);
mygreen   = (0.4660, 0.6740, 0.1880);
mycyan    = (0.3010, 0.7450, 0.9330);
myred     = (0.6350, 0.0780, 0.1840);
myblack   = (0.2500, 0.2500, 0.2500);
mybrown   = (0.6500, 0.1600, 0.1600);

### Other globals
LineWidth = 1
MarkerSize = 8



def create_fig_ax():
    """
    Creates a tuple of figure and axis for future plots.
    The size, the visibility of the grid and the log-scale of x and y is preset
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    grid_minor_color = (.9, .9, .9)
    grid_major_color = (.8, .8, .8)
    ax.grid(True, which="major", axis="both", linestyle='-', linewidth=1, color=grid_major_color)
    ax.grid(True, which="minor", axis="both", linestyle=':', linewidth=1, color=grid_minor_color)
    ax.loglog()
    return fig, ax


def plot_figure(fig, file_path):
    """Plots the given figure fig as various formats with a base-name of file_path"""
    p_bbox = "tight"
    p_pad = 0
    p_dpi = 300  # Only useful for non-scalable formats
    with PdfPages(file_path+".pdf") as export_pdf:
        export_pdf.savefig(fig, dpi=p_dpi, bbox_inches=p_bbox, pad_inches=p_pad)
    fig.savefig(file_path+".svg", dpi=p_dpi, bbox_inches=p_bbox, pad_inches=p_pad, format="svg")
    fig.savefig(file_path+".png", dpi=p_dpi, bbox_inches=p_bbox, pad_inches=p_pad, format="png")


def plot_for_all(ax, data, x_key, y_key):
    """
    plots given x and y keys for all precisions of interest on the axis ax.
    """
    markers = ('X', 'P', 'x', '+')
    colors = (mygreen, myblue, myorange, myyellow)
    precs = ("double", "float",  "Ac<3, d, d>", "Ac<3, d, f>")
    labels = ("double precision", "single precision",  "Accessor<d, d>", "Accessor<d, s>")
    for i in range(len(precs)):
        ax.plot(data[precs[i]][x_key], data[precs[i]][y_key], label=labels[i],
                marker=markers[i], color=colors[i], linewidth=LineWidth,
                markersize=MarkerSize)


if __name__ == "__main__":
    # Change to the directory where the script is placed
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Make sure the folder `plots` exists
    if not os.path.exists('plots'):
        os.makedirs('plots')

    data_dict, i_dict = read_csv("../20201125_A100_roofline_d3.csv")

    filt_lambda = lambda x : int(x[i_dict["oiters"]]) == 4 and int(x[i_dict["iiters"]]) == 8
    data_dict = filter_data(data_dict, filt_lambda)

    # Generate data for plotting for all available precisions
    plot_data = {}
    # TODO: when available, use "elems" instead
    #       Here, it is assumed that "double" is always available, and all have the same number of elements
    num_elems = int(int(data_dict["double"][0][i_dict["size"]]) / 8)
    for key, lines in data_dict.items():
        current_dict = {"OP_pb": [], "OP_pv": [], "GOPS": [], "BW": []}
        for line in lines:
            size_i = i_dict["size"]
            bw_i = i_dict["BW"]
            ops_i = i_dict["GOPS"]
            comp_i = i_dict["comps"]
            current_dict["OP_pb"].append(int(line[comp_i]) / int(line[size_i]))
            current_dict["OP_pv"].append(int(line[comp_i]) / num_elems)
            current_dict["GOPS"].append(float(line[ops_i]))
            current_dict["BW"].append(float(line[bw_i]))

        plot_data[key] = current_dict


    fig, ax = create_fig_ax()
    plot_for_all(ax, plot_data, "OP_pb", "BW")
    ax.axhline(1555.0, linestyle='--', marker='', linewidth=LineWidth,
            color=myblack, label="Peak double performance")

    ax.set_xlabel("Arithmetic Intensity [FLOP / Byte]")
    ax.set_ylabel("Bandwidth [GB / s]")
    #ax.legend(loc="best")
    ax.legend(loc="lower left")
    plot_figure(fig, "plots/roofline_bandwidth_pai_d3_a100")


    fig, ax = create_fig_ax()
    plot_for_all(ax, plot_data, "OP_pv", "BW")
    ax.axhline(1555.0, linestyle='--', marker='', linewidth=LineWidth,
            color=myblack, label="Peak double performance")

    ax.set_xlabel("Arithmetic Intensity [FLOP / Value]")
    ax.set_ylabel("Bandwidth [GB / s]")
    ax.legend(loc="lower left")
    plot_figure(fig, "plots/roofline_bandwidth_pv_d3_a100")


    fig, ax = create_fig_ax()
    plot_for_all(ax, plot_data, "OP_pb", "GOPS")

    ax.axhline(9746.0, linestyle='--', marker='', linewidth=LineWidth,
            color=myblack, label="Peak double performance")
    ax.axhline(19490.0, linestyle='--', marker='', linewidth=LineWidth,
            color=mybrown, label="Peak single performance")

    ax.set_xlabel("Arithmetic Intensity [FLOP / Byte]")
    ax.set_ylabel("Compute Performance [FLOP / s]")
    #ax.legend(loc="best")
    ax.legend(loc="lower right")
    plot_figure(fig, "plots/roofline_performance_pai_d3_a100")


    fig, ax = create_fig_ax()
    plot_for_all(ax, plot_data, "OP_pv", "GOPS")

    ax.axhline(9746.0, linestyle='--', marker='', linewidth=LineWidth,
            color=myblack, label="Peak double performance")
    ax.axhline(19490.0, linestyle='--', marker='', linewidth=LineWidth,
            color=mybrown, label="Peak single performance")

    ax.set_xlabel("Arithmetic Intensity [FLOP / Value]")
    ax.set_ylabel("Compute Performance [FLOP / s]")
    #ax.legend(loc="best")
    ax.legend(loc="lower right")
    plot_figure(fig, "plots/roofline_performance_pv_d3_a100")
