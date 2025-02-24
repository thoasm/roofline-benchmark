#!/usr/bin/env python3
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages


# Peak data: flops in GFLOPS; BW in GB/s
plot_info = {}

plot_folder = "./plots/"
filt_lambda_12 = lambda x : int(x[i_dict["oiters"]]) == 1 and int(x[i_dict["iiters"]]) == 2
filt_lambda_14 = lambda x : int(x[i_dict["oiters"]]) == 1 and int(x[i_dict["iiters"]]) == 4
filt_lambda_48 = lambda x : int(x[i_dict["oiters"]]) == 4 and int(x[i_dict["iiters"]]) == 8
filt_lambda_18 = lambda x : int(x[i_dict["oiters"]]) == 1 and int(x[i_dict["iiters"]]) == 8
filt_lambda_416 = lambda x : int(x[i_dict["oiters"]]) == 4 and int(x[i_dict["iiters"]]) == 16

plot_info["radeon7"] = {
        "peak_fp64": 3360.0,
        "peak_fp32": 13440.0,
        "peak_fp16": 26880.0,
        "peak_bw": 1024.0,
        #"file": "../20210315_1003_radeon7.csv",
        #"file": "../20210316_1701_radeon7.csv",
        #"file": "../20210316_1711_radeon7.csv",
        #"file": "../20210316_1712_radeon7.csv",
        #"file": "../20210316_1721_radeon7.csv",
        #"file": "../20210316_1712_radeon7.csv",
        "file": "../20210316_1720_radeon7.csv", # best one so far with 18
        "prefix": "radeon7_",
        "filter": filt_lambda_18,
        }
plot_info["mi100"] = {
        "peak_fp64": 11500.0,
        "peak_fp32": 23100.0,
        "peak_fp16": 18460.0,
        "peak_bw": 1200.0,
        "file": "../20210312_2000_MI100.csv",
        "prefix": "mi100_",
        "filter": filt_lambda_18,
        }
plot_info["h100"] = {
        "peak_fp64": 26000.0,
        "peak_fp32": 51000.0,
        "peak_fp16": 1513000.0,
        "peak_bw": 2000.0,
        #"file": "../2024-03-15_H100_hexane.csv",
        "file": "../2024-06-04_1150_H100_hexane_frsz.csv",
        "prefix": "h100_",
        "filter": filt_lambda_14,
        }
plot_info["a100"] = {
        "peak_fp64": 9746.0,
        "peak_fp32": 19490.0,
        "peak_fp16": 77970.0,
        "peak_bw": 1935.0,
        #"file": "../20201125_A100_roofline_d3.csv",
        #"file": "../20230517_0900_A100_guyot_frsz2.csv",
        "file": "../20230517_1030_A100_guyot_frsz2.csv",
        "prefix": "a100_",
        "filter": filt_lambda_14,
        }
plot_info["v100"] = {
        "peak_fp64": 7800.0,
        "peak_fp32": 15700.0,
        "peak_fp16": 125000.0, # TENSOR performance, not pure FP16
        "peak_bw": 900.0,
        #"file": "../20210311_1730_V100_summit.csv",
        "file": "../20220124_1931_V100_Summit.csv",
        "prefix": "v100_",
        "filter": filt_lambda_48,
        }
plot_info["3900X"] = {
        "peak_fp64": 792.0, # Estimated with https://github.com/Mysticial/Flops
        "peak_fp32": 1579.0, # same as above
        "peak_fp16": 0.0, # No fp16 unit, therefore, 0
        "peak_bw": 51.2,
        "file": "../20210309_0435_Ryzen3900X_OMP24.csv",
        "prefix": "3900X_",
        "filter": filt_lambda_48,
        }
plot_info["bwuni-rw"] = {
# 2x Intel Xeon Gold 6230 (full specs: https://wiki.bwhpc.de/e/BwUniCluster_2.0_Hardware_and_Architecture)
        "peak_fp64": 0.0,
        "peak_fp32": 0.0,
        "peak_fp16": 0.0,
        "peak_bw": 2 * 6 * 23.47,
        "file": "../20210312_2200_bwuni_2s80t_rw.csv",
        "prefix": "bwuni-rw_",
        "filter": filt_lambda_416,
        }
plot_info["bwuni-ro"] = plot_info["bwuni-rw"].copy()
plot_info["bwuni-ro"]["prefix"] = "bwuni-ro_"
plot_info["bwuni-ro"]["file"] = "../20210312_2230_bwuni_2s80t_ro.csv"

plot_info["AMD-7742-rw"] = {
# 2x AMD EPYC 7742 (on ForHLR2)
# Processor specs: https://www.amd.com/en/products/cpu/amd-epyc-7742
        "peak_fp64": 0.0,
        "peak_fp32": 0.0,
        "peak_fp16": 0.0,
        "peak_bw": 2 * 204.8,
        "file": "../20210324_1130_AMD_EPYC_7742_2s256t_rw.csv",
        "prefix": "EPYC-rw_",
        "filter": filt_lambda_416,
        }
plot_info["AMD-7742-ro"] = plot_info["AMD-7742-rw"].copy()
plot_info["AMD-7742-ro"]["prefix"] = "EPYC-ro_"
plot_info["AMD-7742-ro"]["file"] = "../20210324_1020_AMD_EPYC_7742_2s256t_ro.csv"


plot_info["fhlr2"] = {
        "peak_fp64": 0.0,
        "peak_fp32": 0.0,
        "peak_fp16": 0.0,
        "peak_bw": 0.0,
        "file": "../20210310_1000_fhlr2_1s_intel.csv",
        "prefix": "fhlr2_",
        "filter": filt_lambda_48,
        }

"""
plot_info["arm_xavier"] = {
        "peak_fp64": 0.0,
        "peak_fp32": 0.0,
        "peak_fp16": 0.0,
        "peak_bw": 0.0,
        "file": "../20210422_1430_jetson_xavier_benchmark_run.csv",
        "prefix": "xavier_",
        "filter": filt_lambda_416,
        }
"""

plot_list = [
        #"mi100",
        #"radeon7",
        "h100",
        #"a100",
        #"v100",
        #"bwuni-rw",
        #"AMD-7742-rw",
        #"AMD-7742-ro",
        #"arm_xavier",
        ]

### dictionary to match purpose to CSV header
h_dict = {
        "prec" : "Precision",
        #"GOPS": "[GOPs/s]", # Old value, changed to `[GOP/s]`
        "GOPS": "[GOP/s]", # New value
        "BW": "BW [GB/s]",
        "time": "time [ms]",
        "comps": "computations",
        "size": "data [Bytes]",
        "oiters": "Outer Its",
        "iiters": "Inner Its",
        "citers": "Comp Its",
        "#elms": "# Elements",
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
        raise Exception("No filename specified! Unable to read file.")
    with open(path, 'r') as f:
        print("The csv file <{}> is opened".format(path))
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

dark_mod = 2
mydarkred     = (0.6350 / dark_mod, 0.0780 / dark_mod, 0.1840 / dark_mod);
mydarkgreen   = (0.4660 / dark_mod, 0.6740 / dark_mod, 0.1880 / dark_mod);
mydarkblue    = (0, 0.4470 / dark_mod, 0.7410 / dark_mod);

### Other globals
LineWidth = 1
MarkerSize = 8


precisions_to_print = ("double",
        "float",
        #"Ac<3, d, d>", "Ac<1, d, d>",
        "Ac<3, d, f>", "Ac<1, d, f>",
        "Ac<3, d, p32>", "Ac<1, d, p32>",
        "Ac<3, f, p16>", "Ac<1, f, p16>",
        #"frsz2-16", "frsz2-21", "frsz2-32"
        )
precision_details = {
        "double": {
            "marker": 'X',
            "color": mygreen,
            "label": "fp64",
            },
        "float": {
            "marker": 'P',
            "color": myblue,
            "label": "fp32",
            },
        "Ac<3, d, d>": {
            "marker": 'x',
            "color": myorange,
            "label": "Acc<fp64, fp64>",
            },
        "Ac<3, d, f>": {
            "marker": '+',
            "color": myyellow,
            "label": "Acc<fp64, fp32>",
            },
        "Ac<3, d, p32>": {
            "marker": 'D',
            "color": mymagenta,
            "label": "Acc<fp64, posit32>",
            },
        "Ac<3, f, p16>": {
            "marker": 'o',
            "color": myblack,
            "label": "Acc<fp32, posit16>",
            },
        "frsz2-16": {
            "marker": '1',
            "color": mybrown,
            "label": "Acc<fp64, frsz2_16>",
            },
        "frsz2-21": {
            "marker": '+',
            "color": myorange,
            "label": "Acc<fp64, frsz2_21>",
            },
        "frsz2-32": {
            "marker": '3',
            "color": myyellow,
            "label": "Acc<fp64, frsz2_32>",
            },
        }
precision_details["Ac<1, d, d>"] = precision_details["Ac<3, d, d>"]
precision_details["Ac<1, d, f>"] = precision_details["Ac<3, d, f>"]
precision_details["Ac<1, d, p32>"] = precision_details["Ac<3, d, p32>"]
precision_details["Ac<1, f, p16>"] = precision_details["Ac<3, f, p16>"]


def create_fig_ax():
    """
    Creates a tuple of figure and axis for future plots.
    The size, the visibility of the grid and the log-scale of x and y is preset
    """
    fig = Figure(figsize=(10, 4)) # Properly garbage collected
    #fig = Figure(figsize=(6, 3)) # Properly garbage collected
    ax = fig.add_subplot()
    #fig, ax = plt.subplots(figsize=(10, 4)) # NOT garbage collected!
    grid_minor_color = (.9, .9, .9)
    grid_major_color = (.8, .8, .8)
    ax.grid(True, which="major", axis="both", linestyle='-', linewidth=1, color=grid_major_color)
    ax.grid(True, which="minor", axis="both", linestyle=':', linewidth=1, color=grid_minor_color)
    ax.loglog()
    return fig, ax


def plot_figure(fig, file_name, plot_prefix):
    """Plots the given figure fig as various formats with a base-name of file_name.
    plot_folder will be used as the filder for the file; plot_prefix will be the
    prefix of each file."""

    file_path = plot_folder + plot_prefix + file_name
    p_bbox = "tight"
    p_pad = 0.01
    p_dpi = 300  # Only useful for non-scalable formats
    with PdfPages(file_path+".pdf") as export_pdf:
        export_pdf.savefig(fig, dpi=p_dpi, bbox_inches=p_bbox, pad_inches=p_pad)
    fig.savefig(file_path+".svg", dpi=p_dpi, bbox_inches=p_bbox, pad_inches=p_pad, format="svg")
    #fig.savefig(file_path+".png", dpi=p_dpi, bbox_inches=p_bbox, pad_inches=p_pad, format="png")


def plot_for_all(ax, data, x_key, y_key):
    """
    plots given x and y keys for all precisions of interest on the axis ax.
    """
    for prec in precisions_to_print:
        if prec not in data:
            continue
        ax.plot(data[prec][x_key], data[prec][y_key], label=precision_details[prec]["label"],
                marker=precision_details[prec]["marker"], color=precision_details[prec]["color"],
                linewidth=LineWidth, markersize=MarkerSize)
    """ To get / set x- and y-limits:
    ax.set_xlim(0.7070722721781199, 1449.6396483523677)
    ax.set_ylim(148.24516110946269, 24024.62127583265)
    xl, xr = ax.get_xlim()
    yl, yr = ax.get_ylim()
    print("xlim: ({}, {}); ylim: ({}, {})".format(xl, xr, yl, yr));
    """

def add_table(ax, plot_data, value_key, colLabel, value_lambda):
    table_row_labels = []
    table_vals = []
    for prec in precisions_to_print:
        if prec not in plot_data:
            continue
        data = plot_data[prec]
        if prec in precision_details:
            print("{}: key: {}; data[key] = {}".format(prec, value_key, data[value_key]))
            label = precision_details[prec]["label"]
            value = value_lambda(data[value_key])
            table_row_labels.append(label)
            table_vals.append([value])

    #print(table_row_labels, table_vals)
    new_table = ax.table(cellText=table_vals, rowLabels=table_row_labels,
            colLabels=[colLabel], colWidths=[0.15], cellLoc="right",
            colLoc="left", rowLoc="left", loc="lower center", zorder=3,
            edges='closed')



if __name__ == "__main__":
    bw_color = myblack
    fp64_color = mydarkgreen
    fp32_color = mydarkblue

    # Change to the directory where the script is placed
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Make sure the plot folder exists
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    for device in plot_list:
        info = plot_info[device]
        data_dict, i_dict = read_csv(info["file"])
        print("Keys: {}".format([data_dict.keys()]))

        data_dict = filter_data(data_dict, info["filter"])

        # Generate data for plotting for all available precisions
        plot_data = {}
        use_global_elms = False
        # When available, use "elems"
        if "#elms" not in i_dict:
            # Here, it is assumed that "double" is always available, and all have the same number of elements
            use_global_elms = True
            glob_num_elems = int(int(data_dict["double"][0][i_dict["size"]]) / 8)
        for key, lines in data_dict.items():
            current_dict = {"OP_pb": [], "OP_pv": [], "GOPS": [], "BW": []}
            for line in lines:
                size_i = i_dict["size"]
                bw_i = i_dict["BW"]
                ops_i = i_dict["GOPS"]
                comp_i = i_dict["comps"]
                if use_global_elms:
                    num_elems = glob_num_elems
                else:
                    num_elems = int(line[i_dict["#elms"]])

                current_dict["OP_pb"].append(int(line[comp_i]) / int(line[size_i]))
                current_dict["OP_pv"].append(int(line[comp_i]) / num_elems)
                current_dict["GOPS"].append(float(line[ops_i]))
                current_dict["BW"].append(float(line[bw_i]))

            plot_data[key] = current_dict


        fig, ax = create_fig_ax()
        plot_for_all(ax, plot_data, "OP_pb", "BW")
        if "peak_bw" in info and info["peak_bw"] > 0:
            ax.axhline(info["peak_bw"], linestyle='--', marker='', linewidth=LineWidth,
                    color=bw_color, label="Peak bandwidth")

        ax.set_xlabel("Arithmetic Intensity [FLOP/Byte]")
        ax.set_ylabel("Bandwidth [GB/s]")
        #ax.legend(loc="best")
        ax.legend(loc="lower left", ncols=1)
        plot_figure(fig, "roofline_bandwidth_pai_d3", info["prefix"])


        fig, ax = create_fig_ax()
        plot_for_all(ax, plot_data, "OP_pv", "BW")
        if "peak_bw" in info and info["peak_bw"] > 0:
            ax.axhline(info["peak_bw"], linestyle='--', marker='', linewidth=LineWidth,
                    color=bw_color, label="Peak bandwidth")

        ax.set_xlabel("Arithmetic Intensity [FLOP/Value]")
        ax.set_ylabel("Bandwidth [GB/s]")
        ax.legend(loc="lower left", ncols=1)
        plot_figure(fig, "roofline_bandwidth_pv_d3", info["prefix"])


        fig, ax = create_fig_ax()
        plot_for_all(ax, plot_data, "OP_pb", "GOPS")

        if "peak_fp64" in info and info["peak_fp64"] > 0:
            ax.axhline(info["peak_fp64"], linestyle='--', marker='', linewidth=LineWidth,
                    color=fp64_color, label="Peak fp64 perf.")
        if "peak_fp32" in info and info["peak_fp32"] > 0:
            ax.axhline(info["peak_fp32"], linestyle='-.', marker='', linewidth=LineWidth,
                    color=fp32_color, label="Peak fp32 perf.")

        #add_table(ax, plot_data, "GOPS", "Peak GFLOP/s", lambda x: "{:,}".format(round(max(x))))
        ax.set_xlabel("Arithmetic Intensity [FLOP/Byte]")
        ax.set_ylabel("Compute Performance [GFLOP/s]")
        #ax.legend(loc="best")
        ax.legend(loc="lower right", ncols=1)
        plot_figure(fig, "roofline_performance_pai_d3", info["prefix"])


        fig, ax = create_fig_ax()
        plot_for_all(ax, plot_data, "OP_pv", "GOPS")

        if "peak_fp64" in info and info["peak_fp64"] > 0:
            ax.axhline(info["peak_fp64"], linestyle='--', marker='', linewidth=LineWidth,
                    color=fp64_color, label="Peak fp64 perf.")
        if "peak_fp32" in info and info["peak_fp32"] > 0:
            ax.axhline(info["peak_fp32"], linestyle='-.', marker='', linewidth=LineWidth,
                    color=fp32_color, label="Peak fp32 perf.")

        #add_table(ax, plot_data, "GOPS", "Peak GFLOP/s", lambda x: "{:,}".format(round(max(x))))
        ax.set_xlabel("Arithmetic Intensity [FLOP/Value]")
        ax.set_ylabel("Compute Performance [GFLOP/s]")
        #ax.legend(loc="best")
        ax.legend(loc="lower right", ncols=1)
        plot_figure(fig, "roofline_performance_pv_d3", info["prefix"])
