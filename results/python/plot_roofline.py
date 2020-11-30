#!/usr/bin/env python3
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


h_dict = {
        "prec" : "Precision",
        "GOP": "[GOPs/s]", # Will change to '[GOP/s]
        "BW": "BW [GB/s]",
        "time": "time [ms]",
        "comps": "computations",
        "size": "data [Bytes]",
        "oiters": "Outer Its",
        "iiters": "Inner Its",
        "citers": "Comp Its",
        "#elms": "# Elements"
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
        print(header)
        
        i_dict = {}
        for key, val in h_dict.items():
            for i in range(len(header)):
                if header[i] == val:
                    i_dict[key] = i
        print(i_dict)

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

### Other globals
LineWidth = 1
MarkerSize = 8



os.chdir(os.path.dirname(os.path.abspath(__file__)))

if not os.path.exists('plots'):
    os.makedirs('plots')

### Save as PDF
def plot_figure(fig, name):
    with PdfPages(name) as export_pdf:
        export_pdf.savefig(fig, bbox_inches='tight', pad_inches=0)

def create_fig_ax():
    fig, ax = plt.subplots(figsize=(10, 4))
    grid_minor_color = (.9, .9, .9)
    grid_major_color = (.8, .8, .8)
    ax.grid(True, which="major", axis="both", linestyle='-', linewidth=1, color=grid_major_color)
    ax.grid(True, which="minor", axis="both", linestyle=':', linewidth=1, color=grid_minor_color)
    ax.loglog()
    return fig, ax

def plot_4(ax, xy1, xy2, xy3, xy4):
    ax.plot(xy1[0], xy1[1], label=xy1[2], marker='X', color=mygreen, linewidth=LineWidth, markersize=MarkerSize)
    ax.plot(xy2[0], xy2[1], label=xy2[2],  marker='P', color=myblue, linewidth=LineWidth, markersize=MarkerSize)
    ax.plot(xy3[0], xy3[1], label=xy3[2],  marker='x', color=myorange, linewidth=LineWidth, markersize=MarkerSize)
    ax.plot(xy4[0], xy4[1], label=xy4[2],  marker='+', color=myyellow, linewidth=LineWidth, markersize=MarkerSize)


if __name__ == "__main__":
    data_dict, i_dict = read_csv()

    filt_lambda = lambda x : int(x[i_dict["oiters"]]) == 4 and int(x[i_dict["iiters"]]) == 8
    data_dict = filter_data(data_dict, filt_lambda)

    # Prepare data later used for plotting
    dbl = data_dict["double"]
    dbl_op_p_byte = [int(line[i_dict["comps"]]) / int(line[i_dict["size"]]) for line in dbl]
    dbl_op_p_val = [x / 8 for x in dbl_op_p_byte]
    dbl_bw = [float(line[i_dict["BW"]]) for line in dbl]
    dbl_gop = [float(line[i_dict["GOP"]]) for line in dbl]

    flt = data_dict["float"]
    flt_op_p_byte = [int(line[i_dict["comps"]]) / int(line[i_dict["size"]]) for line in flt]
    flt_op_p_val = [x / 4 for x in flt_op_p_byte]
    flt_bw = [float(line[i_dict["BW"]]) for line in flt]
    flt_gop = [float(line[i_dict["GOP"]]) for line in flt]

    add = data_dict["Ac<3, d, d>"]
    add_op_p_byte = [int(line[i_dict["comps"]]) / int(line[i_dict["size"]]) for line in add]
    add_op_p_val = [x / 8 for x in add_op_p_byte]
    add_bw = [float(line[i_dict["BW"]]) for line in add]
    add_gop = [float(line[i_dict["GOP"]]) for line in add]

    adf = data_dict["Ac<3, d, f>"]
    adf_op_p_byte = [int(line[i_dict["comps"]]) / int(line[i_dict["size"]]) for line in adf]
    adf_op_p_val = [x / 4 for x in adf_op_p_byte]
    adf_bw = [float(line[i_dict["BW"]]) for line in adf]
    adf_gop = [float(line[i_dict["GOP"]]) for line in adf]


    fig, ax = create_fig_ax()
    plot_4(
        ax, (dbl_op_p_byte, dbl_bw, "double precision"),
        (flt_op_p_byte, flt_bw, "single precision"),
        (add_op_p_byte, add_bw, "Accessor<d, d>"),
        (adf_op_p_byte, adf_bw, "Accessor<d, s>"))
    ax.axhline(1555.0, linestyle='--', marker='', linewidth=LineWidth, color=myblack, label="Peak double performance")

    ax.set_xlabel("FLOP / Byte")
    ax.set_ylabel("GB / s")
    #ax.legend(loc="best")
    ax.legend(loc="lower left")
    plot_figure(fig, "plots/roofline_bandwidth_d3_a100.pdf")

    ### Start second plot:
    fig, ax = create_fig_ax()
    plot_4(
        ax, (dbl_op_p_byte, dbl_gop, "double precision"),
        (flt_op_p_byte, flt_gop, "single precision"),
        (add_op_p_byte, add_gop, "Accessor<d, d>"),
        (adf_op_p_byte, adf_gop, "Accessor<d, s>"))

    ax.axhline(9746.0, linestyle='--', marker='', linewidth=LineWidth, color=myblack, label="Peak double performance")
    ax.axhline(19490.0, linestyle='--', marker='', linewidth=LineWidth, color=myblack, label="Peak single performance")

    ax.set_xlabel("FLOP / Byte")
    ax.set_ylabel("FLOP / s")
    #ax.legend(loc="best")
    ax.legend(loc="lower right")
    plot_figure(fig, "plots/roofline_performance_d3_a100.pdf")
