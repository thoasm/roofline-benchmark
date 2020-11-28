#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


with open("../20201125_A100_roofline_d3.csv", 'r') as f:
    print("The csv file is opened")
    csv = csv.reader(f, delimiter=';', skipinitialspace=True)
    header = next(csv)
    print(header)
    
    prec_idx=-1
    gop_idx=-1
    bw_idx=-1
    comp_idx=-1
    size_idx=-1
    inner_idx=-1
    outer_idx=-1
    # Figure out the indices for important fields
    for i in range(len(header)):
        if header[i].startswith("Precision"):
            prec_idx = i
        elif header[i].startswith("[GOP"):
            gop_idx = i
        elif header[i].startswith("BW"):
            bw_idx = i
        elif header[i].startswith("computations"):
            comp_idx = i
        elif header[i].startswith("data"):
            size_idx = i
        elif header[i].startswith("Outer"):
            outer_idx = i
        elif header[i].startswith("Inner"):
            inner_idx = i
    data=[]
    for r in csv:
        data.append(r)

print("{} {} {} {} {} {} {}".format(header[prec_idx], header[gop_idx], header[bw_idx], header[comp_idx], header[size_idx], header[inner_idx], header[outer_idx]))
prec_set = set()
for line in data:
    prec_set.add(line[prec_idx])

# Filter input data and put it into a dictionary
data_dict = {}
for line in data:
    if (int(line[outer_idx]) == 4 and int(line[inner_idx]) == 8):
        if not line[prec_idx] in data_dict:
            data_dict[line[prec_idx]] = [line]
        else:
            data_dict[line[prec_idx]].append(line)
for key, value in data_dict.items():
    # Sort data by number of computations
    value.sort(key=lambda x:int(x[comp_idx]))
    #print("Data for {}".format(key))
    #for l in value:
    #    print("\t{}".format(l))
#print(prec_set)


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

LineWidth = 1
MarkerSize = 8


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
    ax.plot(xy4[0], xy4[1], label=xy4[2],  marker='+', color=mymagenta, linewidth=LineWidth, markersize=MarkerSize)


print(prec_set)

dbl = data_dict["double"]
dbl_op_p_byte = [int(line[comp_idx]) / int(line[size_idx]) for line in dbl]
dbl_bw = [float(line[bw_idx]) for line in dbl]

flt = data_dict["float"]
flt_op_p_byte = [int(line[comp_idx]) / int(line[size_idx]) for line in flt]
flt_bw = [float(line[bw_idx]) for line in flt]

add = data_dict["Ac<3, d, d>"]
add_op_p_byte = [int(line[comp_idx]) / int(line[size_idx]) for line in add]
add_bw = [float(line[bw_idx]) for line in add]

adf = data_dict["Ac<3, d, f>"]
adf_op_p_byte = [int(line[comp_idx]) / int(line[size_idx]) for line in adf]
adf_bw = [float(line[bw_idx]) for line in adf]



fig, ax = create_fig_ax()
#ax.plot(dbl_op_p_byte, dbl_bw, label="double", marker='x', markersize=8, color=myorange)
plot_4(
    ax, (dbl_op_p_byte, dbl_bw, "double precision"),
    (flt_op_p_byte, flt_bw, "single precision"),
    (add_op_p_byte, add_bw, "Accessor<d, d>"),
    (adf_op_p_byte, adf_bw, "Accessor<d, s>"))
ax.axhline(1555.0, linestyle='--', marker='', linewidth=LineWidth, color=myblack, label="Peak double performance")

#ax.axhline(9746.0, linestyle='--', label="Peak double performance")
#ax.axhline(19490.0, linestyle='--', label="Peak single performance")

ax.set_xlabel("FLOP / Byte")
ax.set_ylabel("GB / s")
#ax.legend(loc="best")
ax.legend(loc="lower left")
#ax.loglog()
plot_figure(fig, "plots/test.pdf")

### Start second plot:
fig, ax = create_fig_ax()
ax.plot(dbl_op_p_byte, dbl_bw, label="double", marker='o', markersize=8)
ax.set_xlabel("FLOP / Byte")
ax.set_ylabel("GB / s")
ax.legend()

plot_figure(fig, "plots/test2.pdf")
