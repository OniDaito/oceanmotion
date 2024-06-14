#!/usr/bin/env python
r"""
     ___                   _  _      _  o          
     )) ) __  __ ___  _ _  )\/,) __  )L _  __  _ _ 
    ((_( ((_ (('((_( ((\( ((`(( ((_)(( (( ((_)((\( 

An attempt to speedily visualise our fast_eval data
detections from our model. We use bokeh and bokeh serve
to run an interactive visualisation

To run, try this:

bokeh serve --show visualise.py --args 
  -s ~/tmp/fast_eval/20230401-000000_20230402-000000.sqlite3
  -u sealhits_ro -w 9T0J^MpjMrE! -n juve.st-andrews.ac.uk

"""

from __future__ import annotations

__all__ = ["main"]
__version__ = "0.9.0"
__author__ = "Benjamin Blundell <bjb8@st-andrews.ac.uk>"

import argparse
import numpy as np
import pytz
import sqlite3
from datetime import datetime, timedelta
from sealhits.bbox import XYBox
from sealhits.db.db import DB
from sealhits.db.dbschema import Groups

from bokeh.models import TextInput
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    RangeTool,
)
from bokeh.plotting import figure, curdoc

# Naughty globals
RANGE_START = None
RANGE_END = None
TEXT_BOX_RANGE = None
TEXT_BOX_COUNT = None
HEATMAP = None
BOX_DATA = []
SQL_DATA = {}

def redraw_heatmap():
    global RANGE_START
    global RANGE_END
    global BOX_DATA
    global SQL_DATA
    global HEATMAP
    global TEXT_BOX_RANGE
    global TEXT_BOX_COUNT
    print("Redrawing Heatmap...")

    # Redraw our heatmap - look at our box_data
    xyvalues = np.zeros((867, 256))
    count = 0

    for dd, boxes in SQL_DATA.items():
        if dd >= RANGE_START and dd < RANGE_END:
            for box in boxes:
                xyvalues[box.y_min:box.y_max, box.x_min:box.x_max] += 1
                count +=1

    # must give a vector of image data for image parameter
    HEATMAP.image(image=[xyvalues], x=0, y=0, dw=256, dh=867, palette="Viridis256")
    print("Updated Heatmap.")

    TEXT_BOX_RANGE.value = str((RANGE_END - RANGE_START))
    TEXT_BOX_COUNT.value = str(count)
    

def callback_start(attr, old, new):
    global RANGE_START
    RANGE_START = datetime.fromtimestamp(int(new) / 1000).astimezone(pytz.UTC)
    redraw_heatmap()

def callback_end(attr, old, new):
    global RANGE_END
    RANGE_END = datetime.fromtimestamp(int(new) / 1000).astimezone(pytz.UTC)
    redraw_heatmap()

def main(args):
    global HEATMAP
    global RANGE_START
    global RANGE_END
    global BOX_DATA
    global SQL_DATA
    global TEXT_BOX_RANGE
    global TEXT_BOX_COUNT

    # Load the SQLITE3
    con = sqlite3.connect(args.sqlite)
    cur = con.cursor()
    SQL_DATA = {}
    BOX_DATA = []

    # Populate initially for the graph. All zeros at 100ms intervals
    trez = timedelta(seconds=0.25)
    res = cur.execute("SELECT * FROM detections order by datetime asc")
    rows = res.fetchall()
    start_date = datetime.strptime(rows[0][0], "%Y-%m-%d %H:%M:%S.%f%z")
    end_date = datetime.strptime(rows[-1][0], "%Y-%m-%d %H:%M:%S.%f%z")

    for ridx, ra in enumerate(rows[:-1]):
        rb = rows[ridx + 1]
        dts, xmin, ymin, xmax, ymax, _ = ra
        dtbs, xminb, yminb, xmaxb, ymaxb, _ = rb

        for df in ["%Y-%m-%d %H:%M:%S.%f%z", "%Y-%m-%d %H:%M:%S%z"]:
            try:
                dt = datetime.strptime(dts, df)
                dtb = datetime.strptime(dtbs, df)
            except ValueError:
                pass

        dd = dtb - dt

        # Fill in the missing time periods with blanks
        if dd > trez:
            cb = dt + trez
            while cb < dtb:
                SQL_DATA[cb] = []
                cb += trez

        if dt not in SQL_DATA.keys():
            SQL_DATA[dt] = []

        bbox = XYBox(xmin, ymin, xmax, ymax)
        BOX_DATA.append(bbox)
        SQL_DATA[dt].append(bbox)

    sorted_items = sorted(SQL_DATA.items())
    dates_sql = sorted(list(SQL_DATA.keys()))
    areas_sql = []

    for a, b in sorted_items:
        area = 0

        for bb in b:
            area += bb.area()

        areas_sql.append(area)

    #counts_sql = [len(b) for a, b in sorted_items]

    # counts_ds = ColumnDataSource(data=dict(date=dates_sql, close=counts_sql))
    areas_ds = ColumnDataSource(data=dict(date=dates_sql, close=areas_sql))

    # Add areas for the original annotations
    seal_db = DB(
        db_name=args.dbname,
        username=args.dbuser,
        password=args.dbpass,
        host=args.dbhost,
    )

    groups = seal_db.get_groups_filters(
        [
            Groups.timestart >= start_date,
            Groups.timeend < end_date,
            Groups.code != "seal",
        ]
    )

    anno_x = []

    for group in groups:
        anno_x.append(
            (group.timestart.astimezone(pytz.UTC), group.timeend.astimezone(pytz.UTC))
        )

    groups_seals = seal_db.get_groups_filters(
        [
            Groups.timestart >= start_date,
            Groups.timeend < end_date,
            Groups.code == "seal",
        ]
    )

    anno_seals = []

    for group in groups_seals:
        anno_seals.append(
            (group.timestart.astimezone(pytz.UTC), group.timeend.astimezone(pytz.UTC))
        )

    fig_0 = figure(
        height=400,
        width=1200,
        tools="xpan",
        toolbar_location=None,
        x_axis_type="datetime",
        x_axis_location="above",
        background_fill_color="#efefef",
        x_range=(dates_sql[1500], dates_sql[5500]),
    )

    for anno in anno_x:
        fig_0.quad(
            top=100,
            bottom=0,
            left=anno[0],
            right=anno[1],
            fill_color="green",
            line_color="white",
            alpha=0.3,
        )

    for anno in anno_seals:
        fig_0.quad(
            top=100,
            bottom=0,
            left=anno[0],
            right=anno[1],
            fill_color="red",
            line_color="white",
            alpha=0.3,
        )

    # p.line('date', 'close', source=counts_ds)
    fig_0.line("date", "close", source=areas_ds, color="blue")
    fig_0.yaxis.axis_label = "Areas (blue)"

    select = figure(
        title="Drag the middle and edges of the selection box to change the range above",
        height=160,
        width=1200,
        y_range=fig_0.y_range,
        x_axis_type="datetime",
        y_axis_type=None,
        tools="",
        toolbar_location=None,
        background_fill_color="#efefef",
    )

    range_tool = RangeTool(x_range=fig_0.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2

    select.line("date", "close", source=areas_ds)
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)

    # Add callbacks on the range tool to change our image
    range_tool.x_range.on_change('start', callback_start)
    range_tool.x_range.on_change('end', callback_end)
    RANGE_START = start_date
    RANGE_END = end_date
    TEXT_BOX_RANGE = TextInput(value="", title="Current Date Range:")
    TEXT_BOX_COUNT = TextInput(value="0", title="Number of AI Detections:")
    HEATMAP = figure(x_range=(0, 256), y_range=(0, 867), height=867, width=256)
    redraw_heatmap()
    curdoc().add_root(row(column(fig_0, select, TEXT_BOX_RANGE, TEXT_BOX_COUNT), HEATMAP))
    print("Finished Load")

# Would normally have if __name__ etc guard here but since we are serving with Bokeh, I've removed it.

parser = argparse.ArgumentParser(
    prog="oceanmotion - run",
    description="Run a model over an NPZ file and get a prediction back.",
    epilog="SMRU St Andrews",
)

parser.add_argument(
    "-s", "--sqlite", default="", help="The path to the sqlite3 file."
)
parser.add_argument("-o", "--outpath", default=".", help="The path for the output.")

parser.add_argument(
    "-d",
    "--dbname",
    default="sealhits",
    help="The name of the postgresql database (default: sealhits)",
)
parser.add_argument(
    "-u",
    "--dbuser",
    default="sealhits",
    help="The username for the postgresql database (default: sealhits)",
)
parser.add_argument(
    "-w",
    "--dbpass",
    default="kissfromarose",
    help="The password for the postgresql database (default: kissfromarose)",
)
parser.add_argument(
    "-n",
    "--dbhost",
    default="localhost",
    help="The hostname for the postgresql database (default: localhost)",
)

args = parser.parse_args()
main(args)
