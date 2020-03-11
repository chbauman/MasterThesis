========
Cookbook
========

This page explains how to use the main script, `BatchRL.py`.
You will need to `cd` into the folder `BatchRL` and activate
the virtual environment before running these commands.

Verbose mode
------------

One option that can be used in all cases, 
is :option:`-v`:
	
   python BatchRL.py -v [other_options]

In this case the output will usually be more
verbose than without that option.

Retrieve data from the NEST database
------------------------------------

If you want to loat the data from the database to your
local PC, 
simply run::

    python BatchRL.py -d --data_end_date 2020-02-21

This will retrieve, process, and store the data from 
beginning of 2019 until the specified date with the option
:option:`--data_end_date`. There is no need to specify a room
number, this will load the data for all room. Also includes
the data of the battery.

Battery
-------

Running the script with the option :option:`-b` for
battery::

    python BatchRL.py -b --data_end_date 2020-02-21

Will fit and evaluate the battery model, based on the
data that was collected up to the specified date.

