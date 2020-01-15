<#
.SYNOPSIS
    Script that runs various commands commands.

.DESCRIPTION
    What is run depends on the flags that are passed.

.PARAMETER cp_plots
    Set if you want to copy plots from Euler.
    You will need a vpn connection!
.PARAMETER act
    Set if you want to activate the python environment.
.PARAMETER docs
    Set if you want to build the documentation.

.EXAMPLE
    If you want to do everything, run:
    PS C:\> ./automate -cp_plots -act -docs

.NOTES
    Author: Christian Baumann
    Last Edit: 2019-01-15
    Version 1.0 - initial release
#>

# Parameter definition
param(
[switch]$cp_plots = $false,
[switch]$cp_data = $false,
[switch]$cp_hop = $false,
[switch]$act = $false,
[switch]$docs = $false)

# Copy plots from Euler
if ($cp_plots){
    scp -rp chbauman@euler.ethz.ch:MT/MasterThesis/Plots/ ./EulerPlots/
}
# Copy data to Euler
if ($cp_data){
    scp -rp ./Data/Datasets/ chbauman@euler.ethz.ch:MT/MasterThesis/Data/
}
# Copy hyperoptimization data from Euler
if ($cp_hop){
    scp -rp chbauman@euler.ethz.ch:MT/MasterThesis/Models/Hop/ ./Models/
}

# Activate python env
if ($act){
    Invoke-Expression "$($PSScriptRoot)/venv/Scripts/Activate.ps1"
}

# Build the docs
if ($docs){
    ./make_doc.ps1
}