# MasterThesis

This is the repository of my Master thesis. It contains all the codes, files etc.
that were generated for the thesis. 

## [Overleaf @ ...](https://github.com/chbauman/Master-ThesisOverLeaf)

This folder contains the overleaf repository
with all the latex code.

## [BatchRL](BatchRL)

This folder contains the python code
for batch reinforcement learning. The code was tested using Python version 3.6,
it is not compatible with version 3.5 or below. It was tested using PyCharm and
Visual Studio, part of the code was also run on Euler.

## [Data](Data)

Here the data is put when the code is 
executed. Should be empty on the git repo
except for the [README.md](Data/README.md).

## Documentation

The code was written in such a way that it
allows for generating a documentation automatically with Sphinx.
To generate it first install Sphinx and then use:

```console
$ cd ..
$ mkdir Docs
$ cd Docs
$ sphinx-apidoc -F -H 'BatchRL' -A 'Chris' -o . '../MasterThesis/BatchRL/'
$ cp ../MasterThesis/conf.py .
$ ./make html
```

Or you can directly use the Windows Powershell script [make_doc.ps1](make_doc.ps1)
which basically runs these commands. The script also removes previously existing
folders 'Docs', so it can be used to update the documentation.
