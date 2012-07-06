Waffles
=======

A collection of machine learning and data mining algorithms and tools in C++. Developed by [Mike Gashler](mailto:mikegashler@gmail.com) and [Eric Moyer](mailto:eric_moyer@yahoo.com)
Builds on 32 and 64-bit platforms, including Linux, Windows, OSX, etc. Distributed under the LGPL license.

###Overview
--

Waffles provides the following command-line applications, each containing several tools for performing machine learning operations:

- **waffles_cluster** Tools for clustering.
- **waffles_dimred** Tools for dimensionality reduction, attribute selection, etc.
- **waffles_generate** Tools to sample distributions, sample manifolds, and generate certain types of data.
- **waffles_learn** Tools for supervised learning.
- **waffles_plot** Tools for visualizing data.
- **waffles_recommend** Tools for collaborative filtering, recommendation systems, imputation, etc.
- **waffles_sparse** Tools to learning from sparse data, document classification, etc.
- **waffles_transform** Tools for manipulating data, shuffling rows, swapping columns, matrix operations, etc.
- **waffles_audio** Tools for processing audio files.
- **waffles_wizard** For people who prefer not to have to remember commands, waffles also includes a graphical tool called

###Wizards and API
--

####waffles_wizard
This application opens a web browser and guides the user through a series of forms to create a command that will perform the desired task. This provides all the convenience of a GUI interface, but without locking the user in. That is, since it generates a command-line command for you to use, it becomes easy to perform the same operations without the GUI. Thus, the Waffles tools are particularly well-suited for experiments that need to be automated.
All of the functionality of the waffles tools is provided in an object-oriented C++ class library called

####GClasses
The API, in other words, the command-line tools are just thin wrappers around functionality implemented in the GClasses library. Thus, if you find any of the operations performed by the Waffles command-line tools to be useful, it is easy to incorporate the same functionality into your C++ programs.
Finally, in order to fully demonstrate how to use the GClasses library, several demo applications are also included with Waffles.