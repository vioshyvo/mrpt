#MATLAB implementation of the MRPT algorithm

##Get started

###Java
The MATLAB implementation is dependent on some simple Java structures, thus
before using the method some simple procedures need to be undertaken.

First, you need to compile the Java class `java/RPTNode.java` by typing in 
a terminal window

    javac -source <version> -target <version> java/RPTNode.java,

where \<version\> is the java version supported by your MATLAB installation 
(1.7 for MATLAB R2014b)

Next, you need to include this class to your Java path. This happens by 
typing

    javaaddpath ./java

to the MATLAB command window while working in the `mrpt/matlab` directory.

###Usage
Now you are all set to use MRPT in MATLAB. In the simplest scenario the 
index is built with command

    index = mrpt(data);

Simple as that! Now you can find the approximate `k` nearest neighbors of 
object `obj` by 

    neighbors = ann(data, index, obj);

### Optional
Advanced users might want to specify the maximum leaf size and number of 
trees for the mrpt index. These are optional parameters for the mrpt 
function:

    index = mrpt(data, <leaf size>, <number of trees>);

## Extras
The directory `mnist_utils` contains functions for loading and visualizing
the popular mnist handwritten digit dataset.