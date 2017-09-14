# bloom-filter-iris-indexing
Reference implementation for Bloom filter-based iris indexing proposed in [1] and extended in [2,3].

## License
This work is licensed under license agreement provided by Hochschule Darmstadt ([h_da-License](/hda-license.pdf)).

## Instructions
### Dependencies
* [Python3.5+](https://www.python.org/ "Python")
* [NumPy](http://www.numpy.org "NumPy")

### Data format and experiments
* See [example_binary_template.txt](/example_binary_template.txt) file for Iris-Code data format.
* Currently, the number of enrolled subjects and block width must be powers of 2.
* Implement the split_dataset and extract_source_data methods as needed for your experiment and data filenames.
* Add result processing code as needed.

### Usage
bloomfilter.py [-h] [-v] -d DIRECTORY -n ENROLLED -bh HEIGHT -bw WIDTH -T CONSTRUCTED -t TRAVERSED

*required named arguments:*
* **-d** DIRECTORY, **--directory** DIRECTORY : directory where the binary templates are stored
* **-n** ENROLLED, **--enrolled** ENROLLED : number of enrolled subjects
* **-bh** HEIGHT, **--height** HEIGHT : filter block height
* **-bw** WIDTH, **--width** WIDTH : fitler block width
* **-T** CONSTRUCTED, **--constructed** CONSTRUCTED : number of trees constructed
* **-t** TRAVERSED, **--traversed** TRAVERSED : number of trees traversed

*optional arguments:*
* **-h**, **--help** : show this help message and exit
* **-v**, **--version** : show program's version number and exit

## References
* [1] Christian Rathgeb, Frank Breitinger, Harald Baier, Christoph Busch, "Towards Bloom Filter-based Indexing of Iris Biometric Data", In Proceedings of the 8th IAPR International Conference on Biometrics (ICB'15), 2015.
* [2] Pawel Drozdowski, Christian Rathgeb, Christoph Busch, "Bloom Filter-based Search Structures for Indexing and Retrieving Iris-Codes", in IET Biometrics, 2017.
* [3] Pawel Drozdowski, Christian Rathgeb, Christoph Busch, "Multi-Iris Indexing and Retrieval: Fusion Strategies for Bloom Filter-based Search Structures", in Proc. International Joint Conference on Biometrics (IJCB), Denver, USA, October 2017.

© [Hochschule Darmstadt](https://www.h-da.de/ "Hochschule Darmstadt website")
