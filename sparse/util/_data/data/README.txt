Data for the Deep Learning project
==================================

This directory contains the following subdirectories:

- train

	119685 jpg images for training data
	
	the filenames are shot<video number>_<shot number in video>_RKF.jpg
	
- test

	146788 jpg images for test data (evaluation of performance only)
	
- ann

	annotation files for 500 concepts
	
	each file contains:      <shot_name>   annotation
	where annotation is either N = negative,   P = positive,  S = skip (uncertain)
	
	not all shots have an annotation
	
For evaluation, compute the Average Precision over the first 2000 images.
(test images without annotation are assumed to be negative)

