# S_Classify-ML-RandomSubwET-Train
A Cytomine software for classifying crops with Random subwindows and ExtraTrees. 

All images and related data are downloaded to the `/data` folder of the container.

The final model object is serialized and compressed using `joblib.dump` with compression level of 3 and attached to the job.
The compressed model is either a `PyxitClassifier` (`--svm` set to `false`) or a `SvmPyxitClassifier` (`--svm` set to  `true`).

Also attached as job properties:
- `classes`: the sorted ids of the terms that were used to train the classifier. Indexes of this table are the actual classes used by the classifier.
- `positive_classes`: if the problem was binarized, the ids of the terms used as positive

# See also

[Pyxit implementation](https://github.com/Cytomine-ULiege/pyxit)

# Thanks to
This software is a contribution from the [ULiège Cytomine research team](https://uliege.cytomine.org/)
