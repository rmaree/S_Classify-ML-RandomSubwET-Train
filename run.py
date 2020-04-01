# -*- coding: utf-8 -*-

# /**
# * Copyright (c) 2009-2020.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */


__author__          = "Mormont Romain <r.mormont@uliege.be>"
__copyright__       = "Copyright 2010-2020 University of Liège, Belgium, http://www.cytomine.org/"


import os
import numpy as np
from pathlib import Path
from sklearn.externals import joblib
from cytomine.models import *
from cytomine import CytomineJob
from cytomine.utilities.software import setup_classify, parse_domain_list, stringify
from pyxit import build_models


def main(argv):
    with CytomineJob.from_cli(argv) as cj:
        # annotation filtering
        cj.logger.info(str(cj.parameters))

        # use only images from the current project
        cj.parameters.cytomine_id_projects = "{}".format(cj.parameters.cytomine_id_project)

        cj.job.update(progress=1, statuscomment="Preparing execution (creating folders,...).")
        base_path, downloaded = setup_classify(
            args=cj.parameters, logger=cj.job_logger(1, 40),
            dest_pattern=os.path.join("{term}", "{image}_{id}.png"),
            root_path=Path.home(), set_folder="train", showTerm=True
        )

        x = np.array([f for annotation in downloaded for f in annotation.filenames])
        y = np.array([int(os.path.basename(os.path.dirname(filepath))) for filepath in x])

        # transform classes
        cj.job.update(progress=50, statusComment="Transform classes...")
        classes = parse_domain_list(cj.parameters.cytomine_id_terms)
        positive_classes = parse_domain_list(cj.parameters.cytomine_positive_terms)
        classes = np.array(classes) if len(classes) > 0 else np.unique(y)
        n_classes = classes.shape[0]

        # filter unwanted terms
        cj.logger.info("Size before filtering:")
        cj.logger.info(" - x: {}".format(x.shape))
        cj.logger.info(" - y: {}".format(y.shape))
        keep = np.in1d(y, classes)
        x, y = x[keep], y[keep]
        cj.logger.info("Size after filtering:")
        cj.logger.info(" - x: {}".format(x.shape))
        cj.logger.info(" - y: {}".format(y.shape))

        if cj.parameters.cytomine_binary:
            cj.logger.info("Will be training on 2 classes ({} classes before binarization).".format(n_classes))
            y = np.in1d(y, positive_classes).astype(np.int)
        else:
            cj.logger.info("Will be training on {} classes.".format(n_classes))
            y = np.searchsorted(classes, y)

        # build model
        cj.job.update(progress=55, statusComment="Build model...")
        _, pyxit = build_models(
            n_subwindows=cj.parameters.pyxit_n_subwindows,
            min_size=cj.parameters.pyxit_min_size,
            max_size=cj.parameters.pyxit_max_size,
            target_width=cj.parameters.pyxit_target_width,
            target_height=cj.parameters.pyxit_target_height,
            interpolation=cj.parameters.pyxit_interpolation,
            transpose=cj.parameters.pyxit_transpose,
            colorspace=cj.parameters.pyxit_colorspace,
            fixed_size=cj.parameters.pyxit_fixed_size,
            verbose=int(cj.logger.level == 10),
            create_svm=cj.parameters.svm,
            C=cj.parameters.svm_c,
            random_state=cj.parameters.seed,
            n_estimators=cj.parameters.forest_n_estimators,
            min_samples_split=cj.parameters.forest_min_samples_split,
            max_features=cj.parameters.forest_max_features,
            n_jobs=cj.parameters.n_jobs
        )
        cj.job.update(progress=60, statusComment="Train model...")
        pyxit.fit(x, y)

        cj.job.update(progress=90, statusComment="Save model....")
        model_filename = joblib.dump(pyxit, os.path.join(base_path, "model.joblib"), compress=3)[0]

        AttachedFile(
            cj.job,
            domainIdent=cj.job.id,
            filename=model_filename,
            domainClassName="be.cytomine.processing.Job"
        ).upload()

        Property(cj.job, key="classes", value=stringify(classes)).save()
        Property(cj.job, key="binary", value=cj.parameters.cytomine_binary).save()
        Property(cj.job, key="positive_classes", value=stringify(positive_classes)).save()

        cj.job.update(status=Job.TERMINATED, status_comment="Finish", progress=100)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
