# -*- coding: utf-8 -*-

# /**
# * Copyright (c) 2009-2018.
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
__contributors__    = ["Marée Raphael <raphael.maree@uliege.be>"]
__copyright__       = "Copyright 2010-2018 University of Liège, Belgium, http://www.cytomine.org/"


import os
from cytomine.models import *
import numpy as np

from cytomine import CytomineJob
from sklearn.externals import joblib

from pyxit import build_models



def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")


def stringify(l):
    return ",".join(map(str, l))


def parse_domain_list(s):
    if s is None or len(s) == 0:
        return []
    return list(map(int, s.split(',')))


def get_annotations(id_project, images=None, terms=None, users=None, reviewed=False, **kwargs):
    annotations = AnnotationCollection(
        project=id_project, images=images, term=terms,
        users=users, showTerm=True, **kwargs
    ).fetch()

    if reviewed:
        annotations += AnnotationCollection(
            project=id_project, images=images, term=terms,
            users=users, reviewed=True, showTerm=True, **kwargs
        ).fetch()

    return annotations


def main(argv):
    with CytomineJob.from_cli(argv) as cj:
        # annotation filtering
        cj.logger.info(str(cj.parameters))

        cj.job.update(progress=1, statuscomment="Preparing execution (creating folders,...).")
        base_path = "/data"
        images_path = os.path.join(
            base_path, "images", "train",
            "zoom_level", str(cj.parameters.cytomine_zoom_level),
            "alpha", str(int(cj.parameters.cytomine_download_alpha))
        )
        ls_path = os.path.join(images_path, "train")

        if os.path.exists(ls_path):
            import shutil
            shutil.rmtree(ls_path)
        os.makedirs(ls_path)

        cj.job.update(progress=2, statusComment="Fetching annotations...")
        filters = {
            "terms": parse_domain_list(cj.parameters.cytomine_id_terms),
            "images": parse_domain_list(cj.parameters.cytomine_id_images),
            "users": parse_domain_list(cj.parameters.cytomine_id_users)
        }
        projects = parse_domain_list(cj.parameters.cytomine_id_projects)
        if projects is None or len(projects) == 0:  # if projects is missing, fetch only from current project
            projects = [cj.project.id]

        annotations = AnnotationCollection()

        for id_project in cj.monitor(projects, start=2, end=20, period=0.1, prefix="Download annotations from project"):
            annotations += get_annotations(id_project, **filters, reviewed=cj.parameters.cytomine_reviewed, showTerms=True)
            cj.logger.info("Downloaded {} annotation(s) so far...".format(len(annotations)))
        cj.logger.info("Downloaded a total of {} annotation(s)...".format(len(annotations)))

        cj.job.update(progress=2, statusComment="Fetching images in {}...".format(images_path))
        filenames = list()
        terms = list()
        for annotation in cj.monitor(annotations, start=20, end=50, period=200, prefix="Download crops of annotations"):
            for term in annotation.term:
                dump_params = {
                    "dest_pattern": os.path.join(ls_path, str(term), "{image}_{id}.png"),
                    "override": True,
                    "alpha": cj.parameters.cytomine_download_alpha,
                    "zoom": cj.parameters.cytomine_zoom_level
                }
                if annotation.dump(**dump_params):
                    terms.append(term)
                    filenames.append(os.path.join(
                        ls_path,
                        dump_params["dest_pattern"].format(image=annotation.image, id=annotation.id)
                    ))
                else:
                    cj.logger.error("Failed to download crop for annotation {} (term={})".format(annotation.id, term))

        cj.logger.info("Downloaded a total of {} crops(s)...".format(len(filenames)))
        x = np.array(filenames)
        y = np.array(terms)

        # transform classes
        cj.job.update(progress=50, statusComment="Transform classes...")
        classes = np.unique(y)
        n_classes = len(classes)
        positive_classes = parse_domain_list(cj.parameters.cytomine_positive_terms)

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
        Property(cj.job, key="positive_classes", value=stringify(positive_classes)).save()

        cj.job.update(status=Job.TERMINATED, status_comment="Finish", progress=100)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
