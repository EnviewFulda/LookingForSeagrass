## Overview
In this project we examined the automatic seagrass coverage estimation of the sea bottom.
A total of **12682** images of the seabed at different depths along Croatia's Adriatic coast were taken with the help of a diving robot.
Of these, **6036** images were manually polygon-annotated by hand and made available to the public as pixel maps.
Using this dataset, we tested a superpixel classification of seagrass images. To achieve this, we used several different feature extraction methods as for example CNN-Features that turned out to be the best ones in our experiments.

This project is a joint work between *University of Zadar - Croatia* and *University of Applied Sciences Fulda - Germany*
### Envolved People
* Gereon Reus, Fulda
* Thomas Möller, Fulda
* Jonas Jäger, Fulda
* Julian Hasenauer, Fulda
* Dr. Stewart T. Schultz, Zadar
* Dr. Claudia Kruschel, Zadar
* Dr. Viviane Wolff,  Fulda
* Dr. Klaus Fricke-Neuderth, Fulda


### Paper

The full paper **Looking for Seagrass: Deep Learning for Visual Coverage Estimation (accepted for publication@IEEE - OCEANS 2018 Kobe)** is available at: [HS Fulda](https://www.hs-fulda.de/fileadmin/user_upload/FB_ET/Projekte_Forschung/Enview_Jaeger/EnView_News_2018-04/Conference_Kobe_2018_Seagrass.pdf)

### Dataset

[looking-for-seagrass-dataset](https://drive.google.com/open?id=1X0pmRIkPRC672_vuWqotfLdgbHx1QpFZ)

### Deep Net for feature extraction
http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz

## Experiments
### Getting started
#### Dependencies
* Anaconda 3
* CUDA 8 & cuDNN 6
* VirtualEnv (Dependencies located in conda-env.yml)

#### Checklist
1. Install dependencies 
2. Create VirtualEnv
3. Download InceptionNet V3 Graph 
4. Download LookingForSeagrass Dataset (and extract)
5. Adapt three paths in experiment scripts
6. Activate Virtual Env
7. Run experiments 

#### Paths in experiments
```
#Path to InceptionNetV3 ProtoBuf
GRAPH="/path/to/classify_image_graph_def.pb"

# Root Path of LookingForSeagrass Dataset
FOLDER_ROOT="/path/to/datasetroot/dataset"

# Path for storing your results
OUTPUT_PATH="/path/to/output/results"
```

### Citation
Please cite our accepted Paper:
```
@unpublished{Reus2018,
title = {Looking for Seagrass: Deep Learning for Visual Coverage Estimation},
author = {Gereon Reus and Thomas M{\"o}ller and Jonas J{\"a}ger and Stewart T. Schultz and Claudia Kruschel and Julian Hasenauer and Viviane Wolff and Klaus Fricke-Neuderth},
booktitle = {MTS/IEEE OCEANS Conference Kobe},
year = {2018},
note = {accepted for publication}
}
```


