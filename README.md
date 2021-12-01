# MOTIF Dataset

The Malware Open-source Threat Intelligence Family (MOTIF) dataset contains 3,095 disarmed PE malware samples from 454 families, labeled with ground truth confidence. Family labels were obtained by surveying thousands of open-source threat reports published by 14 major cybersecurity organizations between Jan. 1st, 2016 Jan. 1st, 2021. The dataset also provides a comprehensive alias mapping for each family and EMBER raw features for each file.

Further information about the MOTIF dataset is provided in our [paper](https://arxiv.org/abs/2111.15031).

If you use the provided data or code, please make sure to cite our paper:

```
@misc{joyce2021motif,
      title={MOTIF: A Large Malware Reference Dataset with Ground Truth Family Labels},
      author={Robert J. Joyce and Dev Amlani and Charles Nicholas and Edward Raff},
      year={2021},
      eprint={2111.15031},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```

## Downloading the Dataset

Due to the size of the dataset, you must use Git LFS in order to clone the repository. Installation instructions for Git LFS are linked [here](https://github.com/git-lfs/git-lfs). On Debian-based systems, the Git LFS package can be installed using:

```sudo apt-get install git-lfs```

Once Git LFS is installed, you can clone this repository using:

```
git lfs clone https://github.com/boozallen/MOTIF.git
```

## Dataset Contents

The main dataset is located in ```dataset/``` and contains the following files:

### motif_dataset.jsonl
Each line of motif_dataset.jsonl is a .json object with the following entries:

| Name | Description |
|---|---------|
| md5 | MD5 hash of malware sample |
| sha1| SHA-1 hash of malware sample |
| sha256| SHA-256 hash of malware sample |
| reported_hash | Hash of malware sample provided in report |
| reported_family | Normalized family name provided in report |
| aliases | List of known aliases for family |
| label | Unique id for malware family (for ML purposes) |
| report_source | Name of organization that published report |
| report_date | Date report was published |
| report_url | URL of report |
| report_ioc_url | URL to report appendix (if any) |
| appeared | Year and month malware sample was first seen |

Each .json object also contains EMBER raw features (version 2) for the file:
| Name | Description |
|---|---------|
| histogram | EMBER histogram |
| byteentropy | EMBER byte histogram |
| strings | EMBER strings metadata |
| general | EMBER general file metadata |
| header | EMBER PE header metadata |
| section | EMBER PE section metadata |
| imports | EMBER imports metadata |
| exports | EMBER exports metadata |
| datadirectories | EMBER data directories metadata |

### motif_families.csv

This file contains an alias mapping for each of the 454 malware families in the MOTIF dataset. It also contains a succinct description of the family and the threat group or campaign that the family is attributed to (if any).

| Column | Description |
|---|---------|
| Aliases | List of known aliases for family |
| Description | Brief sentence describing capabilities of malware family |
| Attribution (If any) | Name of threat actor malware/campaign is attributed to |


### motif_reports.csv
This file provides information gathered from our original survey of open-source threat reports. We identified 4,369 malware hashes with 595 distinct reported family names during the survey, but we were unable to obtain some of the files and we restricted the MOTIF dataset to only files in the PE file format. The reported hash, family, source, date, URL, and IOC URL of any malware samples which did not make it into the final MOTIF dataset are located here.

### MOTIF.7z
The disarmed malware samples are provided in this 1.47GB encrypted .7z file, which can be unzipped using the following password:

```i_assume_all_risk_opening_malware```

Each file is named in the format MOTIF_MD5, with MD5 indicating the file's hash prior to when it was disarmed.

### X_train.dat and y_train.dat
EMBERv2 feature vectors and labels are provided in X_train.dat and y_train.dat, respectively. Feature vectors were computed using LIEF v0.9.0. These files are named for compatibility with the EMBER read_vectorized_features() function. MOTIF is not split into a training or test set, and X_train.dat and y_train.dat contain feature vectors and labels for the entire dataset.


## Benchmark Models
We provide code for training the ML models described in our paper, located in ```benchmarks/```. To support these models, code for modified versions of [MalConv2](https://github.com/NeuromorphicComputationResearchProgram/MalConv2) is included in the ```MalConv2/``` directory.

### Requirements:
Packages required for training the ML models can be installed using the following commands:

```
pip3 install -r requirements.txt
python3 setup.py install
```

Training the LightGBM or outlier detection models also requires EMBER:

```
pip3 install git+https://github.com/elastic/ember.git
```

### Training the models:
The LightGBM model can be trained using the following command, where ```/path/to/MOTIF/dataset/``` indicates the path to the ```dataset/``` directory. 

```python3 lgbm.py /path/to/MOTIF/dataset/```

The MalConv2 model can be trained using the following command, where ```/path/to/MOTIF/MOTIF_defanged/``` indicates the path to the unzipped folder containing the disarmed malware samples:

```python3 malconv.py /path/to/MOTIF/MOTIF_defanged/ /path/to/MOTIF/dataset/motif_dataset.jsonl```

The three outlier detection models can be trained using the following command:

```python3 outliers.py /path/to/MOTIF/dataset/```

## Proper Use of Data

Use of this dataset must follow the provided terms of licensing. We intend this dataset to be used for research purposes and have taken measures to prevent abuse by attackers. All files are prevented from running using the same technique as the SOREL dataset. We refer to [their statement](https://ai.sophos.com/2020/12/14/sophos-reversinglabs-sorel-20-million-sample-malware-dataset/) regarding safety and abuse of the data.

> The malware we’re releasing is “disarmed” so that it will not execute. This means it would take knowledge, skill, and time to reconstitute the samples and get them to actually run. That said, we recognize that there is at least some possibility that a skilled attacker could learn techniques from these samples or use samples from the dataset to assemble attack tools to use as part of their malicious activities. However, in reality, there are already many other sources attackers could leverage to gain access to malware information and samples that are easier, faster and more cost effective to use. In other words, this disarmed sample set will have much more value to researchers looking to improve and develop their independent defenses than it will have to attackers.
