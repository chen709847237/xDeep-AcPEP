# xDeep-AcPEP
## Deep Learning Method for Anticancer Peptide Activity Prediction through Convolutional Neural Network and Multi-Task Learning
## :fire:UPDATE(2021.7.12): The online server for xDeep-AcPEP is now available: [Here](https://app.cbbio.online/acpep/home)
In this repository:
1. We provide python scripts for reproducing the experiments of 5-folds cross-validation model comparison.
2. We provide our final production models for peptide activity prediction.

## Requirements 
* Anaconda 4.7.0
* Python 3.6.9
* Scikit-learn 0.21.3
* Pytorch 1.2.0 with CUDA 10.0
* Scipy 1.4.1
* Pandas 1.0.2 
* Numpy 1.18.1
* Openpyxl 3.0.6

## Model and Data
1. Models and data used for reproducing experiments are available at: [Here](https://drive.google.com/drive/folders/1DXHppIKO0vNqvpGrFAQyqnBodi3dr3fX?usp=sharing)
2. Final production models for peptide activity prediction are available at: ```./prediction/model/```

## Run the script
### 1. Reproducing Experiments  
The script is located in ```model_comparison_CV``` folder
```bash
python reproduce.py -mo <model_folder_path> -da <data_folder_path> -o <output_folder_path>
```
#### Example:
```bash
python reproduce.py -mo ./model/ -da ./data/ -o ./
```

### 2. Final production model prediction ([Online Server](https://app.cbbio.online/acpep/home))
The script is located in ```prediction``` folder
```bash
python prediction.py -t <tissue_type> -m <model_folder_path> -d <fasta_file_path> -o <output_folder_path>
```
where:  
```<tissue_type>``` could be selected from ```breast```, ```cervix```, ```colon```, ```lung```, ```prostate``` and ```skin```.   

#### Example:
```bash
python prediction.py -t breast -m ./model/ -d ./test_breast.fasta -o ./result/
```
**Note: The input peptide data must in the form of the following FASTA format.**
```bash
>AmphiArc1
KWVKKVHNWLRRWIKVFEALFG
>AmphiArc2
KIFKKFKTIIKKVWRIFGRF
>Gradient2
AWLKRIKKFLKALFWVWVW 
>AmphiArc3
AFRHSVKEELNYIRRRLERFPNRL
```
## References
1. We used ```iFeature``` to extract all peptide features. ([Github](https://github.com/Superzchen/iFeature/), [Paper](https://academic.oup.com/bioinformatics/article-abstract/34/14/2499/4924718))

