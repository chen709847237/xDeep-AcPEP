# xDeep-AcPEP
## Explainable Deep Learning Method for Anticancer Peptide Activity Prediction through Convolutional Neural Network and Multi-Task Learning
 
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

## Data
1. Models and data used for reproducing experiments are available at: [Here](https://drive.google.com/drive/folders/1DXHppIKO0vNqvpGrFAQyqnBodi3dr3fX?usp=sharing)
2. Final production models for peptide activity prediction are available at:TBC

## Run the script
1. Reproducing Experiments  
The script is located in ‘model_comparison_CV’
```bash
python reproduce.py -mo model_folder_path -da data_folder_path -o output_folder_path
```
Example:
```bash
python reproduce.py -mo ./model/ -da ./data/ -o ./
```

2. Final production model prediction  
TBC


 

