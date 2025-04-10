# cncFinder

## Table of Contents
- [Dataset Descriptions](#dataset-descriptions)
  - [Test Dataset](#1-test-dataset)
  - [Model testing dataset](#2-model-testing-dataset)
  - [Training dataset](#3-training-dataset)
  - [Orgin training dataset](#4-orgin-training-dataset)
- [How-to-run-cncFinder?](#how-to-run-cncfinder)
- [How-to-train-cncFinder?](#how-to-train-cncfinder)
- [The-Architecture-of-cncFinder](#the-architecture-of-cncfinder)


### Dataset Descriptions

#### 1. Test Dataset
***File path***: `cncFinder/data/Test_dataset.xlsx`

This test datasets including all test datasets used in this study.

#### 2. Model testing dataset
***File path***: `cncFinder/data/test_neg13.fasta`

this test dataset specifically for testing the cncFinder.

#### 3. Training dataset
***File path***: `cncFinder/data/train.csv`

This dataset is a balanced dataset specifically for training the cncFinder.

#### 4. Orgin training dataset
***File path***: `cncFinder/data/train_neg.csv|train_pos.csv`

The orgin training datasets, neg is for negative, and pos is positive.

### How to run cncFinder
#### 1. Input File
Prepare your input file, which should be in fasta format. For example, the file name could be `input.fasta`.

#### 2. Output File
Define a name for your output file, which will be used to store the results processed by the script. For example, you might name your output file `output.txt`.

#### 3. Download this repository and open the Command Line Interface ####
   Open a terminal (Linux or MacOS) or Command Prompt/PowerShell (Windows).

#### 4. Run the Script ####
   Use the following command to run the script, replacing <input_file> and <output_file> with your actual file paths.
   ```bash
   python predict_user.py <input_file> <output_file>
   ```
   For example:
   ```bash
   python predict_user.py ./data/test_neg13.fasta output.txt
   ```
   
#### 5. Example Output ####
   The output file will contain the calculated score (the probility of dual functional lncRNA), and the identifier of each sequence. For example:
   ```python
   0.9992  seq1
   0.5238  seq2
   0.4213  seq3
    ...
   ```

### How to train cncFinder  
#### 1. Input File
    Prepare your training dataset input file, which should be in csv format with the title rna and label, 
    and replacing the `cncFinder/data/train.csv`.

#### 2. Params setting
    All params can be redefined in the file `cncFinder/util/config.py`

#### 3. Run the Script
    Use the following command to run the script.
   ```bash
   python train.py
   ```
   
#### 4. Model files
    The training files are under in checkpoint.
-------------------------------------------------
### The Architecture of cncFinder

<p style="text-align: justify;">
cncFinder, comprising four components: graph construction, node feature extraction, the GAT layer, and a classification module. First, the graph construction module transforms RNA sequences into graph structures, wherein each node corresponds to a k-mer and edges are established based on the sequential adjacency of these k-mers. Subsequently, the node feature extraction module calculates Word2Vec feature representations for each node. Thereafter, the GAT layer employs an attention mechanism to allocate differential weights to neighboring nodes, thereby enhancing the extraction of RNA structural and functional features. Finally, the classification module employs a fully connected layer for classification, thereby facilitating high-precision prediction of bifunctional RNAs.
</p>
<div align="center"> 
<img width="60%" alt="image" src="https://github.com/user-attachments/assets/6052b646-6967-4437-ac12-509d7c740c79">
</div>
