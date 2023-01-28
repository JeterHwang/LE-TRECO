# LE-TRECO

an Lstm-based Embedding Method for
the TREe COnstruction Step in Multiple Sequence
Alignment

## Guide Tree Construction Workflow
![LE-TRECO Architecture](https://i.imgur.com/B7LCmiG.png)

- ***Sequence Embedding***:
Embed protein sequences into vector representations by ML models
- ***Pre-Clustering***:
Divide the sequences into small clusters with bi-secting K-means++ and traditional K-means++
- ***Sub Guide Tree Constructions***:
Construct guide tree for each cluster with the Needleman-Wunsch and UPGMA algorithms 
- ***Merge Guide Trees***
Merge all guide trees into one guide tree with the UPGMA algorithm

## Performance on Large Protein Datasets
We replace the guide tree construction step of MAFFT, Clusta Omega, FAMSA, and T-Coffee with our algorithm and evaluate the MSA produced by these programs. The results show a significant boost on SP and TC scores if LE-TRECO is applied.

HomFam             |  extHomFam-v2
:-------------------------:|:-------------------------:
![](https://i.imgur.com/QnajkKm.png) | ![](https://i.imgur.com/YPKhxE5.png)


## Environment Setup
### Install Linux Packages
```
apt -y update
apt -y upgrade
apt -y install wget vim
apt -y install gawk bison
apt -y install ruby-full default-jdk
apt -y install golang-go
apt -y install r-base
apt -y install r-cran-rocr
apt -y install libboost-all-dev
apt -y install libblas-dev liblapack-dev
apt -y install gfortran
```
### Check/Install glibc-2.29
```
strings /lib/x86_64-linux-gnu/libm.so.6 | grep GLIBC_
```
If **GLIBC_2.29** does not appear on the screen, install it by the following commands
```
cd /usr/local
wget http://ftp.gnu.org/gnu/glibc/glibc-2.29.tar.gz
tar -zxvf glibc-2.29.tar.gz
cd glibc-2.29
mkdir build
cd build/
../configure --prefix=/usr/local --disable-sanity-checks
make -j18
make install
cd /lib/x86_64-linux-gnu
cp /usr/local/lib/libm-2.29.so /lib/x86_64-linux-gnu/
ln -sf libm-2.29.so libm.so.6
```
Then check glibc version by the same command.

### Compile MAFFT 7.490
The folder mafft-7.490-with-extensions is downloaded from the [MAFFT source page](https://mafft.cbrc.jp/alignment/software/source.html). To make MAFFT work with our code, please follow the commands below.
```
cd mafft-7.490-with-extensions/core
make clean
make
make install
cp ../scripts/mafft ../../src/
```

### Compile Clustal Omega 1.2.4
The folder clustal-omega-1.2.4 is downloaded from the [Clustal Omega homepage](http://www.clustal.org/omega/). To make Clustal Omega work with our code, please follow the commands below.
```
cd clustal-omega-1.2.4
./configure 
make clean
make

cp clustalo ../../src/
```

### Compile and Install FAMSA v2.0.1
The folder FAMSA-2.0.1 is the repository cloned from [FAMSA](https://github.com/refresh-bio/FAMSA), and it is of v2.0.1. To make FAMSA work with our code, please follow the commands below.
```
cd FAMSA-2.0.1
make clean
make
cp famsa /usr/local/bin/
```

### Install T-Coffee
Download the .bin installer from [T-coffee Beta release](http://tcoffee-packages.s3-website.eu-central-1.amazonaws.com/#Beta/Latest/), and install T-Coffee following the commands below.
```
wget https://s3.eu-central-1.amazonaws.com/tcoffee-packages/Beta/Latest/T-COFFEE_installer_Version_13.45.61.3c310a9_linux_x64.bin
chmod +x T-COFFEE_installer_Version_"version"_linux_x64.bin
./T-COFFEE_installer_Version_"version"_linux_x64.bin
```
Follow the wizard instructions and complete the installation. Be sure to open a new terminal session after the installation. 


### Install Python Packages
```
pip install -r requirements.txt
```

## Download Pretrained Models

### ESM-2 35M
```
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t12_35M_UR50D.pt
wget https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t12_35M_UR50D-contact-regression.pt
```
### ESM-2 150M
```
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt
wget https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t30_150M_UR50D-contact-regression.pt
```
### ESM-2 650M
```
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt
wget https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt
```
### 3-layer BiLSTM
Google drive link: [LSTM.pt](https://drive.google.com/file/d/1Y_bHIuRaJeqlM0V5ko29Pb6YGWHJegoq/view?usp=share_link)
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Y_bHIuRaJeqlM0V5ko29Pb6YGWHJegoq' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Y_bHIuRaJeqlM0V5ko29Pb6YGWHJegoq" -O LSTM.pt && rm -rf /tmp/cookies.txt
```
## Download Datasets
### HomFam
One can download the ContTest dataset from the Google drive link: [homfam.zip](https://drive.google.com/file/d/1oRAOd4rCM8Yur_SwIJivzftKv9F5s9ae/view?usp=share_link) 
or type the following command in the terminal.
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1oRAOd4rCM8Yur_SwIJivzftKv9F5s9ae' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1oRAOd4rCM8Yur_SwIJivzftKv9F5s9ae" -O homfam.zip && rm -rf /tmp/cookies.txt
unzip homfam.zip
rm homfam.zip
```
### extHomFam-v2
```

```
### ContTest
One can download the ContTest dataset from the Google drive link: [ContTest.zip](https://drive.google.com/file/d/1XwcPso7c8UwBVORjRuUqW9Qi6mhSgS5P/view?usp=sharing) 
or type the following command in the terminal.
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1XwcPso7c8UwBVORjRuUqW9Qi6mhSgS5P' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1XwcPso7c8UwBVORjRuUqW9Qi6mhSgS5P" -O ContTest.zip && rm -rf /tmp/cookies.txt
unzip ContTest.zip
rm ContTest.zip
```
## Usage
`python run_alignment.py [options]`
### Options
- Commonly used options
    * `--input <file/folder>`- Specify the inpout protein sequence file or the folder contains multiple files. Once a folder is given, all files ending with .tfa will be aligned.
    * `--msf_dir <path to a folder>`- Specify a folder to place final MSA sequence files (default: ./msf)
    * `--ref <file/folder>`- Specify the reference alignment fasta file or the folder contain multiple reference alignments. Notice that it should be the same type with input (i.e. if a file is given as input, only a file is allowed for this argument)
    * `--model_path <path to a .pt file>`- Specify the path to the pretrained model
    * `--embed_type <LSTM|esm-43M|esm35M|esm-150M|esm-650M>`- Specify which embedding model to use
    * `--align_prog <clustalo|mafft|famsa|tcoffee>`- Specify which program to use for alignment stage
    * `--no_tree`- If specified, will use the program given by --align_prog for the all process



- Other options
    * `--tree_dir <path to a folder>`- Specify a folder to place guide trees (default: ./trees)
    * `--fasta_dir <path to a folder>`- Specify a folder to place temporary fasta files (default: ./fasta)
    * `--log_dir <path to a folder>`- Specify a folder to place outputed logs (default: ./logs)
    * `--gpu <gpu index>`- Specify which gpu to work on
    * `--seed <integer>`- Specify the seed for Pytorch, numpy, and random
    * `--thread <integer>`- Specify how many threads the MSA program (e.g. famsa) should work on
    * `--max_cluster_size <integer>`- Specify the threshold for the Bi-secting K-means++ algorithm
    * `--batch_size <integer>`- Specify the batch size for inference, the recommended batch size for LSTM is 64, and 16 for transformers
    * `--num_workers <integer>`- Specify the thread for Pytorch dataLoader
    * `--newick2mafft_path <path to newick2mafft.rb>`- Specify the path to newick2mafft.rb, which is used to convert guide tree format for MAFFT
    * `--fastSP_path <path to FastSP.jar>`- Specify the path to FastSP.jar, which is used to evaluate the final alignment

## Run the code
Here we provide an example for homfam/small dataset, which uses 3-layer BiLSTM for embedding sequences and MAFFT for final alignment

``` 
python run_alignment.py --input ./data/homfam/small --ref ./data/homfam/small --model_path ./ckpt/LSTM.pt --embed_type LSTM --align_prog mafft 
```
change 3-layer BiLSTM to esm-650M transformer by the command below
```
python run_alignment.py --input ./data/homfam/small --ref ./data/homfam/small --model_path ./ckpt/esm2_t33_650M_UR50D.pt --embed_type esm-650M --align_prog mafft 
```
If pure mafft is desired (do not use LE-TRECO), one can type the command below
```
python run_alignment.py --input ./data/homfam/small --ref ./data/homfam/small --align_prog mafft --no_tree
```
## Citation

