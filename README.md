# LE-TRECO

an Lstm-based Embedding Method for
the TREe COnstruction Step in Multiple Sequence
Alignment

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
```

```
## Download Dataset
### HomFam

### extHomFam-v2

### ContTest

## Usage
`python run_alignment.py [options]`
### Arguments
- Commonly used options
    * `--input <file/folder>`- Specify the inpout protein sequence file or the folder contains multiple files. Once a folder is given, all files ending with .tfa will be aligned.
    * `--ref <file/folder>`- Specify the reference alignment fasta file or the folder contain multiple reference alignments. Notice that it should be the same type with input (i.e. if a file is given as input, only a file is allowed for this argument)
    * `--model_path <path to a .pt file>`- Specify the path to the pretrained model
    * `--embed_type <LSTM|esm-43M|esm35M|esm-150M|esm-650M>`- Specify which embedding model to use
    * `--align_prog <clustalo|mafft|famsa|tcoffee>`- Specify which program to use for alignment stage
    * `--no_tree`- If specified, will use the program given by --align_prog for the all process



- Other options
    * `--tree_dir <path to a folder>`- Specify a folder to place guide trees (default: ./trees)
    * `--msf_dir <path to a folder>`- Specify a folder to place final MSA sequence files (default: ./msf)
    * `--fasta_dir <path to a folder>`- Specify a folder to place temporary fasta files (default: ./fasta)
    * `--log_dir <path to a folder>`- Specify a folder to place outputed logs (default: ./logs)
    * `--gpu <gpu index>`- Specify which gpu to work on
    * `--seed <integer>`- Specify the seed for Pytorch, numpy, and random
    * `--thread <integer>`- Specify how many threads the MSA program (e.g. famsa) should work on
    * `--max_cluster_size <integer>`- Specify the threshold for the Bi-secting K-means++ algorithm
    * `--batch_size <integer>`- Specify the batch size for inference
    * `--num_workers <integer>`- Specify the thread for Pytorch dataLoader
    * `--newick2mafft_path <path to newick2mafft.rb>`- Specify the path to newick2mafft.rb, which is used to convert guide tree format for MAFFT
    * `--fastSP_path <path to FastSP.jar>`- Specify the path to FastSP.jar, which is used to evaluate the final alignment

