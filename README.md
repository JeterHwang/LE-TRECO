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
apt -y install git
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

### Download FastSP
To enable the evaluation of the MSA against reference alignment, download the FastSP with the following command.
```
git clone https://github.com/smirarab/FastSP.git
```

### Download esm
Download the code of [esm](https://github.com/facebookresearch/esm) that is adapted by us to fit our design. Follow the commands below to download the [code](https://drive.google.com/file/d/1cYCictQsOqc4QsuF4kVLH3_eVvJa2Dyl/view?usp=sharing) and place it in the right location.
```
cd ./src
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1cYCictQsOqc4QsuF4kVLH3_eVvJa2Dyl' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1cYCictQsOqc4QsuF4kVLH3_eVvJa2Dyl" -O esm.zip && rm -rf /tmp/cookies.txt
unzip esm.zip
rm esm.zip
```

### Compile MAFFT 7.490
Download mafft-7.490-with-extensions from the [MAFFT source page](https://mafft.cbrc.jp/alignment/software/source.html), and make MAFFT work with our code. Please follow the commands below.
```
wget https://mafft.cbrc.jp/alignment/software/mafft-7.490-with-extensions-src.tgz
tar zxvf mafft-7.490-with-extensions-src.tgz
cd mafft-7.490-with-extensions/core
make clean
make
make install
cp ../scripts/mafft ../../src/
```

### Compile Clustal Omega 1.2.4
Download the source code of clustal-omega-1.2.4 from the [Clustal Omega homepage](http://www.clustal.org/omega/), and make Clustal Omega work with our code. Please follow the commands below.
```
wget http://www.clustal.org/omega/clustal-omega-1.2.4.tar.gz
tar zxvf clustal-omega-1.2.4.tar.gz
cd clustal-omega-1.2.4
./configure 
make clean
make
cp clustalo ../../src/
```

### Compile and Install FAMSA v2.1.2
Download the source code of FAMSA-2.1.2 from [FAMSA](https://github.com/refresh-bio/FAMSA), and make FAMSA work with our code. Please follow the commands below.
```
wget https://github.com/refresh-bio/FAMSA/archive/refs/tags/v2.1.2.zip
unzip v2.1.2.zip
cd FAMSA-2.1.2
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
One can download the ContTest dataset from the Google drive link: [extHomFam-v2.zip](https://drive.google.com/file/d/1JEdegu9ktyzJrel9hexDmmxiFos6Nb-B/view?usp=sharing) or type the following command in the terminal.
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1JEdegu9ktyzJrel9hexDmmxiFos6Nb-B' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1JEdegu9ktyzJrel9hexDmmxiFos6Nb-B" -O extHomFam-v2.zip && rm -rf /tmp/cookies.txt
unzip extHomFam-v2.zip
rm extHomFam-v2.zip
```
### ContTest
One can download the ContTest dataset from the Google drive link: [ContTest.zip](https://drive.google.com/file/d/1XwcPso7c8UwBVORjRuUqW9Qi6mhSgS5P/view?usp=sharing) 
or type the following command in the terminal.
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1XwcPso7c8UwBVORjRuUqW9Qi6mhSgS5P' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1XwcPso7c8UwBVORjRuUqW9Qi6mhSgS5P" -O ContTest.zip && rm -rf /tmp/cookies.txt
unzip ContTest.zip
rm ContTest.zip
```
To use ContTest, please follow the instructions below

1. Compile FreeContact
```
cd FreeContact
chmod +x configure
./configure --with-boost-libdir=/usr/lib/x86_64-linux-gnu/
autoreconf -ivf
make
cp src/freecontact /usr/local/bin/
```
2. Compile PSICOV
```
cd PSICOV/fasta2aln
cc -O fasta2aln.c -o fasta2aln
cp fasta2aln /usr/local/bin/
cd ..
gcc -O3 -march=native -ffast-math -m64 -ftree-vectorize -fopenmp psicov21.c -lm -o psicov
cp psicov /usr/local/bin/
```
3. scripts
```
cd scripts
chmod 777 needle.py
chmod 777 runbenchmark
chmod 777 calculatescoreL5  
chmod 777 labelcontactsev23  
chmod 777 labelcontactspsi23
chmod 777 reorderFASTA
cp needle.py calculatescoreL5 labelcontactsev23 labelcontactspsi23 reorderFASTA runbenchmark /usr/local/bin/
```
On evaluation, put the alignments into the same folder as the input fasta files. Then run the command `runbenchmark -a yourmethod` in the folder in which datasets.txt is located. That way, all the family written in datasets.txt will be evaluated. The final scores will be shown in the folder named "results"

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
    * `--rfa_dir <path to a folder>`- Specify a folder to place temporary rfa files (default: ./rfa)
    * `--log_dir <path to a folder>`- Specify a folder to place outputed logs (default: ./logs)
    * `--gpu <gpu index>`- Specify which gpu to work on
    * `--seed <integer>`- Specify the seed for Pytorch, numpy, and random
    * `--thread <integer>`- Specify how many threads the MSA program (e.g. famsa) should work on
    * `--max_cluster_size <integer>`- Specify the threshold for the Bi-secting K-means++ algorithm
    * `--batch_size <integer>`- Specify the batch size for inference, the recommended batch size for LSTM is 64, and 16 for transformers
    * `--num_workers <integer>`- Specify the thread for Pytorch dataLoader
    * `--newick2mafft_path <path to newick2mafft.rb>`- Specify the path to newick2mafft.rb, which is used to convert guide tree format for MAFFT
    * `--fastSP_path <path to FastSP.jar>`- Specify the path to FastSP.jar, which is used to evaluate the final alignment

## Examples
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
Once the program terminates, the SP and TC scores will show on the terminal. Besides, one can inspect the execution details in the log files. 
## Citation
