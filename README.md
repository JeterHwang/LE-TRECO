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
cp mafft ../../src/
```

### Compile Clustal Omega 1.2.4
The folder clustal-omega-1.2.4 is downloaded from the [Clustal Omega homepage](http://www.clustal.org/omega/). To make Clustal Omega work with our code, please follow the commands below.
```
cd clustal-omega-1.2.4
./configure 
make clean
make
make install
cp clustalo ../../src/
```

### Compile FAMSA v2.0.1
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
chmod +x T-COFFEE_installer_Version_"version"_linux_x64.bin
./T-COFFEE_installer_Version_"version"_linux_x64.bin
```
Follow the wizard instructions and complete the installation. Be sure to open a new terminal session after the installation. 

## Prepare Dataset
### HomFam

### extHomFam-v2

### ContTest

## Run the 

