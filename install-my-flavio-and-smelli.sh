# Prepare directories
cd ~
mkdir my-packages
cd my-packages

# install my version of smelli first
git clone https://github.com/hoodyn/smelli.git
cd smelli
python3.6 -m pip install -e . --user
cd .. 
 
# smelli automatically installs the official flavio. Lets remove it:
python3.6 -m pip uninstall flavio

# install my version of flavio
git clone https://github.com/hoodyn/flavio.git
cd flavio
python3.6 -m pip install -e .[plotting,testing] --user
cd .. 
