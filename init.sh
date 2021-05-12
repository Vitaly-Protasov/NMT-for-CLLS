git clone https://github.com/neulab/awesome-align.git
python -m spacy download es
cd awesome-align/
python setup.py install
cd ../
git clone https://github.com/andreabac3/Word-Alignment-BERT.git
cp -r Word-Alignment-BERT/WordAlignment.py .
rm -rf Word-Alignment-BERT/ awesome-align/