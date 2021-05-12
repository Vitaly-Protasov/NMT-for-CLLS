# NMT-for-CLLS
Application of NMT+Word alignment models for cross-lingual lexical substitution task


# Firstly
```
pip install -r requirements.txt
bash init.sh
```


# Get NMT
* For example
```python3
from nmt import NMT_easy

language = 'es'
outpath = 'test.csv'
model_name = 'opus-mt' # see https://github.com/UKPLab/EasyNMT
nmt_model = NMT_easy(language, outpath, model_name)
nmt_semeval2010_2 = make_nmt(df)
```
* Or
```
python3 nmt.py --df_path <path_to_df> --language <your_language>
```

# Get alignments
* For example
```python3
from alignments import Alignments

alignment_models = Alignments()
positions1, spanish_tws1 = alignment_models.model_1(df)
positions2, spanish_tws2 = alignment_models.model_2(df)
```

