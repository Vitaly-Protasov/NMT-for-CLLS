# NMT-for-CLLS [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1alF33i96d-AEeYOfwfN0Ei-HTWEJ_1f9?usp=sharing)
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


model_name = 'opus-mt' # see https://github.com/UKPLab/EasyNMT

language = 'es'
outpath = 'test.csv'

nmt_model = NMT_easy(model_name)
nmt_semeval2010_2 = nmt_model.make_nmt(<your_df: pd.DataFrame>, language, outpath)
```
* Or
```
python3 nmt.py --df_path <path_to_df> --language <your_language> --model_name <nmt_model_name>
```

# Get alignments
* For example
```python3
from alignments import Alignments

alignment_models = Alignments()
positions1, spanish_tws1 = alignment_models.model_1(df)
positions2, spanish_tws2 = alignment_models.model_2(df)
```

