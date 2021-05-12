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

nmt_model = NMT_easy(df, 'es', 'test.csv').make_nmt()
```
* Or
```
python3 nmt.py --df_path <path_to_df> --language <your_language>
```

# Get alignments
* For example
```python3
from alignments import Alignments

positions1, spanish_tws1 = Alignments(df).model_1()
positions2, spanish_tws2 = Alignments(df).model_2()
```

