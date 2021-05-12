# NMT-for-CLLS
Application of NMT+Word alignment models for cross-lingual lexical substitution task


# Firstly
```
pip install -r requirements.txt
bash init.sh
```


# Get NMT
* For example
```
from nmt import NMT_easy

nmt_model = NMT_easy(df, 'es', 'test.csv').make_nmt()
```
* Or
```
python3 nmt.py --df_path <path_to_df> --language <your_language>
```

# Get alignments
* For example
```
from alignments import Alignments

align_1 = Alignments(df, 'es', 'test1.csv').model_1()
align_1 = Alignments(df, 'es', 'test2.csv').model_2()
```