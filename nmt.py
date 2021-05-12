from easynmt import EasyNMT
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import spacy
import os
import argparse


class NMT_easy:
    def __init__(self, df: pd.DataFrame, language: str, outpath: str):
        self.nmt_model = EasyNMT('opus-mt')
        self.lemmatization_model = spacy.load(language)
        self.language = language
        self.df = df
        self.outpath = outpath
    
    def make_nmt(self) -> None:
        new_df = deepcopy(self.df)
        translated_sentences = []
        for i, line in tqdm(new_df.iterrows()):
            sent = line.context.strip()
            target_w = line.target_word                                                    
            translated_sent = self.nmt_model.translate(sent, target_lang=self.language)
            lemmatized_sent = self.lemmatization_model(translated_sent)
            translated_sentences.append(lemmatized_sent)
        
        new_df['translations'] = translated_sentences
        new_df.to_csv(outpath, index=None)
        print('Dataframe with translations was saved as', self.outpath)


def main(outpath: str, dfpath: str, lang: str)->None:
    df = pd.read_csv(dfpath)
    assert all(c in df.columns() for c in ['context', 'target_word'])

    results = NMT_easy().make_nmt(df, lang, outpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=False, default='after_nmt.csv')
    parser.add_argument('--df_path', type=str, required=True)
    parser.add_argument('--language', type=str, required=True)
    args = parser.parse_args()
    main(args.output_path, args.df_path, args.language)
