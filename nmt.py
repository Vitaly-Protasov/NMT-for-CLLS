from easynmt import EasyNMT
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import spacy
import os
import argparse


class NMT_easy:
    def __init__(self, model_name: str = 'opus-mt', device: str = 'cpu'):
        self.nmt_model = EasyNMT(model_name)
    
    def make_nmt(self, df: pd.DataFrame, language: str, outpath: str) -> None:
        lemmatization_model = spacy.load(language)
        new_df = deepcopy(df)
        translated_sentences = []
        for i, line in tqdm(new_df.iterrows()):
            sent = line.context.strip()
            target_w = line.target_word                                                    
            translated_sent = self.nmt_model.translate(sent, target_lang=self.language)
            lemmatized_sent = lemmatization_model(translated_sent)
            translated_sentences.append(lemmatized_sent)
        
        new_df['translations'] = translated_sentences
        new_df.to_csv(outpath, index=None)
        print('Dataframe with translations was saved as', outpath)


def main(outpath: str, dfpath: str, lang: str, model_name: str)->None:
    df = pd.read_csv(dfpath)
    assert all(c in df.columns() for c in ['context', 'target_word'])

    results = NMT_easy(model_name).make_nmt(df, lang, outpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=False, default='after_nmt.csv')
    parser.add_argument('--df_path', type=str, required=True)
    parser.add_argument('--language', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=False, default='opus-mt')
    args = parser.parse_args()
    main(args.output_path, args.df_path, args.language, args.model_name)
