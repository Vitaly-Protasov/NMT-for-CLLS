import subprocess
from google_trans_new import google_translator
from tqdm import tqdm
from typing import List

def substitute_nonunicode_letters(nonunicode_word: str):
    substitution_word = nonunicode_word[:]
    substitution_letters_pairs = {
        'ñ': 'n', 'ó': 'o', 'í': 'i', 'é': 'e', 'á': 'a', 'ú': 'u', 'ï': 'i', 'ṅ': 'n', 'ā': 'a',
        'а́': 'а', 'ы́': 'ы', 'у́': 'y', 'и́': 'и', 'ю́': 'ю', 'е́': 'е', 'о́': 'о', 
    }
    for i in substitution_letters_pairs:
        if i in substitution_letters_pairs:
            substitution_word = substitution_word.replace(i, substitution_letters_pairs[i])

    return substitution_word


def print_results_semeval2010(best_cands_path: str, gold_filepath: str, path_to_score_pl: str) -> None: 
    command_list = ['perl', path_to_score_pl, best_cands_path, gold_filepath]

    result = subprocess.run(command_list, 
                            stderr=subprocess.PIPE,
                            stdout=subprocess.PIPE, encoding='utf-8')
    output = result.stdout.split('\n')
    for i in output:
        print(i)

def get_google_translations(df: pd.DataFrame, language: str, outpath: str):
    translator = google_translator()
    translated_sentences: List[str] = []
    for i, line in tqdm(df.iterrows()):
        sent = line.context.strip()
        target_w = line.target_word                                                    
        translated_sent = translator.translate(sent, lang_tgt=language)
        translated_sentences.append(translated_sent)
    df['translations'] = translated_sentences
    df.to_csv(outpath, index=None)
    print('Dataframe with translations was saved as', outpath)


def from_csv_to_file_best(df: pd.DataFrame, outpath: str) -> None:
    f_test_result = open(outpath, 'w+')
    delimeter_best = ' :: '

    for i, line in df.iterrows():
        lemma_cand = substitute_nonunicode_letters(line.tw_translations)
        string_to_write = line.tw_for_metrics + delimeter_best + str(lemma_cand) + '\n'
        f_test_result.write(string_to_write)
    f_test_result.close()
