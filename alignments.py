from WordAlignment import WordAlignment
from tqdm import tqdm
import re
import transformers
from typing import List, Tuple
import torch
import itertools
import pandas as pd


class Alignments:
    def __init__(self):
        BERT_NAME: str = "bert-base-multilingual-cased"
        self.model1 = WordAlignment(model_name=BERT_NAME, tokenizer_name=BERT_NAME, device='cpu', fp16=False)
        self.model2 = transformers.BertModel.from_pretrained(BERT_NAME)
        self.tokenizer2 = transformers.BertTokenizer.from_pretrained(BERT_NAME)
    
    def _clear_word(self, word: str) -> str:
        return re.sub(r'[^\w]', '', word.lower().strip())
    
    def model_1(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        positions = []
        spanish_tws = []
        for i, line in tqdm(df.iterrows()):
            if hasattr(line, 'context') and hasattr(line, 'translations') and hasattr(line, 'target_word'):
                sent1 = line.context
                sent2 = line.translations
                target_w = line.target_word

                _, decoded = self.model1.get_alignment(sent1.split(), sent2.split(), calculate_decode=True)

                for sentence1_w, sentence2_w in decoded:
                    sentence1_w = self._clear_word(sentence1_w)
                    sentence2_w = self._clear_word(sentence2_w)
                    if sentence1_w == target_w:
                        start = sent2.find(sentence2_w)
                        end = start + len(sentence2_w)
                        positions.append(f'{start}-{end}')
                        spanish_tws.append(sentence2_w)
                        break
                else:
                    positions.append(f'{0}-{0}')
                    spanish_tws.append('')
            else:
                raise Exception('Object \'line\' has no attributes: context, translations, target_word')
        return positions, spanish_tws

    def _align_each_pair(self, str_text: str, target_text: str) -> Tuple[List[str], List[str]]:
        # pre-processing
        sent_src, sent_tgt = str_text.strip().split(), target_text.strip().split()
        token_src, token_tgt = [self.tokenizer2.tokenize(word) for word in sent_src], [self.tokenizer2.tokenize(word) for word in sent_tgt]
        wid_src, wid_tgt = [self.tokenizer2.convert_tokens_to_ids(x) for x in token_src], [self.tokenizer2.convert_tokens_to_ids(x) for x in token_tgt]
        ids_src, ids_tgt = self.tokenizer2.prepare_for_model(list(itertools.chain(*wid_src)), \
            return_tensors='pt', model_max_length=self.tokenizer2.model_max_length, truncation=True)['input_ids'], \
                self.tokenizer2.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, \
                    model_max_length=self.tokenizer2.model_max_length)['input_ids']
        sub2word_map_src = []
        for i, word_list in enumerate(token_src):
            sub2word_map_src += [i for x in word_list]
        sub2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            sub2word_map_tgt += [i for x in word_list]

        # alignment
        align_layer = 8
        threshold = 1e-3
        self.model2.eval()
        with torch.no_grad():
            out_src = self.model2(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
            out_tgt = self.model2(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

            dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

            softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
            softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

            softmax_inter = (softmax_srctgt > threshold)*(softmax_tgtsrc > threshold)

        align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
        align_words = set()
        for i, j in align_subwords:
            align_words.add( (sub2word_map_src[i], sub2word_map_tgt[j]))
        
        return sent_src, sent_tgt

    def model_2(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        positions = []
        spanish_tws = []
        for i, line in tqdm(df.iterrows()):
            sent1 = line.context
            sent2 = line.translations
            target_w = line.target_word

            sent1_align, sent2_align = self._align_each_pair(sent1, sent2)

            for (sentence1_w, sentence2_w) in zip(sent1_align, sent2_align):
                sentence1_w = self._clear_word(sentence1_w)
                sentence2_w = self._clear_word(sentence2_w)
                if sentence1_w == target_w:
                    start = sent2.find(sentence2_w)
                    end = start + len(sentence2_w)
                    positions.append(f'{start}-{end}')
                    spanish_tws.append(sentence2_w)
                    break
            else:
                positions.append(f'{0}-{0}')
                spanish_tws.append('')
        return positions, spanish_tws
