from modelling.transformer import Transformer
from dataset.translation_dataset import TranslationDataset
from transformers import GPT2Tokenizer
import torch.nn.functional as F

from tqdm import tqdm
import torch
import json
import evaluate
import torch.nn as nn
from pathlib import Path
from pprint import pprint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerTester(nn.Module):
     """
     TransformerTester module for testing Transformer models.

     Args:
          test_dataset (TranslationDataset): Testing dataset.
          d_model (int): Dimensionality of model.
          vocab_size (int): Size of vocabulary.
          max_seq_len (int): Maximum sequence length.
          num_heads (int): Number of attention heads.
          num_decoder_layers (int): Number of decoder layers.
          num_encoder_layers (int): Number of encoder layers.
          dim_feedforward (int): Dimensionality of feedforward layer.
          dropout (float): Dropout probability.
          src_pad_idx (int): Padding index for source sequence.
          trg_eos_idx (int): End-of-sequence index for target sequence.
          trg_sos_idx (int): Start-of-sequence index for target sequence.
          ckpt_name (str): Name of the checkpoint file.
     """
     def __init__(self,
                 test_dataset: TranslationDataset,
                 d_model: int,
                 vocab_size: int,
                 max_seq_len: int,
                 num_heads: int,
                 num_decoder_layers: int,
                 num_encoder_layers: int,
                 dim_feedforward: int,
                 dropout: float,
                 src_pad_idx: int,
                 trg_eos_idx: int,
                 trg_sos_idx: int,
                 ckpt_name: str) -> None:
        super().__init__()


        self.ckpt_name = Path(ckpt_name)

        self.test_data = test_dataset

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.src_pad_idx = src_pad_idx
        self.trg_eos_idx = trg_eos_idx
        self.trg_sos_idx = trg_sos_idx

        self.model = Transformer(vocab_size=self.vocab_size,
                                 d_model=self.d_model,
                                 num_heads=self.num_heads,
                                 num_encoder_layers=self.num_encoder_layers,
                                 num_decoder_layers=self.num_decoder_layers,
                                 dim_feedforward=self.dim_feedforward,
                                 dropout=self.dropout,
                                 max_len=self.max_seq_len)
        
        trained_model = torch.load(self.ckpt_name, map_location=torch.device(device))
        
        self.model = self.model.to(device)

        self.model.load_state_dict(trained_model['model'])
        
        self.tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('./resources/gpt2_from_pretrained')

     def evaluate_bleu(self, results):
          actual_translations = [triple['true_trg'] for triple in results]
          generated_translations = [triple['generated_trg'] for triple in results]

          bleu = evaluate.load("bleu")

          blue_score = bleu.compute(predictions=generated_translations, references=actual_translations)

          pprint(blue_score)

     def test(self):
          """
          Test the trained model on the test dataset and save the results to a JSON file.
          """
          self.model.eval()
          results = []
          pbar = tqdm(enumerate(self.test_data))
          for _, pair in pbar:
               src_input, trg_output = pair['source'].to(device), pair['target_output'].to(device)
               encoder_mask = (src_input.unsqueeze(0) != self.src_pad_idx).int()
               encoder_output = self.model.encode(src_input.unsqueeze(0).to(device), encoder_mask.to(device))

               last_words = torch.LongTensor([self.src_pad_idx] * self.max_seq_len)
               last_words[0] = self.trg_sos_idx
               current_length = 1

               for i in range(self.max_seq_len):
                    decoder_mask = (last_words.unsqueeze(0) != self.src_pad_idx).int()
                    decoder_output = self.model.decode(encoder_output.to(device), encoder_mask.to(device), last_words.unsqueeze(0).to(device), decoder_mask.to(device))
                    output = self.model.projection(decoder_output)
                    softmax_output = F.softmax(output, dim=-1)
                    output = torch.argmax(softmax_output, dim=-1)
                    last_word_id = output[0][i].cpu().item()

                    if i < self.max_seq_len - 1:
                         last_words[i + 1] = last_word_id
                         current_length += 1

                    if last_word_id == self.src_pad_idx or last_word_id == self.trg_eos_idx:
                         break


               decoded_output = last_words[1:].tolist()

               src_data_filtered = [token for token in src_input.cpu().tolist() if token not in [self.src_pad_idx]]
               tgt_data_filtered = [token for token in trg_output.cpu().tolist() if token not in [self.src_pad_idx]]
               decoded_output_filtered = [token for token in decoded_output if token not in [self.src_pad_idx]]

               src_decoded = self.tokenizer_gpt2.decode(src_data_filtered, skip_special_tokens=True)
               true_output_decoded = self.tokenizer_gpt2.decode(tgt_data_filtered, skip_special_tokens=True)
               generated_output_decoded = self.tokenizer_gpt2.decode(decoded_output_filtered, skip_special_tokens=True)

               results.append({'src': src_decoded, 'true_trg': true_output_decoded, 'generated_trg': generated_output_decoded})

          with open('./resources/results/test_results.json', 'w') as json_file:
               json.dump(results, json_file, indent=2)

          self.evaluate_bleu(results=results)




          
