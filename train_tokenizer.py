
import pandas as pd
import sentencepiece as spm
import os
df = pd.read_csv('en-fr.csv', nrows=2**23)

# Rename the columns to src and tgt
df.rename(columns={'en': 'src', 'fr': 'tgt'}, inplace=True)

def train_tokenizers(text_list, tokenizer_path):
    # Cleanup the tokenizer_path
    tokenizer_path = tokenizer_path.split('.')[0]
    
    # Temporary file to write the string to
    with open('tmp.txt', 'w') as f:
        # Write the data to the file in chunks of 5 entries per line
        for i in range(0, len(text_list), 5):
            chunk = text_list.iloc[i:i+5]
            chunk = chunk.astype(str)
            f.write('\t'.join(chunk) + '\n')
        
    # Train the tokenizer
    spm.SentencePieceTrainer.train(input='tmp.txt', model_prefix=tokenizer_path, 
                                    vocab_size=32768, model_type='bpe',
                                    bos_id=1, eos_id=2, pad_id=0, unk_id=3,
                                    bos_piece='<s>', eos_piece='</s>', 
                                    pad_piece='<pad>', unk_piece='<unk>',
                                    normalization_rule_name="nmt_nfkc",
                                    remove_extra_whitespaces=True,
                                    shuffle_input_sentence=True,
                                    character_coverage=0.9999,
                                    split_digits=True,
                                    split_by_unicode_script=True,
                                    split_by_whitespace=True,
                                    split_by_number=True,
                                    add_dummy_prefix=True)
    
    # Remove the temporary file
    os.remove('tmp.txt')
        
    

train_tokenizers(df['src'], 'en_tokenizer')
train_tokenizers(df['tgt'], 'fr_tokenizer')


