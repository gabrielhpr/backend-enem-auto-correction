import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset
from tensorflow import keras


class EnemAutoCorrection( object ):

    def __init__( self ):
        pass

    def data_preparation(self, df):
        # Convert Pandas dataframe to hugging face Dataset
        df_dataset = Dataset.from_pandas( df );

        return df_dataset

    def tokenization(self, test_dataset, checkpoint ):
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        def tokenize_function(example):
            return tokenizer(example['essay'],
                            add_special_tokens=True,
                            padding='max_length',
                            truncation='longest_first',
                            max_length=512,
                            return_attention_mask=True)

        tokenized_test = test_dataset.map(tokenize_function, batched=True)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

        tf_test_dataset = tokenized_test.to_tf_dataset(
            columns=["attention_mask", "input_ids", "token_type_ids"],
            shuffle=False,
            collate_fn=data_collator,
            batch_size=8,
        )
        return tf_test_dataset

    def get_prediction( self, model, original_data, test_data ):

        # Get the logits from the prediction
        preds = model.predict( test_data )["logits"]
        y_hat = (np.max(preds, axis=1) * 1000).astype(int)

        original_data['prediction'] = y_hat

        return original_data.to_json(orient='records', date_format='iso')
