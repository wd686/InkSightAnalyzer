import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset  
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Define model name (from hugging face repo)
model_name = "nusebacra/MultilabelClassifier_distilBERT_fine-tuned"



def aspectClassification(self, rawInput_file):

    #####################################################################################################

    # TODO TO UNCOMMENT

    # # Load pre-trained zero-shot classification model for aspect classification
    # classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # df = rawInput_file.copy()

    # # Pre-processing steps to strip each review into multiple sentences

    # def split_into_sentences(df):
    #     rows = []
        
    #     for idx, row in df.iterrows():
    #         sentences = sent_tokenize(row['Reviews'])  # Split the review into sentences
    #         for sentence in sentences:
    #             rows.append({'Review_ID': idx, 'Sentence': sentence})  # Keep track of Review ID
                
    #     return pd.DataFrame(rows)

    # # Apply the function to your rawInput_df
    # sentence_df = split_into_sentences(df)

    # # Run classification model for each sentence

    # # Aspects you're classifying
    # aspects = ['Price', 'Customer Service', 'Product Quality', 'Delivery']

    # def classify_aspect(row, aspects):
    #     # Classify aspect using the provided classifier
    #     aspect_result = classifier(row, aspects)
    #     aspect = aspect_result['labels'][0]  # The most probable aspect
    #     return aspect

    # # Apply the aspect classification
    # sentence_df['Aspect'] = sentence_df['Sentence'].apply(lambda row: classify_aspect(row, aspects))

    # aspectOutput_df = sentence_df.copy()
    
    #####################################################################################################

    # TODO TO COMMENT

    # Data as a list of dictionaries
    '''
    data = [
        {"Review_ID": 0, "Sentence": "Reasonable priced with a high capacity of prints.", "Aspect": "Price"},
        {"Review_ID": 1, "Sentence": "Quick delivery, easy to order!", "Aspect": "Delivery"},
        {"Review_ID": 2, "Sentence": "I bought HP ink for my printer and it is the only brand I will use.", "Aspect": "Product Quality"},
        ]
'''
    #####################################################################################################


    df = rawInput_file.copy()


    # Function to prepare dataset from text
    def prepare_dataset(texts, tokenizer, max_length=512):
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        return list(zip(encodings['input_ids'], encodings['attention_mask']))

    # Custom dataset for batching
    class TextDataset(Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings)

        def __getitem__(self, idx):
            return {
                'input_ids': self.encodings[idx][0],
                'attention_mask': self.encodings[idx][1]
            }

    # Function to load model, predict, and update DataFrame
    def predict_and_update_dataframe (model_name, df, text_column='Combined Text', output_labels=None, batch_size=16):
        # Load the tokenizer using relative paths
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            return

        # Instantiate the model and load the weights
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(output_labels))
            
        except Exception as e:
            print(f"Error loading model or weights: {e}")
            return

        # Move model to device (use CPU if GPU is not available or out of memory)
        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        device = torch.device('cpu')
        model.to(device)
        model.eval()

        # Prepare the dataset from the DataFrame
        dataset = df[text_column].tolist()  # Assuming 'Combined Text' contains the text data
        encodings = prepare_dataset(dataset, tokenizer)

        # Use DataLoader to load data in batches
        data_loader = DataLoader(TextDataset(encodings), batch_size=batch_size, shuffle=False)

        predictions = []

        # Perform inference in batches
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.sigmoid(logits).cpu().numpy()

                # Convert predictions to label columns
                for pred in preds:
                    labels = [output_labels[i] for i in range(len(output_labels)) if pred[i] >= 0.5]  # Threshold of 0.5
                    predictions.append(labels)

        # Add predictions back to the DataFrame
        df['Predicted Labels'] = predictions


    predict_and_update_dataframe(
        model_name,
        df,
        output_labels=['Delivery', 'Product Quality', 'Price', 'Customer Service'],
        batch_size=8  
    )

    # Create DataFrame
    aspectOutput_df = pd.DataFrame(df)

    return aspectOutput_df