import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset  
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define model name (from hugging face repo)
model_name = "nusebacra/MultilabelClassifier_distilBERT_fine-tuned"

def aspectClassification(self, rawInput_file):

    #####################################################################################################

    # TODO REMAIN AS COMMENT

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

    # TODO TO UNCOMMENT

    # # Data as a list of dictionaries
    # data = [
    #     {"Review_ID": 0, "Sentence": "Reasonable priced with a high capacity of prints.", "Aspect": "Price"},
    #     {"Review_ID": 1, "Sentence": "Quick delivery, easy to order!", "Aspect": "Delivery"},
    #     {"Review_ID": 2, "Sentence": "I bought HP ink for my printer and it is the only brand I will use.", "Aspect": "Product Quality"},
    #     {"Review_ID": 2, "Sentence": "The quality of my printouts is excellent.", "Aspect": "Product Quality"},
    #     {"Review_ID": 2, "Sentence": "The colors are vibrant and true.", "Aspect": "Product Quality"},
    #     {"Review_ID": 3, "Sentence": "Best price for good ink cartridge.", "Aspect": "Price"},
    #     {"Review_ID": 3, "Sentence": "Easy to install and ink is always good.", "Aspect": "Product Quality"},
    #     {"Review_ID": 4, "Sentence": "always buy brand have had such bad luck with aftermarket ink.", "Aspect": "Product Quality"},
    #     {"Review_ID": 5, "Sentence": "Have used HP printers and ink for over 40 years and have never been disappointed in their quality or the clarity of the printing", "Aspect": "Product Quality"},
    #     {"Review_ID": 6, "Sentence": "extremely unhappy printer allow printing unless cartridge ink exclusively print black ink color cartridge fail force purchase entire packet color cartridge extremely high expense buy cheap cartridge elsewhere forego replace printer another product", "Aspect": "Price"},
    #     {"Review_ID": 7, "Sentence": "No problems.", "Aspect": "Price"},
    #     {"Review_ID": 7, "Sentence": "It works well.", "Aspect": "Product Quality"},
    #     {"Review_ID": 8, "Sentence": "This is the ink that works for my printer", "Aspect": "Product Quality"},
    #     {"Review_ID": 9, "Sentence": "ink fill need office", "Aspect": "Price"},
    #     {"Review_ID": 10, "Sentence": "It does what it's supposed to do.", "Aspect": "Price"},
    #     {"Review_ID": 11, "Sentence": "Good progress duct", "Aspect": "Delivery"},
    #     {"Review_ID": 12, "Sentence": "as expected just wish it had more ink in it.", "Aspect": "Product Quality"},
    #     {"Review_ID": 13, "Sentence": "Fast shipping and good price", "Aspect": "Price"},
    #     {"Review_ID": 14, "Sentence": "Always use HP ink in our printers.", "Aspect": "Product Quality"},
    #     {"Review_ID": 14, "Sentence": "Lasts better for our needs.", "Aspect": "Price"},
    #     {"Review_ID": 15, "Sentence": "Great Product", "Aspect": "Product Quality"},
    #     {"Review_ID": 16, "Sentence": "its an excellent fit for my hp officejet 4650..", "Aspect": "Product Quality"},
    #     {"Review_ID": 17, "Sentence": "My printer uses these number printer ink glad I can order from Walmart.", "Aspect": "Product Quality"},
    #     {"Review_ID": 18, "Sentence": "Excllent quality; easy to order, received promptly.", "Aspect": "Product Quality"},
    #     {"Review_ID": 19, "Sentence": "As expected.", "Aspect": "Price"},
    #     {"Review_ID": 19, "Sentence": "I have puchased this cartridge for several years.", "Aspect": "Price"},
    #     {"Review_ID": 20, "Sentence": "Never had a problem and last me a few months and I print often.", "Aspect": "Product Quality"},
    #     {"Review_ID": 21, "Sentence": "good ink ship arrive good condition", "Aspect": "Product Quality"},
    #     {"Review_ID": 22, "Sentence": "I wish i could tell you.", "Aspect": "Price"},
    #     {"Review_ID": 22, "Sentence": "We think it was tossed in the trash.", "Aspect": "Price"},
    #     {"Review_ID": 22, "Sentence": "DO NOT do this hahahaha", "Aspect": "Price"},
    #     {"Review_ID": 23, "Sentence": "Order came very quick!", "Aspect": "Delivery"},
    #     {"Review_ID": 24, "Sentence": "Why would you purchase any other ink for your HP printer?", "Aspect": "Price"},
    #     {"Review_ID": 25, "Sentence": "The quality of printing is so much better using HP ink", "Aspect": "Product Quality"},
    #     {"Review_ID": 26, "Sentence": "The quality of printing is so much better using HP ink", "Aspect": "Product Quality"},
    #     {"Review_ID": 27, "Sentence": "Been an HP user and customer for a long time.", "Aspect": "Product Quality"},
    #     {"Review_ID": 27, "Sentence": "Last two color cartridges have not printed correctly.", "Aspect": "Product Quality"},
    #     {"Review_ID": 27, "Sentence": "Followed instructions on-line and recommendations by HP.", "Aspect": "Price"},
    #     {"Review_ID": 27, "Sentence": "Never could get the last color cartridges to print correctly.", "Aspect": "Product Quality"},
    #     {"Review_ID": 27, "Sentence": "Ran cleaner sequence twice.", "Aspect": "Price"},
    #     {"Review_ID": 27, "Sentence": "As last time I had to go to a well known office store to buy a color cartridge that worked.", "Aspect": "Product Quality"},
    #     {"Review_ID": 28, "Sentence": "This ink lasts.", "Aspect": "Product Quality"},
    #     {"Review_ID": 28, "Sentence": "I do a lot of printing.", "Aspect": "Price"},
    #     {"Review_ID": 28, "Sentence": "The quality of the ink is great.", "Aspect": "Product Quality"},
    #     {"Review_ID": 28, "Sentence": "I use it to type documents, photos, etc.", "Aspect": "Delivery"},
    #     {"Review_ID": 29, "Sentence": "ink last lot print quality ink great use type document photo etc", "Aspect": "Product Quality"},
    #     {"Review_ID": 30, "Sentence": "I was on a tight deadline and ran out of ink-I went to my closest Walmart, went straight back to where the ink is under locked display, saw what I needed-A gentleman working in the area came right over, got the cartridge for me and checked me out-In and out in 10 minutes!", "Aspect": "Customer Service"},
    #     {"Review_ID": 30, "Sentence": "!", "Aspect": "Price"},
    #     {"Review_ID": 31, "Sentence": "customer long time try printer nothing compare quality printer", "Aspect": "Product Quality"},
    #     {"Review_ID": 32, "Sentence": "Exactly what I ordered and shipped really fast!", "Aspect": "Product Quality"},
    #     {"Review_ID": 33, "Sentence": "Exactly what I ordered and shipped really fast!", "Aspect": "Product Quality"},
    #     {"Review_ID": 34, "Sentence": "Quality product", "Aspect": "Product Quality"},
    #     {"Review_ID": 35, "Sentence": "Works great on my Computer.", "Aspect": "Product Quality"},
    #     {"Review_ID": 36, "Sentence": "XL lasts a long time in my HP OfficeJet 4650.", "Aspect": "Product Quality"},
    #     {"Review_ID": 37, "Sentence": "XL lasts a long time in my HP OfficeJet 4650.", "Aspect": "Product Quality"},
    #     {"Review_ID": 38, "Sentence": "Been using this ink for years and even though it has gone up a little it's still a good price.", "Aspect": "Price"},
    #     {"Review_ID": 39, "Sentence": "Just opened a brand new one and my printer won't read it because it's defective inside.", "Aspect": "Product Quality"},
    #     {"Review_ID": 39, "Sentence": "I compared it to my old cartridge and the new one is all scratched.", "Aspect": "Price"},
    #     {"Review_ID": 39, "Sentence": "And I can't return it to the store because it's been opened.", "Aspect": "Product Quality"},
    #     {"Review_ID": 39, "Sentence": "The picture on the left is the old one.", "Aspect": "Price"},
    #     {"Review_ID": 39, "Sentence": "The one on the right is the new one.", "Aspect": "Price"},
    #     {"Review_ID": 40, "Sentence": "I have been buying this for years works great..", "Aspect": "Product Quality"},
    #     {"Review_ID": 41, "Sentence": "Lasts longer than the 63 but I find this printer uses much more ink than my previous printer", "Aspect": "Product Quality"},
    #     {"Review_ID": 42, "Sentence": "always buy authentic hp products with hp printers or else u risk the chance of ur printer not operating correctly...", "Aspect": "Product Quality"},
    #     {"Review_ID": 43, "Sentence": "I like that I can count on my go to store, Walmart, to always carry this brand of printer ink for my Photosmart 7510 printer.", "Aspect": "Product Quality"},
    #     {"Review_ID": 43, "Sentence": "It lasts a long time and the quality is great.", "Aspect": "Product Quality"},
    #     {"Review_ID": 44, "Sentence": "worked like it was suppose to.", "Aspect": "Price"},
    #     {"Review_ID": 45, "Sentence": "This HP black 902cartridge broke my HP printer when I inserted the cartridge.It RUINED my printer and I had to pay $500 to have a new printer purchased at MircoCenter and installed to my HP computer.", "Aspect": "Price"},
    #     {"Review_ID": 45, "Sentence": "I think Walmart is liable for he $500  if they stand behind their merchandise.", "Aspect": "Price"},
    #     {"Review_ID": 46, "Sentence": "far good work 6455 printer", "Aspect": "Product Quality"},
    #     {"Review_ID": 47, "Sentence": "great support from HP", "Aspect": "Customer Service"},
    #     {"Review_ID": 48, "Sentence": "Great service fast felivery", "Aspect": "Customer Service"},
    #     {"Review_ID": 49, "Sentence": "I just got the ink cartridge today and it was a bit pricy.", "Aspect": "Price"},
    #     {"Review_ID": 49, "Sentence": "Printed few papers and it seems to go well, it's doing better than the refurbished cartridge I got from Office Depot that's for sure.", "Aspect": "Product Quality"}
    # ]

    # # Create DataFrame
    # aspectOutput_df = pd.DataFrame(data)

    #####################################################################################################

    # TODO COMMENT

    df = rawInput_file.copy()

    aspect_list = ['Delivery', 'Product Quality', 'Price', 'Customer Service']

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
    def predict_and_update_dataframe (model_name, df, text_column='Reviews', output_labels=None, batch_size=16):
        # Load the tokenizer using relative paths
        # try:

        # import pdb;pdb.set_trace()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # except Exception as e:
        #     print(f"Error loading tokenizer: {e}")
        #     return

        # Instantiate the model and load the weights
        # try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(output_labels))
            
        # except Exception as e:
        #     print(f"Error loading model or weights: {e}")
        #     return

        # Move model to device (use CPU if GPU is not available or out of memory)
        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        device = torch.device('cpu')
        model.to(device)
        model.eval()

        # Prepare the dataset from the DataFrame
        dataset = df[text_column].tolist()  # Assuming 'Reviews' contains the text data
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

    # import pdb;pdb.set_trace()

    predict_and_update_dataframe(
        model_name,
        df,
        output_labels=aspect_list,
        batch_size=8  
    )

    # Create DataFrame
    aspectOutput_df = pd.DataFrame(df)

    return aspectOutput_df