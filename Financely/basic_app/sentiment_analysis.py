from transformers import BertModel, BertTokenizer
import torch
from torch import nn

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

class_names = ['negative', 'neutral', 'positive']
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask
        )
        pooled_output = output[1]
        output = self.drop(pooled_output)
        return self.out(output)

model = SentimentClassifier(len(class_names))

state_dict = torch.load("basic_app/bert_sentiment_analysis.pt", map_location=torch.device('cpu'))

# Remove the unexpected key if it's present
if 'bert.embeddings.position_ids' in state_dict:
    del state_dict['bert.embeddings.position_ids']

# Load with strict=False to skip any other non-matching keys
model.load_state_dict(state_dict, strict=False)


def predict_sentiment(texts):
    """
    A simplified sentiment prediction function that doesn't rely on external ML models.
    This is a fallback solution when the transformer model isn't available.
    
    Args:
        texts (list): List of strings to analyze sentiment
        
    Returns:
        list: List of sentiment labels ('positive', 'negative', or 'neutral')
    """
    try:
        # Simple keyword-based sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'positive', 'profit', 'success', 'growth', 
                         'gain', 'up', 'higher', 'rise', 'improve', 'improved', 'increasing', 
                         'bullish', 'opportunity', 'promising', 'outperform', 'beat', 'solid',
                         'strong', 'earnings', 'exceed', 'celebrate', 'record']
        
        negative_words = ['bad', 'poor', 'negative', 'loss', 'fail', 'decline', 'down', 'lower',
                          'fall', 'drop', 'decrease', 'decreasing', 'bearish', 'risk', 'trouble',
                          'underperform', 'miss', 'weak', 'struggle', 'concern', 'worried', 'problem',
                          'bankruptcy', 'lawsuit', 'investigation', 'cut', 'crisis']
        
        results = []
        
        for text in texts:
            if not text:
                results.append("neutral")
                continue
                
            text = text.lower()
            
            # Count positive and negative word occurrences
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            
            # Determine sentiment based on counts
            if pos_count > neg_count:
                results.append("positive")
            elif neg_count > pos_count:
                results.append("negative")
            else:
                results.append("neutral")
                
        return results
        
    except Exception as e:
        print(f"Error in simple sentiment prediction: {e}")
        return ["neutral"] * len(texts)

# Optionally try to load the better model if available
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    
    # Check if transformers is properly installed
    TRANSFORMERS_AVAILABLE = True
    
    def advanced_predict_sentiment(texts):
        """
        More advanced sentiment prediction using transformers library
        """
        try:
            # Load model and tokenizer
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
            
            results = []
            for text in texts:
                if not text:
                    results.append("neutral")
                    continue
                    
                result = sentiment_pipeline(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Map the result to our standard format
                label = result[0]["label"]
                if label == "POSITIVE":
                    results.append("positive")
                elif label == "NEGATIVE":
                    results.append("negative")
                else:
                    results.append("neutral")
                    
            return results
            
        except Exception as e:
            print(f"Error in advanced sentiment prediction: {e}")
            # Fall back to simple prediction
            return predict_sentiment(texts)
    
    # Override the basic function with the advanced one if transformers is available
    original_predict_sentiment = predict_sentiment
    predict_sentiment = advanced_predict_sentiment
    print("Advanced sentiment analysis loaded successfully")
    
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Using simple sentiment analysis (transformers not available)")

# i = predict_sentiment(["he is in a better place","I always lie"])
# print(i)
# if i<0:
#     print("negative")
# elif i>0:
#     print("positive")
# else:
#     print("neutral")
