# practical_implementation/nlp_spacy.py
"""
NLP with spaCy
Named Entity Recognition and Sentiment Analysis on Amazon Reviews
"""

import spacy
from spacy import displacy
import pandas as pd
import random
from collections import Counter
import matplotlib.pyplot as plt

class AmazonReviewAnalyzer:
    def __init__(self):
        # Load spaCy English model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️ spaCy model not found. Please install with: python -m spacy download en_core_web_sm")
            raise
        
        # Define sentiment keywords
        self.positive_words = {
            'excellent', 'amazing', 'great', 'good', 'awesome', 'fantastic', 
            'perfect', 'love', 'wonderful', 'outstanding', 'brilliant', 'superb',
            'recommend', 'satisfied', 'happy', 'pleased', 'impressive'
        }
        
        self.negative_words = {
            'terrible', 'awful', 'horrible', 'bad', 'poor', 'disappointing',
            'waste', 'useless', 'broken', 'defective', 'slow', 'expensive',
            'cheap', 'fake', 'scam', 'avoid', 'regret'
        }
    
    def generate_sample_reviews(self):
        """Generate sample Amazon reviews for demonstration"""
        sample_reviews = [
            "I bought the Apple iPhone 13 from Amazon and it's absolutely amazing! The camera quality is excellent and battery life is outstanding.",
            "The Samsung Galaxy phone I received was defective. The screen had dead pixels and the battery drains too fast. Very disappointing.",
            "Microsoft Surface Pro is a fantastic device for work. The performance is brilliant and the design is sleek.",
            "This Sony headphones are terrible. The sound quality is poor and they broke after one week. Complete waste of money.",
            "Google Pixel has an awesome camera and the Android experience is perfect. Highly recommend this phone!",
            "The Dell laptop I ordered arrived with a broken keyboard. Customer service from Dell was horrible.",
            "Bose speakers provide wonderful sound quality. The bass is impressive and setup was easy.",
            "This HP printer is useless. It jams constantly and the HP ink is too expensive.",
            "Lenovo ThinkPad is a great business laptop. The keyboard is comfortable and performance is superb.",
            "I regret buying this Acer computer. It's slow and the Acer support is awful."
        ]
        
        return sample_reviews
    
    def perform_ner_analysis(self, text):
        """Perform Named Entity Recognition on text"""
        doc = self.nlp(text)
        
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities, doc
    
    def analyze_sentiment(self, text):
        """Perform rule-based sentiment analysis"""
        doc = self.nlp(text.lower())
        
        positive_count = 0
        negative_count = 0
        
        for token in doc:
            if token.text in self.positive_words:
                positive_count += 1
            elif token.text in self.negative_words:
                negative_count += 1
        
        # Determine overall sentiment
        if positive_count > negative_count:
            sentiment = "POSITIVE"
            confidence = positive_count / (positive_count + negative_count + 1)
        elif negative_count > positive_count:
            sentiment = "NEGATIVE"
            confidence = negative_count / (positive_count + negative_count + 1)
        else:
            sentiment = "NEUTRAL"
            confidence = 0.5
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_words': positive_count,
            'negative_words': negative_count
        }
    
    def visualize_ner(self, doc, title="Named Entity Recognition"):
        """Visualize NER results"""
        colors = {
            "ORG": "linear-gradient(90deg, #aa9cfc, #fc9ce7)",
            "PRODUCT": "linear-gradient(90deg, #ff6b6b, #ffa726)",
            "GPE": "linear-gradient(90deg, #4ecdc4, #44a08d)",
            "PERSON": "linear-gradient(90deg, #ffd93d, #ff9c3d)"
        }
        
        options = {"colors": colors}
        html = displacy.render(doc, style="ent", options=options, page=True)
        return html
    
    def run_complete_analysis(self):
        """Run complete NLP analysis on sample reviews"""
        print("🔍 Performing NLP Analysis on Amazon Reviews...")
        
        # Generate sample reviews
        reviews = self.generate_sample_reviews()
        
        results = []
        
        for i, review in enumerate(reviews, 1):
            print(f"\n📝 Review {i}: {review}")
            
            # Perform NER
            entities, doc = self.perform_ner_analysis(review)
            
            # Analyze sentiment
            sentiment_result = self.analyze_sentiment(review)
            
            # Store results
            result = {
                'review_id': i,
                'review_text': review,
                'entities': entities,
                'sentiment': sentiment_result['sentiment'],
                'confidence': sentiment_result['confidence'],
                'positive_words': sentiment_result['positive_words'],
                'negative_words': sentiment_result['negative_words']
            }
            results.append(result)
            
            # Print analysis results
            print(f"  🏷️  Entities found: {len(entities)}")
            for entity in entities:
                print(f"     - {entity['text']} ({entity['label']})")
            
            print(f"  😊 Sentiment: {sentiment_result['sentiment']} "
                 