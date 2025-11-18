# ethics_optimization.md
# AI Tools Assignment - Ethics & Optimization

## Part 3: Ethics & Optimization

### 1. Ethical Considerations

**Potential Biases in MNIST Model:**

**Data Bias:**
- MNIST dataset primarily contains handwritten digits from Western education systems
- Underrepresentation of diverse handwriting styles from different cultures and age groups
- Potential bias against unusual digit formations or non-standard writing styles

**Mitigation Strategies with TensorFlow Fairness Indicators:**
- Use TF Fairness Indicators to analyze performance across different demographic groups
- Implement data augmentation to include diverse handwriting styles
- Regular bias audits and model performance monitoring
- Ensure balanced representation in training data

**Potential Biases in Amazon Reviews Model:**

**Language and Cultural Bias:**
- Training data may overrepresent certain demographics or geographic regions
- Sentiment analysis might misinterpret cultural nuances and sarcasm
- NER could be less accurate for non-Western brand names and products

**Mitigation with spaCy's Rule-Based Systems:**
- Create custom rules to handle domain-specific terminology
- Implement cultural sensitivity checks in sentiment analysis
- Use spaCy's pattern matching for brand-specific entity recognition
- Regular validation with diverse user groups

### 2. Troubleshooting Challenge

**Buggy TensorFlow Code Analysis and Fix:**

**Original Buggy Code Issues Identified:**
1. Dimension mismatches in layer connections
2. Incorrect loss function for multi-class classification
3. Missing activation functions in dense layers
4. Improper data reshaping for CNN input

**Fixed Code Implementation:**
```python
# Corrected TensorFlow CNN implementation
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),  # Added activation
    layers.Dense(10, activation='softmax')  # Correct output activation
])

# Correct compilation
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Correct loss for multi-class
    metrics=['accuracy']
)