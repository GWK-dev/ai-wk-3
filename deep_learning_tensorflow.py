# practical_implementation/deep_learning_tensorflow.py
"""
Deep Learning with TensorFlow
MNIST Handwritten Digits Classification using CNN
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class MNISTClassifier:
    def __init__(self):
        self.model = None
        self.history = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess MNIST dataset"""
        print("📊 Loading MNIST Dataset...")
        
        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Normalize pixel values to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape data for CNN (add channel dimension)
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        
        # Convert labels to categorical one-hot encoding
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        print(f"Training data shape: {x_train.shape}")
        print(f"Test data shape: {x_test.shape}")
        
        return (x_train, y_train), (x_test, y_test)
    
    def build_cnn_model(self):
        """Build CNN model architecture"""
        print("🏗️ Building CNN Model Architecture...")
        
        model = keras.Sequential([
            # First convolutional layer
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            
            # Second convolutional layer
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Third convolutional layer
            layers.Conv2D(64, (3, 3), activation='relu'),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model architecture:")
        model.summary()
        
        self.model = model
        return model
    
    def train_model(self, x_train, y_train, x_test, y_test, epochs=10):
        """Train the CNN model"""
        print("🚀 Training CNN Model...")
        
        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
        ]
        
        # Train the model
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=128,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Model training completed!")
    
    def evaluate_model(self, x_test, y_test):
        """Evaluate model performance"""
        print("📈 Evaluating Model Performance...")
        
        # Get predictions
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        y_pred = self.model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
        print(f"📉 Test Loss: {test_loss:.4f}")
        
        # Check if accuracy exceeds 95%
        if test_accuracy > 0.95:
            print("🎉 Target achieved: Accuracy > 95%!")
        else:
            print("⚠️ Target not achieved: Accuracy < 95%")
        
        return test_accuracy, test_loss, y_pred_classes, y_true_classes
    
    def visualize_results(self, x_test, y_true_classes, y_pred_classes):
        """Visualize model predictions and performance"""
        print("🎨 Generating Visualizations...")
        
        # Plot training history
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy plot
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss plot
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # Sample predictions
        axes[1, 1].axis('off')
        self._plot_sample_predictions(x_test, y_true_classes, y_pred_classes)
        
        plt.tight_layout()
        plt.savefig('mnist_cnn_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_sample_predictions(self, x_test, y_true, y_pred):
        """Plot sample predictions"""
        # Find some correct and incorrect predictions
        correct_indices = np.where(y_true == y_pred)[0]
        incorrect_indices = np.where(y_true != y_pred)[0]
        
        # Select samples to display
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        
        for i in range(5):
            # Correct predictions
            if i < len(correct_indices):
                idx = correct_indices[i]
                axes[0, i].imshow(x_test[idx].reshape(28, 28), cmap='gray')
                axes[0, i].set_title(f'True: {y_true[idx]}, Pred: {y_pred[idx]}')
                axes[0, i].axis('off')
            
            # Incorrect predictions
            if i < len(incorrect_indices):
                idx = incorrect_indices[i]
                axes[1, i].imshow(x_test[idx].reshape(28, 28), cmap='gray')
                axes[1, i].set_title(f'True: {y_true[idx]}, Pred: {y_pred[idx]}', color='red')
                axes[1, i].axis('off')
        
        plt.suptitle('Sample Predictions (Top: Correct, Bottom: Incorrect)')
        plt.tight_layout()
    
    def run_complete_analysis(self, epochs=10):
        """Run the complete deep learning pipeline"""
        # Load and preprocess data
        (x_train, y_train), (x_test, y_test) = self.load_and_preprocess_data()
        
        # Build model
        self.build_cnn_model()
        
        # Train model
        self.train_model(x_train, y_train, x_test, y_test, epochs)
        
        # Evaluate model
        test_accuracy, test_loss, y_pred_classes, y_true_classes = self.evaluate_model(x_test, y_test)
        
        # Visualize results
        self.visualize_results(x_test, y_true_classes, y_pred_classes)
        
        return test_accuracy, test_loss

if __name__ == "__main__":
    # Initialize and run the MNIST classifier
    mnist_classifier = MNISTClassifier()
    accuracy, loss = mnist_classifier.run_complete_analysis(epochs=10)
    
    print("\n" + "="*50)
    print("🎉 DEEP LEARNING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Final Results:")
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Test Loss: {loss:.4f}")