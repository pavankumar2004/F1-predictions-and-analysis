import os
import numpy as np
from PIL import Image
import logging
import pathlib
import shutil
from typing import Tuple, Optional, Union, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

class CustomLogisticRegression:
    """
    Custom implementation of multinomial logistic regression for F1 car classification.
    Includes image preprocessing, model training, prediction, and model persistence.
    """
    
    def __init__(self, 
                 model_path: str = 'models/f1_car_classifier.pkl',
                 learning_rate: float = 0.01,
                 max_iterations: int = 1000,
                 tol: float = 1e-4,
                 img_size: Tuple[int, int] = (224, 224)) -> None:
        """
        Initialize the logistic regression classifier.
        
        Args:
            model_path (str): Path to save/load the model
            learning_rate (float): Learning rate for gradient descent
            max_iterations (int): Maximum number of training iterations
            tol (float): Convergence tolerance
            img_size (tuple): Target image size for processing
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.img_height, self.img_width = img_size
        self.class_names = [
            'AlphaTauri', 'Ferrari', 'McLaren', 'Mercedes', 
            'Racing Point', 'Red Bull Racing', 'Renault', 'Williams'
        ]
        self.model_path = model_path
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        # Training parameters
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tol = tol
        
        # Model parameters
        self.weights = None
        self.bias = None
        self.n_classes = len(self.class_names)
        
        self.logger.info("Logistic regression classifier initialized")

    def preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            np.ndarray: Preprocessed image as flattened array
        """
        try:
            # Handle numpy array input
            if isinstance(image, np.ndarray):
                if image.dtype in [np.float32, np.float64]:
                    image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
            
            # Ensure RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize and standardize
            image = image.resize((self.img_width, self.img_height))
            img_array = np.array(image).flatten()
            img_array = img_array.reshape(1, -1)
            
            # Normalize pixel values
            img_array = img_array / 255.0
            
            return img_array
            
        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {str(e)}")
            raise

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Apply sigmoid activation with numerical stability."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def softmax(self, z: np.ndarray) -> np.ndarray:
        """Apply softmax activation with numerical stability."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def one_hot_encode(self, y: np.ndarray) -> np.ndarray:
        """Convert integer labels to one-hot encoded format."""
        n_samples = len(y)
        encoded = np.zeros((n_samples, self.n_classes))
        encoded[np.arange(n_samples), y] = 1
        return encoded

    def initialize_parameters(self, n_features: int) -> None:
        """Initialize model weights and biases."""
        self.weights = np.random.randn(n_features, self.n_classes) * 0.01
        self.bias = np.zeros((1, self.n_classes))

    def compute_cost(self, X: np.ndarray, y_encoded: np.ndarray) -> float:
        """
        Compute cross-entropy loss.
        
        Args:
            X: Input features
            y_encoded: One-hot encoded labels
            
        Returns:
            float: Cross-entropy loss
        """
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        predictions = self.softmax(z)
        
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        cost = -np.mean(np.sum(y_encoded * np.log(predictions + epsilon), axis=1))
        return cost

    def compute_gradients(self, X: np.ndarray, y_encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients for weights and bias.
        
        Args:
            X: Input features
            y_encoded: One-hot encoded labels
            
        Returns:
            tuple: Weight gradients and bias gradients
        """
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        predictions = self.softmax(z)
        
        dz = predictions - y_encoded
        dw = (1/m) * np.dot(X.T, dz)
        db = (1/m) * np.sum(dz, axis=0, keepdims=True)
        
        return dw, db

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, List[float]]:
        """
        Train the model using gradient descent.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            dict: Training history (costs and accuracies)
        """
        try:
            n_samples, n_features = X.shape
            if self.weights is None:
                self.initialize_parameters(n_features)
            
            y_encoded = self.one_hot_encode(y)
            prev_cost = float('inf')
            
            history = {'costs': [], 'accuracies': []}
            
            for iteration in range(self.max_iterations):
                # Forward pass
                current_cost = self.compute_cost(X, y_encoded)
                
                # Compute gradients
                dw, db = self.compute_gradients(X, y_encoded)
                
                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
                # Track metrics
                if iteration % 100 == 0:
                    predictions = self.predict_proba(X)
                    accuracy = accuracy_score(y, np.argmax(predictions, axis=1))
                    history['costs'].append(current_cost)
                    history['accuracies'].append(accuracy)
                    self.logger.info(f"Iteration {iteration}, Cost: {current_cost:.6f}, Accuracy: {accuracy:.4f}")
                
                # Check convergence
                if abs(prev_cost - current_cost) < self.tol:
                    self.logger.info(f"Converged at iteration {iteration}")
                    break
                    
                prev_cost = current_cost
            
            return history
                
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Class probabilities
        """
        try:
            z = np.dot(X, self.weights) + self.bias
            return self.softmax(z)
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise

    def predict(self, image: Union[Image.Image, np.ndarray]) -> Tuple[str, float, np.ndarray]:
        """
        Predict class for an image.
        
        Args:
            image: Input image
            
        Returns:
            tuple: Predicted class, confidence score, and all class probabilities
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Get probabilities
            probabilities = self.predict_proba(processed_image)
            
            # Get prediction and confidence
            predicted_idx = np.argmax(probabilities[0])
            predicted_class = self.class_names[predicted_idx]
            confidence = float(probabilities[0][predicted_idx])
            
            self.logger.info(f"Predicted {predicted_class} with {confidence:.2f} confidence")
            return predicted_class, confidence, probabilities[0]
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise

    def validate_dataset(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Validate and prepare the dataset.
        
        Args:
            data_dir: Path to dataset directory
            
        Returns:
            tuple: Features, labels, count of valid files, count of invalid files
        """
        try:
            data_dir = pathlib.Path(data_dir)
            if not data_dir.exists():
                raise ValueError(f"Dataset directory {data_dir} does not exist")

            valid_files = 0
            invalid_files = 0
            X, y = [], []

            for class_idx, class_name in enumerate(self.class_names):
                class_dir = data_dir / class_name
                if not class_dir.exists():
                    self.logger.warning(f"Class directory {class_name} not found")
                    continue

                for img_path in class_dir.glob('*'):
                    if img_path.suffix.lower() not in self.valid_extensions:
                        self.logger.warning(f"Invalid file type: {img_path}")
                        invalid_files += 1
                        continue

                    try:
                        with Image.open(img_path) as img:
                            processed_img = self.preprocess_image(img)
                            X.append(processed_img.flatten())
                            y.append(class_idx)
                            valid_files += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to process {img_path}: {str(e)}")
                        invalid_files += 1
                        continue

            self.logger.info(f"Dataset validation complete. Valid: {valid_files}, Invalid: {invalid_files}")
            return np.array(X), np.array(y), valid_files, invalid_files

        except Exception as e:
            self.logger.error(f"Dataset validation failed: {str(e)}")
            raise

    def train(self, data_dir: str, test_size: float = 0.2) -> float:
        """
        Train the model on a dataset.
        
        Args:
            data_dir: Path to dataset directory
            test_size: Proportion of data to use for testing
            
        Returns:
            float: Model accuracy on test set
        """
        try:
            # Validate and load dataset
            X, y, valid_files, invalid_files = self.validate_dataset(data_dir)
            
            if valid_files == 0:
                raise ValueError("No valid images found in dataset")
            
            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            # Train model
            self.logger.info(f"Starting training with {valid_files} images")
            history = self.fit(X_train, y_train)
            
            # Evaluate model
            predictions = self.predict_proba(X_test)
            predictions = np.argmax(predictions, axis=1)
            accuracy = accuracy_score(y_test, predictions)

            # Save model
            self.save_model()
            
            self.logger.info(f"Training complete. Test accuracy: {accuracy:.4f}")
            return accuracy

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

    def save_model(self) -> None:
        """Save model parameters to file."""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'weights': self.weights,
                    'bias': self.bias,
                    'class_names': self.class_names,
                    'img_size': (self.img_height, self.img_width)
                }, f)
            self.logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            raise

    def load_model(self) -> bool:
        """
        Load model parameters from file.
        
        Returns:
            bool: True if model was loaded successfully
        """
        try:
            if not os.path.exists(self.model_path):
                self.logger.warning(f"No saved model found at {self.model_path}")
                return False
                
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.weights = data['weights']
                self.bias = data['bias']
                self.class_names = data.get('class_names', self.class_names)
                self.img_height, self.img_width = data.get('img_size', (self.img_height, self.img_width))
                
            self.logger.info(f"Model loaded from {self.model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise