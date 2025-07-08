"""
Failure Detection ML Models using LSTM/GRU
Trains deep learning models on TUNA-cleaned time series data to detect system failures
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization, Input, Concatenate, Attention
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.regularizers import l2
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    TF_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow not available. Install with: pip install tensorflow>=2.8.0")
    TF_AVAILABLE = False

import matplotlib.pyplot as plt
import seaborn as sns


class FailureDetectionDataset:
    """Handles data preparation for failure detection"""
    
    def __init__(self, sequence_length: int = 30, prediction_horizon: int = 5):
        """
        Args:
            sequence_length: Number of time steps to look back
            prediction_horizon: How many steps ahead to predict failure
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scalers = {}
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
    def load_cleaned_datasets(self, data_path: str) -> Dict[str, pd.DataFrame]:
        """Load TUNA-cleaned datasets from exported CSV files"""
        print(f"📂 Loading cleaned datasets from {data_path}...")
        
        data_path = Path(data_path)
        datasets = {}
        
        for metric_dir in data_path.iterdir():
            if not metric_dir.is_dir():
                continue
                
            metric_name = metric_dir.name
            print(f"  📊 Loading {metric_name}...")
            
            metric_data = {}
            for csv_file in metric_dir.glob("*.csv"):
                exp_name = csv_file.stem
                df = pd.read_csv(csv_file)
                metric_data[exp_name] = df
                print(f"    ✅ {exp_name}: {len(df)} points")
            
            datasets[metric_name] = metric_data
        
        print(f" Loaded {len(datasets)} metrics")
        return datasets
    
    def create_failure_labels(self, experiments: Dict[str, pd.DataFrame], 
                            failure_experiments: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Create failure labels based on experiment types
        
        Args:
            experiments: Dictionary of {experiment_name: dataframe}
            failure_experiments: List of experiment names considered as failures
                                If None, auto-detect stress experiments
        
        Returns:
            Dictionary of {experiment_name: failure_labels}
        """
        if failure_experiments is None:
            # Auto-detect failure experiments (non-baseline)
            failure_experiments = [name for name in experiments.keys() if name != 'baseline']
        
        labels = {}
        
        for exp_name, df in experiments.items():
            if exp_name == 'baseline':
                # Baseline: no failure (label = 0)
                labels[exp_name] = np.zeros(len(df), dtype=int)
            elif exp_name in failure_experiments:
                # Failure experiments: failure during stress period
                failure_labels = np.zeros(len(df), dtype=int)
                
                # Assume stress period starts at minute 30 and lasts 50 minutes
                stress_start = 30
                stress_duration = 50
                
                # Mark failure period and prediction horizon before it
                for i, minute in enumerate(df['Minutes']):
                    if stress_start - self.prediction_horizon <= minute <= stress_start + stress_duration:
                        failure_labels[i] = 1
                
                labels[exp_name] = failure_labels
            else:
                # Other experiments: no failure
                labels[exp_name] = np.zeros(len(df), dtype=int)
        
        return labels
    
    def extract_features(self, datasets: Dict[str, Dict[str, pd.DataFrame]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract features from multiple metrics and experiments
        
        Args:
            datasets: Dictionary of {metric_name: {experiment_name: dataframe}}
        
        Returns:
            Tuple of (features, labels, feature_names)
        """
        print("🔧 Extracting features for failure detection...")
        
        all_features = []
        all_labels = []
        feature_names = []
        
        # First pass: determine feature structure
        sample_metric = list(datasets.keys())[0]
        sample_exp = list(datasets[sample_metric].keys())[0]
        sample_df = datasets[sample_metric][sample_exp]
        
        # Extract metric columns (excluding metadata)
        exclude_cols = ['Time', 'Minutes', 'source', 'tuna_processed', 'outliers_detected']
        
        for metric_name in sorted(datasets.keys()):
            sample_df = list(datasets[metric_name].values())[0]
            metric_cols = [col for col in sample_df.columns if col not in exclude_cols]
            
            for col in metric_cols:
                feature_names.append(f"{metric_name}_{col}")
        
        self.feature_names = feature_names
        print(f"  📏 Feature dimensions: {len(feature_names)} features per timestep")
        
        # Process each experiment
        for metric_name, experiments in datasets.items():
            print(f"  📊 Processing {metric_name}...")
            
            # Create failure labels for this metric
            failure_labels = self.create_failure_labels(experiments)
            
            for exp_name, df in experiments.items():
                print(f"    🔄 Processing {exp_name}...")
                
                # Extract metric values
                metric_cols = [col for col in df.columns if col not in exclude_cols]
                values = df[metric_cols].values
                
                # Scale features
                scaler_key = f"{metric_name}_{exp_name}"
                if scaler_key not in self.scalers:
                    self.scalers[scaler_key] = StandardScaler()
                    scaled_values = self.scalers[scaler_key].fit_transform(values)
                else:
                    scaled_values = self.scalers[scaler_key].transform(values)
                
                # Get labels for this experiment
                labels = failure_labels[exp_name]
                
                # Create sequences
                exp_features, exp_labels = self._create_sequences(scaled_values, labels)
                
                if len(exp_features) > 0:
                    all_features.append(exp_features)
                    all_labels.append(exp_labels)
        
        if not all_features:
            raise ValueError("No valid sequences created")
        
        # Combine all features
        X = np.vstack(all_features)
        y = np.hstack(all_labels)
        
        print(f"✅ Created dataset: {X.shape[0]} sequences, {X.shape[1]} timesteps, {X.shape[2]} features")
        print(f"📈 Label distribution: {np.bincount(y)}")
        
        return X, y, feature_names
    
    def _create_sequences(self, values: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM/GRU training"""
        sequences = []
        sequence_labels = []
        
        for i in range(self.sequence_length, len(values)):
            # Input sequence
            seq = values[i-self.sequence_length:i]
            
            # Label at current timestep
            label = labels[i]
            
            sequences.append(seq)
            sequence_labels.append(label)
        
        return np.array(sequences), np.array(sequence_labels)
    
    def prepare_multi_metric_features(self, datasets: Dict[str, Dict[str, pd.DataFrame]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features combining all metrics into unified sequences
        
        Returns:
            Tuple of (X, y) where X has shape (samples, timesteps, total_features)
        """
        print("🔧 Preparing multi-metric features...")
        
        # Get all metric names sorted for consistency
        metric_names = sorted(datasets.keys())
        
        # Find common experiments across all metrics
        common_experiments = set(datasets[metric_names[0]].keys())
        for metric_name in metric_names[1:]:
            common_experiments &= set(datasets[metric_name].keys())
        
        common_experiments = sorted(list(common_experiments))
        print(f"  🎯 Common experiments: {common_experiments}")
        
        all_sequences = []
        all_labels = []
        
        for exp_name in common_experiments:
            print(f"  🔄 Processing experiment: {exp_name}...")
            
            # Collect all metric data for this experiment
            exp_features = []
            exp_length = None
            
            for metric_name in metric_names:
                df = datasets[metric_name][exp_name]
                
                # Extract feature columns
                exclude_cols = ['Time', 'Minutes', 'source', 'tuna_processed', 'outliers_detected']
                metric_cols = [col for col in df.columns if col not in exclude_cols]
                values = df[metric_cols].values
                
                # Scale features
                scaler_key = f"{metric_name}_{exp_name}"
                if scaler_key not in self.scalers:
                    self.scalers[scaler_key] = StandardScaler()
                    scaled_values = self.scalers[scaler_key].fit_transform(values)
                else:
                    scaled_values = self.scalers[scaler_key].transform(values)
                
                exp_features.append(scaled_values)
                
                if exp_length is None:
                    exp_length = len(scaled_values)
                elif len(scaled_values) != exp_length:
                    print(f"    ⚠️ Length mismatch in {metric_name}: {len(scaled_values)} vs {exp_length}")
                    # Truncate to minimum length
                    min_length = min(exp_length, len(scaled_values))
                    exp_features = [feat[:min_length] for feat in exp_features]
                    exp_length = min_length
            
            # Combine all metrics for this experiment
            combined_features = np.hstack(exp_features)  # Shape: (timesteps, total_features)
            
            # Create failure labels for this experiment
            failure_labels = self.create_failure_labels({exp_name: datasets[metric_names[0]][exp_name]})
            labels = failure_labels[exp_name][:exp_length]
            
            # Create sequences
            sequences, seq_labels = self._create_sequences(combined_features, labels)
            
            if len(sequences) > 0:
                all_sequences.append(sequences)
                all_labels.append(seq_labels)
                print(f"    ✅ Created {len(sequences)} sequences")
        
        # Combine all experiments
        X = np.vstack(all_sequences)
        y = np.hstack(all_labels)
        
        print(f"✅ Final dataset: {X.shape[0]} sequences, {X.shape[1]} timesteps, {X.shape[2]} features")
        return X, y


class FailureDetectionModel:
    """LSTM/GRU models for failure detection"""
    
    def __init__(self, model_type: str = 'lstm', sequence_length: int = 30, 
                 n_features: int = None, hidden_units: int = 64):
        """
        Args:
            model_type: 'lstm', 'gru', or 'attention'
            sequence_length: Length of input sequences
            n_features: Number of features per timestep
            hidden_units: Number of hidden units in RNN layers
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for deep learning models")
            
        self.model_type = model_type.lower()
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.hidden_units = hidden_units
        self.model = None
        self.history = None
        
    def build_model(self, n_features: int) -> tf.keras.Model:
        """Build the failure detection model"""
        self.n_features = n_features
        
        if self.model_type == 'lstm':
            return self._build_lstm_model()
        elif self.model_type == 'gru':
            return self._build_gru_model()
        elif self.model_type == 'attention':
            return self._build_attention_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _build_lstm_model(self) -> tf.keras.Model:
        """Build LSTM-based model"""
        model = Sequential([
            LSTM(self.hidden_units, return_sequences=True, 
                 input_shape=(self.sequence_length, self.n_features),
                 kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            BatchNormalization(),
            
            LSTM(self.hidden_units // 2, return_sequences=False,
                 kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            BatchNormalization(),
            
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def _build_gru_model(self) -> tf.keras.Model:
        """Build GRU-based model"""
        model = Sequential([
            GRU(self.hidden_units, return_sequences=True,
                input_shape=(self.sequence_length, self.n_features),
                kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            BatchNormalization(),
            
            GRU(self.hidden_units // 2, return_sequences=False,
                kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            BatchNormalization(),
            
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def _build_attention_model(self) -> tf.keras.Model:
        """Build attention-based model"""
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # LSTM with return sequences for attention
        lstm_out = LSTM(self.hidden_units, return_sequences=True, 
                       kernel_regularizer=l2(0.01))(inputs)
        lstm_out = Dropout(0.3)(lstm_out)
        lstm_out = BatchNormalization()(lstm_out)
        
        # Attention mechanism (simplified)
        attention_weights = Dense(1, activation='tanh')(lstm_out)
        attention_weights = tf.nn.softmax(attention_weights, axis=1)
        context_vector = tf.reduce_sum(lstm_out * attention_weights, axis=1)
        
        # Final layers
        dense = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(context_vector)
        dense = Dropout(0.2)(dense)
        outputs = Dense(1, activation='sigmoid')(dense)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 100, batch_size: int = 32, verbose: int = 1) -> Dict[str, Any]:
        """
        Train the failure detection model
        
        Args:
            X_train: Training features (samples, timesteps, features)
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
            
        Returns:
            Training history and metrics
        """
        print(f"🚀 Training {self.model_type.upper()} model...")
        
        # Build model if not already built
        if self.model is None:
            self.model = self.build_model(X_train.shape[2])
        
        print(f"📊 Model architecture:")
        self.model.summary()
        
        # Prepare validation data
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Evaluate on validation set
        val_loss, val_acc, val_prec, val_rec = self.model.evaluate(X_val, y_val, verbose=0)
        val_f1 = 2 * (val_prec * val_rec) / (val_prec + val_rec) if (val_prec + val_rec) > 0 else 0
        
        # Predictions for ROC-AUC
        y_pred_proba = self.model.predict(X_val, verbose=0).flatten()
        val_auc = roc_auc_score(y_val, y_pred_proba)
        
        training_results = {
            'final_epoch': len(self.history.history['loss']),
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_precision': val_prec,
            'val_recall': val_rec,
            'val_f1': val_f1,
            'val_auc': val_auc,
            'history': self.history.history
        }
        
        print(f"✅ Training completed!")
        print(f"📈 Validation Results:")
        print(f"  - Accuracy: {val_acc:.4f}")
        print(f"  - Precision: {val_prec:.4f}")
        print(f"  - Recall: {val_rec:.4f}")
        print(f"  - F1-Score: {val_f1:.4f}")
        print(f"  - AUC: {val_auc:.4f}")
        
        return training_results
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data
        
        Args:
            X: Input features
            threshold: Classification threshold
            
        Returns:
            Tuple of (binary_predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Get probability predictions
        y_proba = self.model.predict(X, verbose=0).flatten()
        
        # Convert to binary predictions
        y_pred = (y_proba >= threshold).astype(int)
        
        return y_pred, y_proba
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Comprehensive evaluation of the model"""
        print("📊 Evaluating model performance...")
        
        y_pred, y_proba = self.predict(X_test)
        
        # Classification metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        print(f"🎯 Test Results:")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - Precision: {precision:.4f}")
        print(f"  - Recall: {recall:.4f}")
        print(f"  - F1-Score: {f1:.4f}")
        print(f"  - AUC: {auc:.4f}")
        
        return results
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a saved model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"📂 Model loaded from {filepath}")


class FailureDetectionPipeline:
    """Complete pipeline for failure detection using TUNA-cleaned data"""
    
    def __init__(self, sequence_length: int = 30, prediction_horizon: int = 5):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.dataset = FailureDetectionDataset(sequence_length, prediction_horizon)
        self.models = {}
        self.results = {}
        
    def prepare_data(self, cleaned_data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and prepare data for training"""
        print("🔧 Preparing failure detection dataset...")
        
        # Load TUNA-cleaned datasets
        datasets = self.dataset.load_cleaned_datasets(cleaned_data_path)
        
        # Extract features combining all metrics
        X, y = self.dataset.prepare_multi_metric_features(datasets)
        
        return X, y
    
    def train_models(self, X: np.ndarray, y: np.ndarray, 
                    model_types: List[str] = ['lstm', 'gru', 'attention'],
                    test_size: float = 0.2) -> Dict[str, Dict[str, Any]]:
        """
        Train multiple model types and compare performance
        
        Args:
            X: Input features
            y: Labels  
            model_types: List of model types to train
            test_size: Fraction of data for testing
            
        Returns:
            Dictionary of training results for each model
        """
        print(f"🚀 Training {len(model_types)} model types...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"📊 Data split:")
        print(f"  - Training: {X_train.shape[0]} samples")
        print(f"  - Testing: {X_test.shape[0]} samples")
        print(f"  - Features: {X.shape[2]}")
        
        # Train each model type
        for model_type in model_types:
            print(f"\n🤖 Training {model_type.upper()} model...")
            
            # Create and train model
            model = FailureDetectionModel(
                model_type=model_type,
                sequence_length=self.sequence_length,
                hidden_units=64
            )
            
            # Train model
            training_results = model.train(X_train, y_train, epochs=100, batch_size=32)
            
            # Evaluate model
            test_results = model.evaluate(X_test, y_test)
            
            # Store model and results
            self.models[model_type] = model
            self.results[model_type] = {
                'training': training_results,
                'testing': test_results,
                'X_test': X_test,
                'y_test': y_test
            }
        
        return self.results
    
    def compare_models(self) -> pd.DataFrame:
        """Compare performance of all trained models"""
        print("📊 Comparing model performance...")
        
        comparison_data = []
        
        for model_type, results in self.results.items():
            test_results = results['testing']
            
            comparison_data.append({
                'Model': model_type.upper(),
                'Accuracy': test_results['accuracy'],
                'Precision': test_results['precision'],
                'Recall': test_results['recall'],
                'F1-Score': test_results['f1_score'],
                'AUC': test_results['auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.round(4)
        
        print("\n📈 Model Comparison:")
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def plot_results(self, save_path: str = None) -> None:
        """Plot training history and evaluation results"""
        n_models = len(self.models)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
        
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        for i, (model_type, results) in enumerate(self.results.items()):
            history = results['training']['history']
            
            # Plot training history
            axes[0, i].plot(history['loss'], label='Training Loss')
            axes[0, i].plot(history['val_loss'], label='Validation Loss')
            axes[0, i].set_title(f'{model_type.upper()} - Training History')
            axes[0, i].set_xlabel('Epoch')
            axes[0, i].set_ylabel('Loss')
            axes[0, i].legend()
            axes[0, i].grid(True)
            
            # Plot confusion matrix
            cm = results['testing']['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, i], cmap='Blues')
            axes[1, i].set_title(f'{model_type.upper()} - Confusion Matrix')
            axes[1, i].set_xlabel('Predicted')
            axes[1, i].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Results plot saved to {save_path}")
        
        plt.show()
    
    def save_best_model(self, save_path: str, metric: str = 'f1_score') -> str:
        """Save the best performing model based on specified metric"""
        if not self.results:
            raise ValueError("No models have been trained yet")
        
        # Find best model
        best_score = -1
        best_model_type = None
        
        for model_type, results in self.results.items():
            score = results['testing'][metric]
            if score > best_score:
                best_score = score
                best_model_type = model_type
        
        # Save best model
        best_model = self.models[best_model_type]
        model_path = f"{save_path}/best_failure_detection_model_{best_model_type}.h5"
        best_model.save_model(model_path)
        
        print(f"🏆 Best model ({best_model_type.upper()}) saved with {metric}={best_score:.4f}")
        return model_path


def train_failure_detection_models(cleaned_data_path: str, output_path: str = "failure_models",
                                 sequence_length: int = 30, model_types: List[str] = None) -> Dict[str, Any]:
    """
    Convenience function to train failure detection models on TUNA-cleaned data
    
    Args:
        cleaned_data_path: Path to TUNA-cleaned datasets
        output_path: Path to save models and results
        sequence_length: Length of input sequences
        model_types: List of model types to train
        
    Returns:
        Dictionary containing all results
    """
    if model_types is None:
        model_types = ['lstm', 'gru']
    
    # Create output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline
    pipeline = FailureDetectionPipeline(sequence_length=sequence_length)
    
    # Prepare data
    X, y = pipeline.prepare_data(cleaned_data_path)
    
    # Train models
    results = pipeline.train_models(X, y, model_types=model_types)
    
    # Compare models
    comparison_df = pipeline.compare_models()
    
    # Save comparison results
    comparison_df.to_csv(f"{output_path}/model_comparison.csv", index=False)
    
    # Plot results
    pipeline.plot_results(f"{output_path}/training_results.png")
    
    # Save best model
    best_model_path = pipeline.save_best_model(output_path)
    
    return {
        'pipeline': pipeline,
        'results': results,
        'comparison': comparison_df,
        'best_model_path': best_model_path,
        'data_shape': X.shape
    }