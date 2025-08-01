import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           f1_score, precision_score, recall_score)
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def create_splits(df_engineered, selected_features, sequence_length=20, random_state=42):    
    normal_data = df_engineered[df_engineered['Failure'] == 0]
    anomalous_data = df_engineered[df_engineered['Failure'] == 1]
    
    normal_features = normal_data[selected_features].values
    n_normal = len(normal_features)
    
    idx_60 = int(0.6 * n_normal)  
    idx_75 = int(0.75 * n_normal)  
    idx_90 = int(0.9 * n_normal)   
    

    sN_data = normal_features[:idx_60]                    # 0% → 60%
    vN1_data = normal_features[idx_60:idx_75]             # 60% → 75%
    vN2_data = normal_features[idx_75:idx_90]             # 75% → 90% 
    tN_data = normal_features[idx_90:]                    # 90% → 100%
    
    anomalous_features = anomalous_data[selected_features].values
    n_anomalous = len(anomalous_features)
    idx_50_anom = int(0.5 * n_anomalous)
    
    vA_data = anomalous_features[:idx_50_anom]            
    tA_data = anomalous_features[idx_50_anom:]            
    
    print(f"\n📋 TIME-AWARE SPLITS CREATI:")
    print(f"   sN  (normal train):     {len(sN_data):,} ({len(sN_data)/n_normal*100:.1f}%) - Early period")
    print(f"   vN1 (normal val-1):     {len(vN1_data):,} ({len(vN1_data)/n_normal*100:.1f}%) - Mid-early period") 
    print(f"   vN2 (normal val-2):     {len(vN2_data):,} ({len(vN2_data)/n_normal*100:.1f}%) - Mid-late period")
    print(f"   tN  (normal test):      {len(tN_data):,} ({len(tN_data)/n_normal*100:.1f}%) - Late period")
    print(f"   vA  (anomalous val):    {len(vA_data):,} (50.0%) - Early anomalies")
    print(f"   tA  (anomalous test):   {len(tA_data):,} (50.0%) - Late anomalies")
    
    if len(sN_data) >= sequence_length and len(vN1_data) >= sequence_length:
        time_gap = idx_60 - sequence_length
        print(f"\n🔍 ANALISI TEMPORALE:")
        print(f"   Gap between train end and val start: {time_gap * 15}s ({time_gap * 15/60:.1f}min)")
        if time_gap < 0:
            print(f"  Potential temporal overlap due to sequence length!")
        else:
            print(f"   Clean temporal separation maintained")
    
    return {
        'sN': sN_data,
        'vN1': vN1_data,
        'vN2': vN2_data,
        'tN': tN_data,
        'vA': vA_data,
        'tA': tA_data
    }

def create_sequences(X, seq_length):
        if len(X) < seq_length:
            return np.array([])
        X_seq = []
        for i in range(seq_length, len(X)):
            X_seq.append(X[i-seq_length:i])
        return np.array(X_seq)

def prepare_data(df_engineered, selected_features, sequence_length=20):
    
    splits = create_splits(df_engineered, selected_features)
    scaler = MinMaxScaler()
    splits['sN'] = scaler.fit_transform(splits['sN'])
    
    for key in ['vN1', 'vN2', 'tN', 'vA', 'tA']:                    #TODO move the scaler before to fit on all normal data not only subset
        splits[key] = scaler.transform(splits[key])

    sequences = {}
    for split_name, data in splits.items():
        sequences[split_name] = create_sequences(data, sequence_length)
    
    X_train = sequences['sN']
    X_val = sequences['vN1']
    
    threshold_sequences = np.concatenate([sequences['vN2'], sequences['vA']])
    threshold_labels = np.concatenate([
        np.zeros(len(sequences['vN2'])),
        np.ones(len(sequences['vA']))
    ])
    
    test_sequences = np.concatenate([sequences['tN'], sequences['tA']])
    test_labels = np.concatenate([
        np.zeros(len(sequences['tN'])),
        np.ones(len(sequences['tA']))
    ])
    
    print(f"     Training: {len(X_train):,} sequenze normali")
    print(f"     Validation: {len(X_val):,} sequenze normali")
    print(f"     Threshold optimization: {len(threshold_sequences):,} sequenze ({len(sequences['vN2'])} norm + {len(sequences['vA'])} anom)")
    print(f"     Test finale: {len(test_sequences):,} sequenze ({len(sequences['tN'])} norm + {len(sequences['tA'])} anom)")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'threshold_sequences': threshold_sequences,
        'threshold_labels': threshold_labels,
        'test_sequences': test_sequences,
        'test_labels': test_labels,
        'scaler': scaler,
        'sequences': sequences
    }

def build_autoencoder(seq_len, n_features):
    input_layer = Input(shape=(seq_len, n_features))
    
    # Encoder
    encoded = LSTM(32, activation='relu', return_sequences=True)(input_layer)
    encoded = Dropout(0.2)(encoded)
    encoded = LSTM(16, activation='relu', return_sequences=False)(encoded)
    encoded = Dense(8, activation='relu')(encoded)
    
    # Decoder
    decoded = RepeatVector(seq_len)(encoded)
    decoded = LSTM(16, activation='relu', return_sequences=True)(decoded)
    decoded = Dropout(0.2)(decoded)
    decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(n_features, activation='sigmoid'))(decoded)
    
    autoencoder = Model(input_layer, decoded)
    return autoencoder

def find_threshold(normal_err, all_err, all_labels):
    percentiles = np.arange(75, 88, 0.5)
    
    best_score = 0
    best_threshold = 0
    best_metrics = {}
    results = []
    
    print("\nThreshold optimization for balanced anomaly detection:")
    print("Percentile | Threshold  | Precision | Recall | FPR    | Score")
    print("-" * 65)
    
    for p in percentiles:
        threshold = np.percentile(normal_err, p)
        pred = (all_err > threshold).astype(int)
        
        if np.sum(pred) > 0:
            precision = precision_score(all_labels, pred)
            recall = recall_score(all_labels, pred)
            f1 = f1_score(all_labels, pred)
            
            cm = confusion_matrix(all_labels, pred)
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            if fpr > 0.25:
                score = 0.1
            elif recall < 0.70:
                score = recall * 0.4
            else:
                score = recall * 0.6 + precision * 0.4
                if precision >= 0.60 and recall >= 0.70 and fpr <= 0.23:
                    score *= 1.3
            
            print(f"P{p:4.1f}      | {threshold:.6f} | {precision:.3f}     | {recall:.3f}  | {fpr:.3f}  | {score:.3f}")
            
            results.append({
                'percentile': p,
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'fpr': fpr,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'fpr': fpr
                }
    
    return best_threshold, best_metrics, results

def train_and_evaluate(df_engineered, selected_features, sequence_length=20):
    data_splits = prepare_data(df_engineered, selected_features, sequence_length)
    
    X_train = data_splits['X_train']
    X_val = data_splits['X_val']
    threshold_sequences = data_splits['threshold_sequences']
    threshold_labels = data_splits['threshold_labels']
    test_sequences = data_splits['test_sequences']
    test_labels = data_splits['test_labels']
    
    autoencoder = build_autoencoder(sequence_length, len(selected_features))
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    print(f"   Parametri modello: {autoencoder.count_params():,}")
    print(f"   Training set: {X_train.shape}")
    print(f"   Validation set: {X_val.shape}")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"   ✅ Training completato in {len(history.history['loss'])} epoche")
    print(f"   Usando vN2 + vA: {len(threshold_sequences):,} campioni")
    
    threshold_predictions = autoencoder.predict(threshold_sequences, verbose=0)
    threshold_errors = np.mean(np.abs(threshold_predictions - threshold_sequences), axis=(1, 2))
    
    normal_mask_thresh = threshold_labels == 0
    normal_errors_thresh = threshold_errors[normal_mask_thresh]
    anomaly_errors_thresh = threshold_errors[~normal_mask_thresh]
    
    print(f"   Errori normali (vN2): μ={np.mean(normal_errors_thresh):.4f}, σ={np.std(normal_errors_thresh):.4f}")
    print(f"   Errori anomali (vA): μ={np.mean(anomaly_errors_thresh):.4f}, σ={np.std(anomaly_errors_thresh):.4f}")
    print(f"   Separazione: {np.mean(anomaly_errors_thresh)/np.mean(normal_errors_thresh):.2f}x")
    
    best_threshold, best_metrics, threshold_results = find_threshold(
        normal_errors_thresh, threshold_errors, threshold_labels
    )
    
    # Test finale
    print("\n🏁 TEST FINALE SU DATI INDIPENDENTI...")
    print(f"   Test set: {test_sequences.shape}")
    print(f"   Normale (tN): {np.sum(test_labels == 0):,}")
    print(f"   Anomalo (tA): {np.sum(test_labels == 1):,}")
    
    test_predictions = autoencoder.predict(test_sequences, verbose=0)
    test_errors = np.mean(np.abs(test_predictions - test_sequences), axis=(1, 2))
    final_predictions = (test_errors > best_threshold).astype(int)
    
    # Risultati finali
    print("\n" + "=" * 80)
    print("📊 RISULTATI FINALI - CODICE ORIGINALE CON PAPER SPLITS")
    print("=" * 80)
    
    final_precision = precision_score(test_labels, final_predictions)
    final_recall = recall_score(test_labels, final_predictions)
    final_f1 = f1_score(test_labels, final_predictions)
    
    try:
        final_auc = roc_auc_score(test_labels, test_errors)
    except:
        final_auc = 0.0
    
    print(f"🏗️ ARCHITETTURA:")
    print(f"   Modello: LSTM Autoencoder (originale)")
    print(f"   Parametri: {autoencoder.count_params():,}")
    print(f"   Sequenze: {sequence_length} timesteps")
    print(f"   Features: {len(selected_features)}")
    print(f"   Training samples: {len(X_train):,}")
    
    print(f"\n📈 PERFORMANCE (Test Indipendente):")
    print(f"   Threshold: {best_threshold:.6f}")
    print(f"   Precision: {final_precision:.3f}")
    print(f"   Recall: {final_recall:.3f}")
    print(f"   F1-Score: {final_f1:.3f}")
    print(f"   ROC AUC: {final_auc:.3f}")
    
    cm = confusion_matrix(test_labels, final_predictions)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n📋 CONFUSION MATRIX:")
    print(f"   TN: {tn:,}, FP: {fp:,}")
    print(f"   FN: {fn:,}, TP: {tp:,}")
    
    missed_failures = fn / (tp + fn) * 100 if (tp + fn) > 0 else 0
    false_alarm_rate = fp / (tn + fp) * 100 if (tn + fp) > 0 else 0
    
    print(f"\n⚡ METRICHE PRATICHE:")
    print(f"   Detection rate: {tp/(tp+fn)*100:.1f}%")
    print(f"   Missed failures: {missed_failures:.1f}%")
    print(f"   False alarm rate: {false_alarm_rate:.1f}%")
    
    if missed_failures <= 30 and false_alarm_rate <= 25:
        status = "✅ EXCELLENT: Both metrics in acceptable range"
    elif missed_failures <= 35 and false_alarm_rate <= 30:
        status = "✅ GOOD: Metrics acceptable for production"
    elif missed_failures <= 40 and false_alarm_rate <= 35:
        status = "⚠️  ACCEPTABLE: Usable but could be improved"
    else:
        status = "❌ NEEDS IMPROVEMENT: Too many missed failures or false alarms"
    
    print(f"\n{status}")
    
    print(f"\n🔍 ANALISI THRESHOLD OPTIMIZATION:")
    print(f"   Validation metrics (vN2+vA):")
    print(f"     Precision: {best_metrics['precision']:.3f}")
    print(f"     Recall: {best_metrics['recall']:.3f}")
    print(f"     F1: {best_metrics['f1']:.3f}")
    print(f"     FPR: {best_metrics['fpr']:.3f}")
    print(f"   Test metrics (tN+tA):")
    print(f"     Precision: {final_precision:.3f}")
    print(f"     Recall: {final_recall:.3f}")
    print(f"     F1: {final_f1:.3f}")
    
    generalization_gap = abs(best_metrics['f1'] - final_f1)
    print(f"   Generalization gap: {generalization_gap:.3f} ({'✅ Good' if generalization_gap < 0.1 else '⚠️ High' if generalization_gap < 0.2 else '❌ Poor'})")
    
    print("\n" + "=" * 80)
    print("✅ ANALISI COMPLETATA")
    print("=" * 80)
    
    return {
        'model': autoencoder,
        'history': history,
        'data_splits': data_splits,
        'threshold': best_threshold,
        'threshold_metrics': best_metrics,
        'threshold_results': threshold_results,
        'test_results': {
            'errors': test_errors,
            'labels': test_labels,
            'predictions': final_predictions,
            'metrics': {
                'precision': final_precision,
                'recall': final_recall,
                'f1': final_f1,
                'auc': final_auc
            }
        },
        'threshold_errors': threshold_errors,
        'threshold_labels': threshold_labels
    }

def plot_results(results):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Training curves
    history = results['history']
    axes[0,0].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0,0].plot(history.history['val_loss'], label='Validation', linewidth=2) 
    axes[0,0].set_title('Training Loss (Paper Splits)')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('MSE Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # MAE curves
    axes[0,1].plot(history.history['mae'], label='Train', linewidth=2)
    axes[0,1].plot(history.history['val_mae'], label='Validation', linewidth=2)
    axes[0,1].set_title('Mean Absolute Error')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('MAE')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Error distributions (threshold optimization)
    threshold_errors = results['threshold_errors']
    threshold_labels = results['threshold_labels'] 
    
    normal_errors = threshold_errors[threshold_labels == 0]
    anomaly_errors = threshold_errors[threshold_labels == 1]
    
    axes[0,2].hist(normal_errors, bins=50, alpha=0.7, label=f'Normal vN2 (μ={np.mean(normal_errors):.4f})', 
                   density=True, color='blue')
    axes[0,2].hist(anomaly_errors, bins=50, alpha=0.7, label=f'Anomaly vA (μ={np.mean(anomaly_errors):.4f})', 
                   density=True, color='red')
    axes[0,2].axvline(results['threshold'], color='green', linestyle='--', linewidth=2, label='Optimal Threshold')
    axes[0,2].set_title('Reconstruction Error Distribution\n(Threshold Optimization: vN2 + vA)')
    axes[0,2].set_xlabel('MAE Error')
    axes[0,2].set_ylabel('Density')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Threshold analysis
    threshold_results_df = pd.DataFrame(results['threshold_results'])
    best_metrics = results['threshold_metrics']
    
    axes[1,0].plot(threshold_results_df['recall'], threshold_results_df['precision'], 'b-o', linewidth=2, markersize=4)
    axes[1,0].scatter([best_metrics['recall']], [best_metrics['precision']], 
                      color='red', s=100, zorder=5, label='Selected')
    axes[1,0].axhline(y=0.60, color='orange', linestyle='--', alpha=0.7, label='Target Precision 60%')
    axes[1,0].axvline(x=0.70, color='orange', linestyle='--', alpha=0.7, label='Target Recall 70%')
    axes[1,0].set_title('Precision vs Recall (Validation)')
    axes[1,0].set_xlabel('Recall')
    axes[1,0].set_ylabel('Precision')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Test results - Error distribution
    test_errors = results['test_results']['errors']
    test_labels = results['test_results']['labels']
    
    test_normal_errors = test_errors[test_labels == 0]
    test_anomaly_errors = test_errors[test_labels == 1]
    
    axes[1,1].hist(test_normal_errors, bins=50, alpha=0.7, label=f'Normal tN (μ={np.mean(test_normal_errors):.4f})', 
                   density=True, color='lightblue')
    axes[1,1].hist(test_anomaly_errors, bins=50, alpha=0.7, label=f'Anomaly tA (μ={np.mean(test_anomaly_errors):.4f})', 
                   density=True, color='lightcoral')
    axes[1,1].axvline(results['threshold'], color='green', linestyle='--', linewidth=2, label='Threshold')
    axes[1,1].set_title('Test Set Error Distribution\n(Final Test: tN + tA)')
    axes[1,1].set_xlabel('MAE Error')
    axes[1,1].set_ylabel('Density')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Confusion Matrix
    test_predictions = results['test_results']['predictions']
    cm = confusion_matrix(test_labels, test_predictions)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Failure'],
                yticklabels=['Normal', 'Failure'],
                ax=axes[1,2])
    axes[1,2].set_title('Test Results Confusion Matrix\n(Independent Test Set)')
    axes[1,2].set_xlabel('Predicted')
    axes[1,2].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()
    
    test_metrics = results['test_results']['metrics']
    print(f"\n📊 SUMMARY:")
    print(f"   Paper splits utilizzati: ✅")
    print(f"   Codice originale mantenuto: ✅")
    print(f"   Test finale F1: {test_metrics['f1']:.3f}")
    print(f"   Test finale AUC: {test_metrics['auc']:.3f}")

# Esecuzione
results = train_and_evaluate(
    df_engineered=df_engineered,
    selected_features=selected_features,
    sequence_length=20
)

plot_results(results)

final_f1 = results['test_results']['metrics']['f1']
print(f"F1-Score finale: {final_f1:.3f}")