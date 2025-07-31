"""
LSTM AUTOENCODER CON DATASET SPLIT DEL PAPER
===========================================

Mantiene il codice originale (architettura, threshold semplice) ma utilizza
la suddivisione del dataset esattamente come descritto nel paper Malhotra et al. 2015

Split del paper:
- sN: normal train (60%) - per training del modello
- vN1: normal validation-1 (15%) - per early stopping
- vN2: normal validation-2 (15%) - per stima distribuzione/threshold
- tN: normal test (10%) - per test finale
- vA: anomalous validation (50%) - per ottimizzazione threshold
- tA: anomalous test (50%) - per test finale
"""

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

# ================================================================
# 1. DATASET PREPARATION CON SPLIT DEL PAPER
# ================================================================

def create_paper_splits_simple(df_engineered, selected_features, sequence_length=20, random_state=42):
    """
    Crea splits seguendo il paper ma per uso con codice originale
    """
    print("🔄 CREAZIONE DATASET SPLITS SEGUENDO IL PAPER...")
    print("=" * 60)
    
    # Separazione dati normali e anomali
    normal_data = df_engineered[df_engineered['Failure'] == 0]
    anomalous_data = df_engineered[df_engineered['Failure'] == 1]
    
    print(f"📊 Dataset Overview:")
    print(f"   Totale campioni: {len(df_engineered):,}")
    print(f"   Campioni normali: {len(normal_data):,} ({len(normal_data)/len(df_engineered)*100:.1f}%)")
    print(f"   Campioni anomali: {len(anomalous_data):,} ({len(anomalous_data)/len(df_engineered)*100:.1f}%)")
    print(f"   Features selezionate: {len(selected_features)}")
    
    # === SPLIT DATI NORMALI ===
    normal_features = normal_data[selected_features].values
    
    # Prima divisione: 60% train, 40% resto
    sN_data, temp_normal = train_test_split(
        normal_features, test_size=0.4, random_state=random_state, shuffle=True
    )
    
    # Seconda divisione: 15% vN1, 25% resto
    vN1_data, temp_normal2 = train_test_split(
        temp_normal, test_size=0.625, random_state=random_state  # 25/40 = 0.625
    )
    
    # Terza divisione: 15% vN2, 10% tN
    vN2_data, tN_data = train_test_split(
        temp_normal2, test_size=0.4, random_state=random_state  # 10/25 = 0.4
    )
    
    # === SPLIT DATI ANOMALI ===
    anomalous_features = anomalous_data[selected_features].values
    vA_data, tA_data = train_test_split(
        anomalous_features, test_size=0.5, random_state=random_state
    )
    
    # Statistiche
    total_normal = len(normal_features)
    print(f"\\n📋 SPLITS CREATI:")
    print(f"   sN  (normal train):     {len(sN_data):,} ({len(sN_data)/total_normal*100:.1f}%)")
    print(f"   vN1 (normal val-1):     {len(vN1_data):,} ({len(vN1_data)/total_normal*100:.1f}%)")
    print(f"   vN2 (normal val-2):     {len(vN2_data):,} ({len(vN2_data)/total_normal*100:.1f}%)")
    print(f"   tN  (normal test):      {len(tN_data):,} ({len(tN_data)/total_normal*100:.1f}%)")
    print(f"   vA  (anomalous val):    {len(vA_data):,} (50.0%)")
    print(f"   tA  (anomalous test):   {len(tA_data):,} (50.0%)")
    
    return {
        'sN': sN_data,
        'vN1': vN1_data,
        'vN2': vN2_data,
        'tN': tN_data,
        'vA': vA_data,
        'tA': tA_data
    }

def prepare_data_with_paper_splits(df_engineered, selected_features, sequence_length=20):
    """
    Prepara i dati usando la suddivisione del paper ma mantenendo
    compatibilità con il codice originale
    """
    print("\\n🔧 PREPARAZIONE DATI CON PAPER SPLITS...")
    
    # Crea splits
    splits = create_paper_splits_simple(df_engineered, selected_features, sequence_length)
    
    # Normalizzazione (fit solo su sN)
    print("\\n   Normalizzazione dati...")
    scaler = MinMaxScaler()
    
    # Fit scaler solo sui dati di training normali
    splits['sN'] = scaler.fit_transform(splits['sN'])
    
    # Transform tutti gli altri
    for key in ['vN1', 'vN2', 'tN', 'vA', 'tA']:
        if len(splits[key]) > 0:
            splits[key] = scaler.transform(splits[key])
    
    # Funzione per creare sequenze
    def create_sequences(X, seq_length):
        if len(X) < seq_length:
            return np.array([])
        X_seq = []
        for i in range(seq_length, len(X)):
            X_seq.append(X[i-seq_length:i])
        return np.array(X_seq)
    
    # Crea sequenze per ogni split
    print("\\n   Creazione sequenze temporali...")
    sequences = {}
    for split_name, data in splits.items():
        sequences[split_name] = create_sequences(data, sequence_length)
        if len(sequences[split_name]) > 0:
            print(f"     {split_name}: {len(sequences[split_name]):,} sequenze")
        else:
            print(f"     {split_name}: ⚠️ dati insufficienti")
    
    # Prepara dati per compatibilità con codice originale
    print("\\n   Preparazione dati compatibili...")
    
    # Training e validation (come nel codice originale)
    X_train = sequences['sN']
    X_val = sequences['vN1']
    
    # Per threshold optimization: combina vN2 (normale) + vA (anomalo)
    threshold_sequences = np.concatenate([sequences['vN2'], sequences['vA']])
    threshold_labels = np.concatenate([
        np.zeros(len(sequences['vN2'])),  # normale = 0
        np.ones(len(sequences['vA']))     # anomalo = 1
    ])
    
    # Per test finale: combina tN (normale) + tA (anomalo) 
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
        'sequences': sequences  # Per analisi dettagliate
    }

# ================================================================
# 2. MODELLO LSTM AUTOENCODER (CODICE ORIGINALE)
# ================================================================

def build_autoencoder_original(seq_len, n_features):
    """
    Stesso modello del codice originale
    """
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

# ================================================================
# 3. THRESHOLD OPTIMIZATION (CODICE ORIGINALE)
# ================================================================

def find_balanced_threshold_original(normal_err, all_err, all_labels):
    """
    Stessa funzione del codice originale per threshold optimization
    """
    percentiles = np.arange(75, 88, 0.5)
    
    best_score = 0
    best_threshold = 0
    best_metrics = {}
    results = []
    
    print("\\nThreshold optimization for balanced anomaly detection:")
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
            
            # Score bilanciato (stesso del codice originale)
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

# ================================================================
# 4. FUNZIONE PRINCIPALE
# ================================================================

def run_original_code_with_paper_splits(df_engineered, selected_features, sequence_length=20):
    """
    Esegue il codice originale ma con la suddivisione del dataset del paper
    """
    print("🚀 LSTM AUTOENCODER - CODICE ORIGINALE CON PAPER SPLITS")
    print("=" * 80)
    print("   Architettura: ORIGINALE (LSTM Autoencoder)")
    print("   Dataset Split: PAPER (Malhotra et al. 2015)")
    print("   Threshold: BALANCED (Precision/Recall trade-off)")
    print("=" * 80)
    
    # === STEP 1: Preparazione Dati ===
    data_splits = prepare_data_with_paper_splits(df_engineered, selected_features, sequence_length)
    
    X_train = data_splits['X_train']
    X_val = data_splits['X_val']
    threshold_sequences = data_splits['threshold_sequences']
    threshold_labels = data_splits['threshold_labels']
    test_sequences = data_splits['test_sequences']
    test_labels = data_splits['test_labels']
    
    # === STEP 2: Costruzione e Training Modello ===
    print("\\n🏗️ COSTRUZIONE E TRAINING MODELLO...")
    
    autoencoder = build_autoencoder_original(sequence_length, len(selected_features))
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    print(f"   Parametri modello: {autoencoder.count_params():,}")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    print(f"   Training set: {X_train.shape}")
    print(f"   Validation set: {X_val.shape}")
    
    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"   ✅ Training completato in {len(history.history['loss'])} epoche")
    
    # === STEP 3: Threshold Optimization ===
    print("\\n🎯 OTTIMIZZAZIONE THRESHOLD SU VALIDATION SETS...")
    print(f"   Usando vN2 + vA: {len(threshold_sequences):,} campioni")
    
    # Calcola reconstruction errors sui dati di threshold optimization
    threshold_predictions = autoencoder.predict(threshold_sequences, verbose=0)
    threshold_errors = np.mean(np.abs(threshold_predictions - threshold_sequences), axis=(1, 2))
    
    # Separa errori normali e anomali per analisi
    normal_mask_thresh = threshold_labels == 0
    normal_errors_thresh = threshold_errors[normal_mask_thresh]
    anomaly_errors_thresh = threshold_errors[~normal_mask_thresh]
    
    print(f"   Errori normali (vN2): μ={np.mean(normal_errors_thresh):.4f}, σ={np.std(normal_errors_thresh):.4f}")
    print(f"   Errori anomali (vA): μ={np.mean(anomaly_errors_thresh):.4f}, σ={np.std(anomaly_errors_thresh):.4f}")
    print(f"   Separazione: {np.mean(anomaly_errors_thresh)/np.mean(normal_errors_thresh):.2f}x")
    
    # Trova threshold ottimale
    best_threshold, best_metrics, threshold_results = find_balanced_threshold_original(
        normal_errors_thresh, threshold_errors, threshold_labels
    )
    
    # === STEP 4: Test Finale ===
    print("\\n🏁 TEST FINALE SU DATI INDIPENDENTI...")
    print(f"   Test set: {test_sequences.shape}")
    print(f"   Normale (tN): {np.sum(test_labels == 0):,}")
    print(f"   Anomalo (tA): {np.sum(test_labels == 1):,}")
    
    # Predizioni su test set
    test_predictions = autoencoder.predict(test_sequences, verbose=0)
    test_errors = np.mean(np.abs(test_predictions - test_sequences), axis=(1, 2))
    
    # Classificazione finale
    final_predictions = (test_errors > best_threshold).astype(int)
    
    # === RISULTATI FINALI ===
    print("\\n" + "=" * 80)
    print("📊 RISULTATI FINALI - CODICE ORIGINALE CON PAPER SPLITS")
    print("=" * 80)
    
    # Metriche principali
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
    
    print(f"\\n📈 PERFORMANCE (Test Indipendente):")
    print(f"   Threshold: {best_threshold:.6f}")
    print(f"   Precision: {final_precision:.3f}")
    print(f"   Recall: {final_recall:.3f}")
    print(f"   F1-Score: {final_f1:.3f}")
    print(f"   ROC AUC: {final_auc:.3f}")
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, final_predictions)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\\n📋 CONFUSION MATRIX:")
    print(f"   TN: {tn:,}, FP: {fp:,}")
    print(f"   FN: {fn:,}, TP: {tp:,}")
    
    # Metriche pratiche
    missed_failures = fn / (tp + fn) * 100 if (tp + fn) > 0 else 0
    false_alarm_rate = fp / (tn + fp) * 100 if (tn + fp) > 0 else 0
    
    print(f"\\n⚡ METRICHE PRATICHE:")
    print(f"   Detection rate: {tp/(tp+fn)*100:.1f}%")
    print(f"   Missed failures: {missed_failures:.1f}%")
    print(f"   False alarm rate: {false_alarm_rate:.1f}%")
    
    # Valutazione
    if missed_failures <= 30 and false_alarm_rate <= 25:
        status = "✅ EXCELLENT: Both metrics in acceptable range"
    elif missed_failures <= 35 and false_alarm_rate <= 30:
        status = "✅ GOOD: Metrics acceptable for production"
    elif missed_failures <= 40 and false_alarm_rate <= 35:
        status = "⚠️  ACCEPTABLE: Usable but could be improved"
    else:
        status = "❌ NEEDS IMPROVEMENT: Too many missed failures or false alarms"
    
    print(f"\\n{status}")
    
    # Confronto threshold vs validation optimization
    print(f"\\n🔍 ANALISI THRESHOLD OPTIMIZATION:")
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
    
    print("\\n" + "=" * 80)
    print("✅ ANALISI COMPLETATA")
    print("=" * 80)
    
    # Ritorna risultati per visualizzazioni
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

# ================================================================
# 5. VISUALIZZAZIONI (OPZIONALI)
# ================================================================

def plot_results_with_paper_splits(results):
    """
    Crea visualizzazioni per i risultati con paper splits
    """
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
    
    # Error distributions (threshold optimization data)
    threshold_errors = results['threshold_errors']
    threshold_labels = results['threshold_labels'] 
    
    normal_errors = threshold_errors[threshold_labels == 0]
    anomaly_errors = threshold_errors[threshold_labels == 1]
    
    axes[0,2].hist(normal_errors, bins=50, alpha=0.7, label=f'Normal vN2 (μ={np.mean(normal_errors):.4f})', 
                   density=True, color='blue')
    axes[0,2].hist(anomaly_errors, bins=50, alpha=0.7, label=f'Anomaly vA (μ={np.mean(anomaly_errors):.4f})', 
                   density=True, color='red')
    axes[0,2].axvline(results['threshold'], color='green', linestyle='--', linewidth=2, label='Optimal Threshold')
    axes[0,2].set_title('Reconstruction Error Distribution\\n(Threshold Optimization: vN2 + vA)')
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
    axes[1,1].set_title('Test Set Error Distribution\\n(Final Test: tN + tA)')
    axes[1,1].set_xlabel('MAE Error')
    axes[1,1].set_ylabel('Density')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Confusion Matrix (test results)
    test_predictions = results['test_results']['predictions']
    cm = confusion_matrix(test_labels, test_predictions)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Failure'],
                yticklabels=['Normal', 'Failure'],
                ax=axes[1,2])
    axes[1,2].set_title('Test Results Confusion Matrix\\n(Independent Test Set)')
    axes[1,2].set_xlabel('Predicted')
    axes[1,2].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()
    
    # Summary
    test_metrics = results['test_results']['metrics']
    print(f"\\n📊 SUMMARY:")
    print(f"   Paper splits utilizzati: ✅")
    print(f"   Codice originale mantenuto: ✅")
    print(f"   Test finale F1: {test_metrics['f1']:.3f}")
    print(f"   Test finale AUC: {test_metrics['auc']:.3f}")

# ================================================================
# 6. ESEMPIO DI UTILIZZO
# ================================================================


results = run_original_code_with_paper_splits(
    df_engineered=df_engineered,
    selected_features=selected_features,
    sequence_length=20
)

# Opzionale: visualizzazioni
plot_results_with_paper_splits(results)

# Accesso ai risultati
final_f1 = results['test_results']['metrics']['f1']
print(f"F1-Score finale: {final_f1:.3f}")
