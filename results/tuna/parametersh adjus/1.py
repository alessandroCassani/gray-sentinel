def find_threshold(normal_err, all_err, all_labels):
    percentiles = np.arange(70, 95, 1.0)
    
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
            
            if fpr > 0.30:
                score = 0.1
            elif recall < 0.60:
                score = recall * 0.4
            else:
                base_score = recall * 0.6 + precision * 0.4
                if recall >= 0.75 and precision >= 0.75: 
                    base_score *= 1.5
                elif precision >= 0.75 and recall >= 0.65 and fpr <= 0.30:
                    base_score *= 1.2
                score = base_score
            
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