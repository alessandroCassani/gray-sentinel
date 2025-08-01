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





📋 TIME-AWARE SPLITS CREATI:
   sN  (normal train):     3,018 (60.0%) - Early period
   vN1 (normal val-1):     755 (15.0%) - Mid-early period
   vN2 (normal val-2):     754 (15.0%) - Mid-late period
   tN  (normal test):      504 (10.0%) - Late period
   vA  (anomalous val):    1,600 (50.0%) - Early anomalies
   tA  (anomalous test):   1,600 (50.0%) - Late anomalies

🔍 ANALISI TEMPORALE:
   Gap between train end and val start: 44970s (749.5min)
   Clean temporal separation maintained
     Training: 2,998 sequenze normali
     Validation: 735 sequenze normali
     Threshold optimization: 2,314 sequenze (734 norm + 1580 anom)
     Test finale: 2,064 sequenze (484 norm + 1580 anom)
   Parametri modello: 64,666
   Training set: (2998, 20, 10)
   Validation set: (735, 20, 10)
Epoch 1/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 8s 20ms/step - loss: 0.8007 - mae: 0.9751 - val_loss: 0.3323 - val_mae: 0.3797 - learning_rate: 5.0000e-04
Epoch 2/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 17ms/step - loss: 0.4932 - mae: 0.6543 - val_loss: 0.2647 - val_mae: 0.3266 - learning_rate: 5.0000e-04
Epoch 3/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 19ms/step - loss: 0.3700 - mae: 0.5328 - val_loss: 0.2245 - val_mae: 0.3074 - learning_rate: 5.0000e-04
Epoch 4/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.2946 - mae: 0.4605 - val_loss: 0.1948 - val_mae: 0.2974 - learning_rate: 5.0000e-04
Epoch 5/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 20ms/step - loss: 0.2440 - mae: 0.4130 - val_loss: 0.1733 - val_mae: 0.2928 - learning_rate: 5.0000e-04
Epoch 6/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 20ms/step - loss: 0.2103 - mae: 0.3832 - val_loss: 0.1548 - val_mae: 0.2871 - learning_rate: 5.0000e-04
Epoch 7/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 20ms/step - loss: 0.1861 - mae: 0.3624 - val_loss: 0.1418 - val_mae: 0.2811 - learning_rate: 5.0000e-04
Epoch 8/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.1739 - mae: 0.3558 - val_loss: 0.1323 - val_mae: 0.2773 - learning_rate: 5.0000e-04
Epoch 9/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 23ms/step - loss: 0.1589 - mae: 0.3436 - val_loss: 0.1225 - val_mae: 0.2726 - learning_rate: 5.0000e-04
Epoch 10/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 5s 25ms/step - loss: 0.1544 - mae: 0.3467 - val_loss: 0.1160 - val_mae: 0.2762 - learning_rate: 5.0000e-04
Epoch 11/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 5s 27ms/step - loss: 0.1412 - mae: 0.3323 - val_loss: 0.1092 - val_mae: 0.2704 - learning_rate: 5.0000e-04
Epoch 12/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 5s 26ms/step - loss: 0.1343 - mae: 0.3306 - val_loss: 0.1021 - val_mae: 0.2597 - learning_rate: 5.0000e-04
Epoch 13/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 5s 27ms/step - loss: 0.1254 - mae: 0.3223 - val_loss: 0.1008 - val_mae: 0.2627 - learning_rate: 5.0000e-04
Epoch 14/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 6s 32ms/step - loss: 0.1208 - mae: 0.3182 - val_loss: 0.0955 - val_mae: 0.2580 - learning_rate: 5.0000e-04
Epoch 15/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 5s 27ms/step - loss: 0.1152 - mae: 0.3133 - val_loss: 0.0918 - val_mae: 0.2513 - learning_rate: 5.0000e-04
Epoch 16/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 5s 28ms/step - loss: 0.1128 - mae: 0.3108 - val_loss: 0.0885 - val_mae: 0.2469 - learning_rate: 5.0000e-04
Epoch 17/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 5s 29ms/step - loss: 0.1132 - mae: 0.3136 - val_loss: 0.0871 - val_mae: 0.2471 - learning_rate: 5.0000e-04
Epoch 18/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 6s 33ms/step - loss: 0.1093 - mae: 0.3086 - val_loss: 0.0860 - val_mae: 0.2418 - learning_rate: 5.0000e-04
Epoch 19/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 5s 28ms/step - loss: 0.1080 - mae: 0.3065 - val_loss: 0.0847 - val_mae: 0.2413 - learning_rate: 5.0000e-04
Epoch 20/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 6s 30ms/step - loss: 0.1063 - mae: 0.3062 - val_loss: 0.0841 - val_mae: 0.2439 - learning_rate: 5.0000e-04
Epoch 21/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 5s 29ms/step - loss: 0.1032 - mae: 0.3023 - val_loss: 0.0826 - val_mae: 0.2442 - learning_rate: 5.0000e-04
Epoch 22/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 22ms/step - loss: 0.1039 - mae: 0.3037 - val_loss: 0.0825 - val_mae: 0.2415 - learning_rate: 5.0000e-04
Epoch 23/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.1044 - mae: 0.3052 - val_loss: 0.0834 - val_mae: 0.2391 - learning_rate: 5.0000e-04
Epoch 24/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 22ms/step - loss: 0.1011 - mae: 0.3002 - val_loss: 0.0809 - val_mae: 0.2459 - learning_rate: 5.0000e-04
Epoch 25/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 22ms/step - loss: 0.1000 - mae: 0.2991 - val_loss: 0.0814 - val_mae: 0.2369 - learning_rate: 5.0000e-04
Epoch 26/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 22ms/step - loss: 0.0993 - mae: 0.2985 - val_loss: 0.0808 - val_mae: 0.2392 - learning_rate: 5.0000e-04
Epoch 27/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 22ms/step - loss: 0.0996 - mae: 0.2996 - val_loss: 0.0797 - val_mae: 0.2357 - learning_rate: 5.0000e-04
Epoch 28/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 22ms/step - loss: 0.1001 - mae: 0.2998 - val_loss: 0.0792 - val_mae: 0.2354 - learning_rate: 5.0000e-04
Epoch 29/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 22ms/step - loss: 0.0975 - mae: 0.2967 - val_loss: 0.0797 - val_mae: 0.2443 - learning_rate: 5.0000e-04
Epoch 30/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 23ms/step - loss: 0.0987 - mae: 0.2988 - val_loss: 0.0793 - val_mae: 0.2463 - learning_rate: 5.0000e-04
Epoch 31/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 5s 25ms/step - loss: 0.0987 - mae: 0.2990 - val_loss: 0.0772 - val_mae: 0.2319 - learning_rate: 5.0000e-04
Epoch 32/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 23ms/step - loss: 0.0974 - mae: 0.2959 - val_loss: 0.0773 - val_mae: 0.2333 - learning_rate: 5.0000e-04
Epoch 33/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 5s 28ms/step - loss: 0.0962 - mae: 0.2946 - val_loss: 0.0778 - val_mae: 0.2347 - learning_rate: 5.0000e-04
Epoch 34/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 22ms/step - loss: 0.0976 - mae: 0.2981 - val_loss: 0.0811 - val_mae: 0.2493 - learning_rate: 5.0000e-04
Epoch 35/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0986 - mae: 0.2981 - val_loss: 0.0788 - val_mae: 0.2400 - learning_rate: 5.0000e-04
Epoch 36/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 22ms/step - loss: 0.0947 - mae: 0.2924 - val_loss: 0.0783 - val_mae: 0.2427 - learning_rate: 5.0000e-04
Epoch 37/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 22ms/step - loss: 0.0983 - mae: 0.2972 - val_loss: 0.0795 - val_mae: 0.2406 - learning_rate: 5.0000e-04
Epoch 38/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 22ms/step - loss: 0.0959 - mae: 0.2934 - val_loss: 0.0771 - val_mae: 0.2313 - learning_rate: 5.0000e-04
Epoch 39/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 7s 35ms/step - loss: 0.0950 - mae: 0.2937 - val_loss: 0.0767 - val_mae: 0.2290 - learning_rate: 5.0000e-04
Epoch 40/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 5s 24ms/step - loss: 0.0951 - mae: 0.2927 - val_loss: 0.0761 - val_mae: 0.2380 - learning_rate: 5.0000e-04
Epoch 41/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 5s 29ms/step - loss: 0.0953 - mae: 0.2940 - val_loss: 0.0776 - val_mae: 0.2376 - learning_rate: 5.0000e-04
Epoch 42/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 5s 27ms/step - loss: 0.0960 - mae: 0.2954 - val_loss: 0.0784 - val_mae: 0.2427 - learning_rate: 5.0000e-04
Epoch 43/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 22ms/step - loss: 0.0951 - mae: 0.2944 - val_loss: 0.0766 - val_mae: 0.2379 - learning_rate: 5.0000e-04
Epoch 44/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0933 - mae: 0.2913 - val_loss: 0.0759 - val_mae: 0.2332 - learning_rate: 5.0000e-04
Epoch 45/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0942 - mae: 0.2915 - val_loss: 0.0749 - val_mae: 0.2309 - learning_rate: 5.0000e-04
Epoch 46/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0948 - mae: 0.2922 - val_loss: 0.0761 - val_mae: 0.2325 - learning_rate: 5.0000e-04
Epoch 47/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0918 - mae: 0.2881 - val_loss: 0.0761 - val_mae: 0.2315 - learning_rate: 5.0000e-04
Epoch 48/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0933 - mae: 0.2910 - val_loss: 0.0761 - val_mae: 0.2329 - learning_rate: 5.0000e-04
Epoch 49/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0943 - mae: 0.2929 - val_loss: 0.0767 - val_mae: 0.2392 - learning_rate: 5.0000e-04
Epoch 50/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0941 - mae: 0.2930 - val_loss: 0.0773 - val_mae: 0.2337 - learning_rate: 5.0000e-04
Epoch 51/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0943 - mae: 0.2922 - val_loss: 0.0758 - val_mae: 0.2397 - learning_rate: 5.0000e-04
Epoch 52/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0939 - mae: 0.2920 - val_loss: 0.0750 - val_mae: 0.2320 - learning_rate: 5.0000e-04
Epoch 53/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0920 - mae: 0.2873 - val_loss: 0.0789 - val_mae: 0.2501 - learning_rate: 5.0000e-04
Epoch 54/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0927 - mae: 0.2908 - val_loss: 0.0744 - val_mae: 0.2291 - learning_rate: 3.5000e-04
Epoch 55/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0933 - mae: 0.2908 - val_loss: 0.0737 - val_mae: 0.2305 - learning_rate: 3.5000e-04
Epoch 56/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0926 - mae: 0.2900 - val_loss: 0.0748 - val_mae: 0.2343 - learning_rate: 3.5000e-04
Epoch 57/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0922 - mae: 0.2897 - val_loss: 0.0747 - val_mae: 0.2294 - learning_rate: 3.5000e-04
Epoch 58/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0912 - mae: 0.2881 - val_loss: 0.0740 - val_mae: 0.2301 - learning_rate: 3.5000e-04
Epoch 59/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0899 - mae: 0.2857 - val_loss: 0.0767 - val_mae: 0.2395 - learning_rate: 3.5000e-04
Epoch 60/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0894 - mae: 0.2852 - val_loss: 0.0736 - val_mae: 0.2340 - learning_rate: 3.5000e-04
Epoch 61/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0910 - mae: 0.2876 - val_loss: 0.0724 - val_mae: 0.2243 - learning_rate: 3.5000e-04
Epoch 62/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0921 - mae: 0.2901 - val_loss: 0.0726 - val_mae: 0.2230 - learning_rate: 3.5000e-04
Epoch 63/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0911 - mae: 0.2871 - val_loss: 0.0739 - val_mae: 0.2318 - learning_rate: 3.5000e-04
Epoch 64/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0902 - mae: 0.2867 - val_loss: 0.0736 - val_mae: 0.2297 - learning_rate: 3.5000e-04
Epoch 65/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0935 - mae: 0.2927 - val_loss: 0.0716 - val_mae: 0.2233 - learning_rate: 3.5000e-04
Epoch 66/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0903 - mae: 0.2867 - val_loss: 0.0725 - val_mae: 0.2281 - learning_rate: 3.5000e-04
Epoch 67/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 22ms/step - loss: 0.0922 - mae: 0.2902 - val_loss: 0.0729 - val_mae: 0.2244 - learning_rate: 3.5000e-04
Epoch 68/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0923 - mae: 0.2909 - val_loss: 0.0727 - val_mae: 0.2280 - learning_rate: 3.5000e-04
Epoch 69/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0892 - mae: 0.2843 - val_loss: 0.0728 - val_mae: 0.2305 - learning_rate: 3.5000e-04
Epoch 70/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0889 - mae: 0.2850 - val_loss: 0.0727 - val_mae: 0.2263 - learning_rate: 3.5000e-04
Epoch 71/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0900 - mae: 0.2861 - val_loss: 0.0738 - val_mae: 0.2271 - learning_rate: 3.5000e-04
Epoch 72/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0923 - mae: 0.2917 - val_loss: 0.0728 - val_mae: 0.2281 - learning_rate: 3.5000e-04
Epoch 73/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0897 - mae: 0.2867 - val_loss: 0.0735 - val_mae: 0.2366 - learning_rate: 3.5000e-04
Epoch 74/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0871 - mae: 0.2817 - val_loss: 0.0715 - val_mae: 0.2216 - learning_rate: 2.4500e-04
Epoch 75/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0878 - mae: 0.2829 - val_loss: 0.0722 - val_mae: 0.2288 - learning_rate: 2.4500e-04
Epoch 76/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0899 - mae: 0.2865 - val_loss: 0.0723 - val_mae: 0.2251 - learning_rate: 2.4500e-04
Epoch 77/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0884 - mae: 0.2844 - val_loss: 0.0704 - val_mae: 0.2201 - learning_rate: 2.4500e-04
Epoch 78/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0884 - mae: 0.2844 - val_loss: 0.0730 - val_mae: 0.2308 - learning_rate: 2.4500e-04
Epoch 79/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0881 - mae: 0.2845 - val_loss: 0.0720 - val_mae: 0.2269 - learning_rate: 2.4500e-04
Epoch 80/80
188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 22ms/step - loss: 0.0892 - mae: 0.2862 - val_loss: 0.0714 - val_mae: 0.2292 - learning_rate: 2.4500e-04
   ✅ Training completato in 80 epoche
   Usando vN2 + vA: 2,314 campioni
   Errori normali (vN2): μ=0.2185, σ=0.0604
   Errori anomali (vA): μ=0.3163, σ=0.1352
   Separazione: 1.45x

Threshold optimization for balanced anomaly detection:
Percentile | Threshold  | Precision | Recall | FPR    | Score
-----------------------------------------------------------------
P70.0      | 0.228508 | 0.830     | 0.678  | 0.300  | 0.887
P71.0      | 0.229931 | 0.833     | 0.673  | 0.290  | 0.884
P72.0      | 0.231817 | 0.837     | 0.667  | 0.281  | 0.882
P73.0      | 0.233093 | 0.841     | 0.663  | 0.270  | 0.881
P74.0      | 0.234253 | 0.845     | 0.661  | 0.260  | 0.882
P75.0      | 0.235121 | 0.850     | 0.659  | 0.251  | 0.882
P76.0      | 0.236790 | 0.855     | 0.655  | 0.240  | 0.882
P77.0      | 0.240106 | 0.858     | 0.647  | 0.230  | 0.731
P78.0      | 0.243475 | 0.861     | 0.634  | 0.221  | 0.725
P79.0      | 0.246135 | 0.865     | 0.623  | 0.210  | 0.720
P80.0      | 0.247710 | 0.869     | 0.618  | 0.200  | 0.719
P81.0      | 0.254655 | 0.869     | 0.589  | 0.191  | 0.236
P82.0      | 0.258984 | 0.872     | 0.570  | 0.180  | 0.228
P83.0      | 0.263097 | 0.875     | 0.554  | 0.170  | 0.222
P84.0      | 0.266144 | 0.879     | 0.543  | 0.161  | 0.217
P85.0      | 0.271063 | 0.883     | 0.525  | 0.150  | 0.210
P86.0      | 0.274222 | 0.888     | 0.515  | 0.140  | 0.206
P87.0      | 0.284860 | 0.887     | 0.477  | 0.131  | 0.191
P88.0      | 0.292982 | 0.888     | 0.443  | 0.120  | 0.177
P89.0      | 0.297809 | 0.891     | 0.418  | 0.110  | 0.167
P90.0      | 0.304221 | 0.894     | 0.396  | 0.101  | 0.158
P91.0      | 0.318212 | 0.894     | 0.352  | 0.090  | 0.141
P92.0      | 0.331776 | 0.894     | 0.315  | 0.080  | 0.126
P93.0      | 0.347120 | 0.896     | 0.285  | 0.071  | 0.114
P94.0      | 0.352565 | 0.907     | 0.272  | 0.060  | 0.109

🏁 TEST FINALE SU DATI INDIPENDENTI...
   Test set: (2064, 20, 10)
   Normale (tN): 484
   Anomalo (tA): 1,580

================================================================================
📊 RISULTATI FINALI - CODICE ORIGINALE CON PAPER SPLITS
================================================================================
🏗️ ARCHITETTURA:
   Modello: LSTM Autoencoder (originale)
   Parametri: 64,666
   Sequenze: 20 timesteps
   Features: 10
   Training samples: 2,998

📈 PERFORMANCE (Test Indipendente):
   Threshold: 0.228508
   Precision: 0.877
   Recall: 0.720
   F1-Score: 0.790
   ROC AUC: 0.751

📋 CONFUSION MATRIX:
   TN: 324, FP: 160
   FN: 443, TP: 1,137

⚡ METRICHE PRATICHE:
   Detection rate: 72.0%
   Missed failures: 28.0%
   False alarm rate: 33.1%

❌ NEEDS IMPROVEMENT: Too many missed failures or false alarms

📈 CLASSIFICATION REPORT:
              precision    recall  f1-score   support

      Normal      0.422     0.669     0.518       484
     Failure      0.877     0.720     0.790      1580

    accuracy                          0.708      2064
   macro avg      0.650     0.695     0.654      2064
weighted avg      0.770     0.708     0.727      2064


🔍 ANALISI THRESHOLD OPTIMIZATION:
   Validation metrics (vN2+vA):
     Precision: 0.830
     Recall: 0.678
     F1: 0.747
     FPR: 0.300
   Test metrics (tN+tA):
     Precision: 0.877
     Recall: 0.720
     F1: 0.790
   Generalization gap: 0.044 (✅ Good)

================================================================================
✅ ANALISI COMPLETATA
================================================================================