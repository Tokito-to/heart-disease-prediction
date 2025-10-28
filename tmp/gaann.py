import csv
import tensorflow as tf

# Logging
log_file = 'models/logs/ReLU_model.csv'
with open(log_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Generation', 'SelectedFeatures', 'n1', 'n2', 'n3', 'lr', 'dr', 'l2', 'alpha', 'Accuracy', 'TP', 'FP', 'FN', 'TN'])

# Final model training
print("Training model with best parameters...")

best_params = tools.selBest(population, k=1)[0]
feature_mask = best_params[:12]
hyperparams = best_params[12:]

selected_indices = [i for i, bit in enumerate(feature_mask) if bit == 1]
X_train_selected = X_train[:, selected_indices]
X_test_selected = X_test[:, selected_indices]

model = create_ann_model(hyperparams, fit_model=False, input_shape=len(selected_indices))
model_history = model.fit(X_train_selected, y_train, validation_split=0.30, epochs=145, batch_size=35,
                          callbacks=[early_stopping, reduce_lr], verbose=0)

model.save('models/ReLU_heart_model.keras')

log_data = {
    'history': model_history.history,
    'selected_features': selected_indices,
    'hyperparams': hyperparams
}
joblib.dump(log_data, 'models/logs/ReLU_model_logs.pkl')

# Final evaluation
y_prob = model.predict(X_test_selected)
y_pred = (y_prob > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print("Final Test Accuracy: {:.2f}%".format(accuracy * 100))

