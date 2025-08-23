#!/bin/bash -e

PYTHON_SCRIPT="gaann.py"
BACKUP_SCRIPT="${PYTHON_SCRIPT}.bak"

mkdir -p models/logs

# Backup Stock script before writing changes
cp $PYTHON_SCRIPT $BACKUP_SCRIPT

declare -A activations
activations["ReLU"]="model.add(ReLU())"
activations["LeakyReLU"]="model.add(LeakyReLU(negative_slope=alpha))"
activations["PReLU"]="model.add(PReLU(alpha_initializer=tf.keras.initializers.Constant(alpha)))"
activations["ELU"]="model.add(ELU())"
activations["SELU"]="model.add(Activation('selu'))"
activations["Tanh"]="model.add(Activation('tanh'))"

cleanup() {
    mv "$BACKUP_SCRIPT" "$PYTHON_SCRIPT"
    if [[ "$1" == "kill" ]]; then
        kill -9 $$
    fi
}

trap 'cleanup' EXIT
trap 'cleanup kill' INT QUIT

for act in "${!activations[@]}"; do
    logfile="'models/logs/${act}_model.csv'"
    modelfile="'models/${act}_heart_model.keras'"
    historyfile="'models/logs/${act}_model_logs.pkl'"

    # Restore Stock Script
    cp $BACKUP_SCRIPT $PYTHON_SCRIPT

    echo "=== Trying activation: $act ==="

    sed -i "s|model.add(ReLU())|${activations[$act]}|g" $PYTHON_SCRIPT
    sed -i "s|log_file = 'models/logs/ReLU_model.csv'|log_file = ${logfile}|g" $PYTHON_SCRIPT
    sed -i "s|model.save('models/ReLU_heart_model.keras')|model.save(${modelfile})|g" $PYTHON_SCRIPT
    sed -i "s|joblib.dump(log_data, 'models/logs/ReLU_model_logs.pkl')|joblib.dump(log_data, ${historyfile})|g" $PYTHON_SCRIPT

    echo "Modified script for activation: $act"
    echo "Log file -> ${logfile}"
    echo "Model save path -> ${modelfile}"
    echo "Model history path -> ${historyfile}"
    echo "------------------------------"

    python $PYTHON_SCRIPT
done

