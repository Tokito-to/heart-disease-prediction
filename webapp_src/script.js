const form = document.getElementById('predict-form');
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(form);
    const data = {};
    formData.forEach((value, key) => {
        data[key] = value;
    });

    const resultBox = document.getElementById('result');
    resultBox.className = "result-box";

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        const resultBox = document.getElementById('result');
        resultBox.className = "result-box";

        if (response.ok) {
            resultBox.textContent = result.msg;

            if (result.prediction === 0) {
                resultBox.classList.add('success');
            } else if (result.prediction === 1) {
                resultBox.classList.add('error');
            } else {
                resultBox.classList.add('error');
            }
        } else {
            resultBox.textContent = "Prediction failed.";
            resultBox.classList.add('error');
        }
    } catch (error) {
        resultBox.textContent = "Prediction failed.";
        resultBox.classList.add('error');
    }

});
