document.addEventListener('DOMContentLoaded', function() {
    // Form submission handler
    const form = document.querySelector('form');
    const resultsSection = document.getElementById('results');
    const newAssessmentBtn = document.getElementById('new-assessment');

    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();

            // Show loading state
            document.getElementById('prediction-result').textContent = 'Analyzing...';
            document.getElementById('result-description').textContent = 'Our AI model is processing your data to provide a personalized assessment.';
            document.getElementById('recommendations').innerHTML = '';

            // Hide form and show results
            document.getElementById('predict').classList.add('hidden');
            resultsSection.classList.remove('hidden');

            // Simulate API call with timeout
            setTimeout(() => {
                // For demo purposes, we'll use a random result
                const riskLevel = Math.random() > 0.5 ? 'High Risk' : 'Low Risk';
                document.getElementById('prediction-result').textContent = riskLevel;

                if (riskLevel === 'High Risk') {
                    document.getElementById('result-description').textContent = 'Our model indicates you may be at higher risk for heart disease. Please consult with a healthcare professional.';

                    const recommendations = [
                        "Schedule an appointment with a cardiologist",
                        "Monitor your blood pressure regularly",
                        "Consider dietary changes to reduce cholesterol",
                        "Engage in moderate exercise most days of the week",
                        "Reduce stress through meditation or relaxation techniques"
                    ];

                    const recommendationsList = document.getElementById('recommendations');
                    recommendations.forEach(rec => {
                        const li = document.createElement('li');
                        li.textContent = rec;
                        recommendationsList.appendChild(li);
                    });
                } else {
                    document.getElementById('result-description').textContent = 'Our model indicates you have a lower risk profile for heart disease. Continue maintaining healthy habits!';

                    const recommendations = [
                        "Continue with regular physical activity",
                        "Maintain a balanced diet rich in fruits and vegetables",
                        "Schedule annual check-ups with your doctor",
                        "Monitor your family history of heart conditions",
                        "Stay aware of any changes in your health"
                    ];

                    const recommendationsList = document.getElementById('recommendations');
                    recommendations.forEach(rec => {
                        const li = document.createElement('li');
                        li.textContent = rec;
                        recommendationsList.appendChild(li);
                    });
                }
            }, 2000);
        });
    }

    if (newAssessmentBtn) {
        newAssessmentBtn.addEventListener('click', function() {
            resultsSection.classList.add('hidden');
            document.getElementById('predict').classList.remove('hidden');
            window.scrollTo({
                top: document.getElementById('predict').offsetTop - 100,
                behavior: 'smooth'
            });
        });
    }

    // Smooth scrolling for navigation
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                window.scrollTo({
                    top: target.offsetTop - 80,
                    behavior: 'smooth'
                });
            }
        });
    });
});

