"""
Flask Web Application for Churn Prediction
"""

from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model and scaler
MODEL_PATH = os.path.join('models', 'best_model.pkl')
SCALER_PATH = os.path.join('models', 'scaler.pkl')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and scaler loaded successfully!")
except Exception as e:
    print(f"⚠️  Could not load model: {e}")
    model = None
    scaler = None

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Churn Predictor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            max-width: 700px;
            width: 100%;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5);
        }

        h1 {
            color: #e94560;
            text-align: center;
            font-size: 2rem;
            margin-bottom: 10px;
        }

        .subtitle {
            color: rgba(255, 255, 255, 0.6);
            text-align: center;
            margin-bottom: 30px;
            font-size: 0.95rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 25px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.85rem;
            margin-bottom: 6px;
            font-weight: 500;
        }

        input, select {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            padding: 10px 14px;
            color: white;
            font-size: 0.9rem;
            transition: all 0.3s ease;
        }

        input:focus, select:focus {
            outline: none;
            border-color: #e94560;
            background: rgba(255, 255, 255, 0.15);
        }

        select option {
            background: #1a1a2e;
            color: white;
        }

        .predict-btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #e94560, #c62a47);
            border: none;
            border-radius: 10px;
            color: white;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            letter-spacing: 1px;
        }

        .predict-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(233, 69, 96, 0.4);
        }

        .result {
            margin-top: 25px;
            padding: 20px;
            border-radius: 12px;
            display: none;
        }

        .result.high-risk {
            background: rgba(233, 69, 96, 0.2);
            border: 1px solid #e94560;
        }

        .result.medium-risk {
            background: rgba(255, 165, 0, 0.2);
            border: 1px solid orange;
        }

        .result.low-risk {
            background: rgba(0, 200, 100, 0.2);
            border: 1px solid #00c864;
        }

        .result-title {
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .result-probability {
            font-size: 2.5rem;
            font-weight: 800;
            margin: 10px 0;
        }

        .result-strategy {
            margin-top: 15px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
        }

        .result-strategy h3 {
            color: rgba(255, 255, 255, 0.9);
            margin-bottom: 10px;
        }

        .result-strategy ul {
            list-style: none;
            padding: 0;
        }

        .result-strategy ul li {
            color: rgba(255, 255, 255, 0.7);
            padding: 4px 0;
            font-size: 0.9rem;
        }

        .result-strategy ul li::before {
            content: "✓ ";
            color: #00c864;
        }

        .loading {
            text-align: center;
            color: rgba(255, 255, 255, 0.6);
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Churn Predictor</h1>
        <p class="subtitle">Predict customer churn probability using Machine Learning</p>

        <form id="churnForm">
            <div class="form-grid">
                <div class="form-group">
                    <label>📅 Tenure (months)</label>
                    <input type="number" id="tenure" value="12" min="0" max="72" required>
                </div>
                <div class="form-group">
                    <label>💰 Monthly Charges ($)</label>
                    <input type="number" id="monthlyCharges" value="65" step="0.01" required>
                </div>
                <div class="form-group">
                    <label>💳 Total Charges ($)</label>
                    <input type="number" id="totalCharges" value="780" step="0.01" required>
                </div>
                <div class="form-group">
                    <label>👤 Gender</label>
                    <select id="gender">
                        <option value="1">Male</option>
                        <option value="0">Female</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>👴 Senior Citizen</label>
                    <select id="seniorCitizen">
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>📄 Contract Type</label>
                    <select id="contract">
                        <option value="0">Month-to-Month</option>
                        <option value="1">One Year</option>
                        <option value="2">Two Year</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>🌐 Internet Service</label>
                    <select id="internetService">
                        <option value="0">DSL</option>
                        <option value="1">Fiber Optic</option>
                        <option value="2">No</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>🔒 Online Security</label>
                    <select id="onlineSecurity">
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>🛠️ Tech Support</label>
                    <select id="techSupport">
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>📱 Phone Service</label>
                    <select id="phoneService">
                        <option value="1">Yes</option>
                        <option value="0">No</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>💑 Partner</label>
                    <select id="partner">
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>👨‍👩‍👧 Dependents</label>
                    <select id="dependents">
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>
            </div>

            <button type="submit" class="predict-btn">
                🔮 PREDICT CHURN PROBABILITY
            </button>
        </form>

        <div class="loading" id="loading">
            <p>🤖 Analyzing customer data...</p>
        </div>

        <div class="result" id="result">
            <div class="result-title" id="resultTitle"></div>
            <div class="result-probability" id="resultProbability"></div>
            <p id="resultMessage"></p>
            <div class="result-strategy" id="resultStrategy"></div>
        </div>
    </div>

    <script>
        document.getElementById('churnForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';

            const data = {
                tenure: parseInt(document.getElementById('tenure').value),
                MonthlyCharges: parseFloat(document.getElementById('monthlyCharges').value),
                TotalCharges: parseFloat(document.getElementById('totalCharges').value),
                gender: parseInt(document.getElementById('gender').value),
                SeniorCitizen: parseInt(document.getElementById('seniorCitizen').value),
                Contract: parseInt(document.getElementById('contract').value),
                InternetService: parseInt(document.getElementById('internetService').value),
                OnlineSecurity: parseInt(document.getElementById('onlineSecurity').value),
                TechSupport: parseInt(document.getElementById('techSupport').value),
                PhoneService: parseInt(document.getElementById('phoneService').value),
                Partner: parseInt(document.getElementById('partner').value),
                Dependents: parseInt(document.getElementById('dependents').value)
            };

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();
                displayResult(result);
            } catch (error) {
                console.error('Error:', error);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        });

        function displayResult(result) {
            const resultDiv = document.getElementById('result');
            const probability = result.churn_probability;
            const percentage = (probability * 100).toFixed(1);

            resultDiv.style.display = 'block';

            let riskClass, title, message, strategies;

            if (probability > 0.7) {
                riskClass = 'high-risk';
                title = '🚨 HIGH RISK - Will Likely Churn';
                message = 'This customer requires IMMEDIATE attention!';
                strategies = [
                    'Call customer within 24 hours',
                    'Offer 30% discount for 3 months',
                    'Free upgrade to premium service',
                    'Assign dedicated account manager',
                    'Resolve any outstanding issues immediately'
                ];
            } else if (probability > 0.4) {
                riskClass = 'medium-risk';
                title = '⚠️ MEDIUM RISK - Monitor Closely';
                message = 'This customer needs proactive engagement.';
                strategies = [
                    'Send personalized retention email',
                    'Offer 15% loyalty discount',
                    'Provide free add-on service for 1 month',
                    'Schedule a satisfaction call this week',
                    'Enroll in loyalty rewards program'
                ];
            } else {
                riskClass = 'low-risk';
                title = '✅ LOW RISK - Customer Likely to Stay';
                message = 'Maintain positive relationship with this customer.';
                strategies = [
                    'Include in regular newsletters',
                    'Enroll in loyalty points program',
                    'Offer early access to new features',
                    'Schedule annual review call',
                    'Ask for referrals and reviews'
                ];
            }

            resultDiv.className = `result ${riskClass}`;

            document.getElementById('resultTitle').innerHTML = title;
            document.getElementById('resultTitle').style.color =
                probability > 0.7 ? '#e94560' : probability > 0.4 ? 'orange' : '#00c864';

            document.getElementById('resultProbability').innerHTML =
                `${percentage}% Churn Risk`;
            document.getElementById('resultProbability').style.color =
                probability > 0.7 ? '#e94560' : probability > 0.4 ? 'orange' : '#00c864';

            document.getElementById('resultMessage').innerHTML = message;
            document.getElementById('resultMessage').style.color = 'rgba(255,255,255,0.7)';

            document.getElementById('resultStrategy').innerHTML = `
                <h3>💡 Recommended Retention Strategies:</h3>
                <ul>
                    ${strategies.map(s => `<li>${s}</li>`).join('')}
                </ul>
            `;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        features = np.array([[
            data.get('gender', 1),
            data.get('SeniorCitizen', 0),
            data.get('Partner', 0),
            data.get('Dependents', 0),
            data.get('tenure', 12),
            data.get('PhoneService', 1),
            data.get('MultipleLines', 1),
            data.get('InternetService', 1),
            data.get('OnlineSecurity', 0),
            data.get('OnlineBackup', 0),
            data.get('DeviceProtection', 0),
            data.get('TechSupport', 0),
            data.get('StreamingTV', 0),
            data.get('StreamingMovies', 0),
            data.get('Contract', 0),
            data.get('PaperlessBilling', 1),
            data.get('PaymentMethod', 2),
            data.get('MonthlyCharges', 65),
            data.get('TotalCharges', 780)
        ]])

        if model and scaler:
            features_scaled = scaler.transform(features)
            churn_probability = float(model.predict_proba(features_scaled)[0][1])
            churn_prediction = int(model.predict(features_scaled)[0])
        else:
            # Demo mode if model not loaded
            import random
            churn_probability = random.uniform(0.1, 0.9)
            churn_prediction = 1 if churn_probability > 0.5 else 0

        return jsonify({
            'churn_probability': churn_probability,
            'churn_prediction': churn_prediction,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
