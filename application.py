# application.py
import sys

from src.logger import logger
from src.exception import CustomException
from src.pipeline.predict_pipeline import PredictPipeline, PatientData

from flask import Flask, request, render_template, jsonify



app = Flask(__name__)
pipeline = PredictPipeline()


@app.route('/')
def home():
    """Renders the main input form."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives form data → runs prediction → renders result.
    """
    try:
        # ── Read form inputs ───────────────────────────────
        patient = PatientData(
            HighBP               = int(request.form['HighBP']),
            HighChol             = int(request.form['HighChol']),
            BMI                  = float(request.form['BMI']),
            Smoker               = int(request.form['Smoker']),
            Stroke               = int(request.form['Stroke']),
            HeartDiseaseorAttack = int(request.form['HeartDiseaseorAttack']),
            PhysActivity         = int(request.form['PhysActivity']),
            Fruits               = int(request.form['Fruits']),
            Veggies              = int(request.form['Veggies']),
            HvyAlcoholConsump    = int(request.form['HvyAlcoholConsump']),
            GenHlth              = int(request.form['GenHlth']),
            MentHlth             = int(request.form['MentHlth']),
            PhysHlth             = int(request.form['PhysHlth']),
            DiffWalk             = int(request.form['DiffWalk']),
            Age                  = int(request.form['Age']),
            Education            = int(request.form['Education']),
            Income               = int(request.form['Income'])
        )

        logger.info(f"Form data received for prediction")

        # ── Run prediction ────────────────────────────────
        result = pipeline.predict(patient)

        logger.info(
            f"Prediction served: {result['label']} "
            f"({result['probability']}%)"
        )

        return render_template(
            'result.html',
            result=result,
            patient=request.form
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise CustomException(e, sys)


@app.route('/health')
def health():
    """Health check endpoint — for deployment monitoring."""
    return jsonify({
        'status' : 'healthy',
        'model'  : 'XGBoost Diabetes Predictor',
        'version': '1.0.0'
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)