from flask import Flask, request, render_template
from src.Heart.pipeline.Prediction_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

# Define the home route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Validate and convert form data to CustomData object with proper numeric casting
            data = CustomData(
                age=int(request.form.get("age", 0)),
                sex=int(request.form.get("sex", 0)),
                cp=int(request.form.get("cp", 0)),
                trestbps=int(request.form.get("trestbps", 0)),
                chol=int(request.form.get("chol", 0)),
                fbs=int(request.form.get("fbs", 0)),
                restecg=int(request.form.get("restecg", 0)),
                thalach=int(request.form.get("thalach", 0)),
                exang=int(request.form.get("exang", 0)),
                oldpeak=float(request.form.get("oldpeak", 0.0)),
                slope=int(request.form.get("slope", 0)),
                ca=int(request.form.get("ca", 0)),
                thal=int(request.form.get("thal", 0))
            )

            final_data = data.get_data_as_dataframe()
            # Make prediction
            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(final_data)
            result = round(pred[0], 2)
            return render_template("result.html", final_result=result)

        except Exception as e:
            # Handle exceptions gracefully
            error_message = f"Error during prediction: {str(e)}"
            return render_template("error.html", error_message=error_message)

    else:
        # Render the initial page
        return render_template("index.html")

# Execution begins
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
