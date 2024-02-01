from django.shortcuts import render
import joblib
# Create your views here.

def heart_predictor(request):
    if request.method == 'POST':
        age = request.POST['age']
        sex = request.POST['sex']
        cp = request.POST['cp']
        trestbps = request.POST['trestbps']
        chol = request.POST['chol']
        fbs = request.POST['fbs']
        restecg = request.POST['restecg']
        thalach = request.POST['thalach']
        exang = request.POST['exang']
        oldpeak = request.POST['oldpeak']
        slope = request.POST['slope']
        ca = request.POST['ca']
        thal = request.POST['thal']
        model2 = joblib.load('train_heart_disease_model.joblib')
        y_pred = model2.predict(
            [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        if y_pred[0] == 0:
            y_pred = 'The Person does not have a Heart Disease'
        elif y_pred[0] == 1:
            y_pred = 'The Person has Heart Disease'
        else:
            y_pred = 'Error'
        return render(request, 'index.html', {'result': y_pred})
    return render(request, 'index.html')


def refresh(request):
    return render(request, 'main.html')