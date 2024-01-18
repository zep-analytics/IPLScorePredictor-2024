from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@ app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        batting_team = request.form['batting_team']
        bowling_team = request.form['bowling_team']
        selected_city = request.form['selected_city']
        target = int(request.form['target'])
        score = int(request.form['score'])
        balls_left = int(request.form['balls_left'])  # Update this line
        wickets = int(request.form['wickets'])

        runs_left = target - score
        wickets_remaining = 10 - wickets
        overs_completed = (120 - balls_left) / 6  # Calculate overs_completed from balls_left
        crr = score / overs_completed
        rrr = runs_left / (balls_left / 6)

        input_data = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets_remaining': [wickets_remaining],
            'total_run_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        pipe = pickle.load(open('ra_pipe.pkl', 'rb'))
        result = pipe.predict_proba(input_data)

        win_probability = round(result[0][1] * 100)
        loss_probability = round(result[0][0] * 100)

        return render_template('result.html', batting_team=batting_team, bowling_team=bowling_team, win_probability=win_probability, loss_probability=loss_probability)


if __name__ == '__main__':
    app.run(debug=True)
