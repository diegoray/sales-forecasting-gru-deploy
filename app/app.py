import pandas as pd
from flask import Flask, render_template, request
from keras.models import load_model


app = Flask(__name__)


# load model from single file
gru_model = load_model(
    "gru_model-bs64_hn64_month0_year2017-2021_limit400.h5")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template('index.html')
    elif request.method == "POST":
        csv_file = request.files.get("file")
        df_pred = pd.read_csv(csv_file)

        # drop unnecessary column
        X = df_pred.drop_duplicates(subset=['barcode'])
        X.fillna(0, inplace=True)
        X.drop(['barcode', 'namabarang'], axis=1, inplace=True)

        # reshape the predict dataset
        X_reshaped = X.values.reshape((X.shape[0], X.shape[1], 1))

        # predict the dataset
        model_pred = gru_model.predict(X_reshaped)

        # get back the barcode to pairing the prediction
        barcode_pred = df_pred[['barcode', 'namabarang']]
        prediction = pd.DataFrame(barcode_pred[['barcode', 'namabarang']], columns=[
                                  'barcode', 'namabarang'])
        prediction['prediction_next_month'] = pd.DataFrame(model_pred)

        # return prediction.to_html()
        # print(prediction)
        return render_template('index.html', column_names=prediction.columns.values,
                               row_data=list(prediction.values.tolist()), zip=zip)


if __name__ == "__main__":
    app.run(debug=True)
