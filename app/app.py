import pandas as pd
from keras.models import load_model
from flask import Flask, render_template, request, redirect, url_for
import os
from os.path import join, dirname, realpath
import mysql.connector as connection
from re import search


app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "static/files"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Database
mydb = connection.connect(
    host="localhost", user="root", password="", database="db_gru_forecasting"
)

mycursor = mydb.cursor()

# load model from single file
gru_model = load_model("gru_model-bs64_hn64_month0_year2017-2021_limit400.h5")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def uploadFiles():
    uploaded_file = request.files["file"]
    if uploaded_file.filename != "":
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file.filename)
        uploaded_file.save(file_path)
        parseCSV(file_path)
    if search("jan", uploaded_file.filename):
        query_date = "tgl >= '2021-01-01' and tgl <= '2021-12-31'"
        prediction_date = "JANUARI 2022"
        prediction = forecastJan(query_date)
    if search("feb", uploaded_file.filename):
        query_date = "tgl >= '2021-02-01' and tgl <= '2022-01-31'"
        prediction_date = "FEBRUARI 2022"
        prediction = forecastFeb(query_date)

    return render_template(
        "index.html",
        column_names=prediction.columns.values,
        row_data=list(prediction.values.tolist()),
        prediction_date=prediction_date,
        zip=zip,
    )


def forecastJan(query_date):
    sql_select_Query = "select * from transactions"
    mycursor.execute(sql_select_Query)
    # get all records
    records = mycursor.fetchall()

    list = []
    for row in records:
        list.append(row)

    df = pd.DataFrame(
        list,
        columns=[
            "id",
            "tgl",
            "namabarang",
            "satuan",
            "qty",
            "barcode",
            "kodecustomer",
            "kodejenis",
            "jenis",
            "kodekategori",
            "kategori",
        ],
    )
    df.drop("id", inplace=True, axis=1)
    dfd = df.drop_duplicates()
    dfd["tgl"] = pd.to_datetime(dfd["tgl"])
    dfd = dfd.query(query_date).reset_index(drop=True)

    dfd["month_year"] = dfd["tgl"].apply(lambda x: x.strftime("%m-%y")).astype(str)
    dfd["month_year"] = dfd["month_year"].map(
        {
            "01-21": 0,
            "02-21": 1,
            "03-21": 2,
            "04-21": 3,
            "05-21": 4,
            "06-21": 5,
            "07-21": 6,
            "08-21": 7,
            "09-21": 8,
            "10-21": 9,
            "11-21": 10,
            "12-21": 11,
        }
    )

    # get barcode & namabarang column for grouping on next step
    df_barcode = dfd.drop(
        [
            "kategori",
            "kodekategori",
            "jenis",
            "kodejenis",
            "kodecustomer",
            "qty",
            "satuan",
            "tgl",
        ],
        axis=1,
    )
    df_pred_grouped = df_barcode.groupby("barcode").first().reset_index()
    product_list = df_pred_grouped["barcode"].to_list()
    namabarang = df_pred_grouped["namabarang"].to_list()

    test_monthly = dfd[["tgl", "month_year", "barcode", "qty"]]
    test_monthly = test_monthly.sort_values("tgl").groupby(
        ["month_year", "barcode"], as_index=False
    )
    test_monthly = test_monthly.agg({"qty": ["sum"]})
    test_monthly.columns = ["month_year", "barcode", "qty_cnt"]
    test_monthly = test_monthly.query("qty_cnt >= 0 and qty_cnt <= 400")

    test_monthly["qty_cnt_month"] = (
        test_monthly.sort_values("month_year").groupby(["barcode"])["qty_cnt"].shift(-1)
    )
    test_monthly_series = test_monthly.pivot_table(
        index=["barcode"], columns="month_year", values="qty_cnt", fill_value=0
    ).reset_index()

    selected_data = test_monthly_series[
        test_monthly_series["barcode"].isin(product_list)
    ].reset_index(drop=True)

    selected_data["namabarang"] = namabarang
    selected_data = selected_data[
        ["barcode", "namabarang", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    ]
    print(selected_data.head())
    print(selected_data.shape)

    # drop unnecessary column
    X = selected_data.drop_duplicates(subset=["barcode"])
    X.fillna(0, inplace=True)
    X.drop(["barcode", "namabarang"], axis=1, inplace=True)
    # reshape the predict dataset
    X_reshaped = X.values.reshape((X.shape[0], X.shape[1], 1))
    # predict the dataset
    model_pred = gru_model.predict(X_reshaped)
    # get back the barcode to pairing the prediction
    barcode_pred = selected_data[["barcode", "namabarang"]]
    prediction = pd.DataFrame(
        barcode_pred[["barcode", "namabarang"]], columns=["barcode", "namabarang"]
    )

    prediction["prediction_next_month"] = pd.DataFrame(model_pred.astype(int))

    return prediction


def forecastFeb(query_date):
    sql_select_Query = "select * from transactions"
    mycursor.execute(sql_select_Query)
    # get all records
    records = mycursor.fetchall()

    list = []
    for row in records:
        list.append(row)

    df = pd.DataFrame(
        list,
        columns=[
            "id",
            "tgl",
            "namabarang",
            "satuan",
            "qty",
            "barcode",
            "kodecustomer",
            "kodejenis",
            "jenis",
            "kodekategori",
            "kategori",
        ],
    )
    df.drop("id", inplace=True, axis=1)
    dfd = df.drop_duplicates()
    dfd["tgl"] = pd.to_datetime(dfd["tgl"])
    dfd = dfd.query(query_date).reset_index(drop=True)

    dfd["month_year"] = dfd["tgl"].apply(lambda x: x.strftime("%m-%y")).astype(str)
    dfd["month_year"] = dfd["month_year"].map(
        {
            "02-21": 0,
            "03-21": 1,
            "04-21": 2,
            "05-21": 3,
            "06-21": 4,
            "07-21": 5,
            "08-21": 6,
            "09-21": 7,
            "10-21": 8,
            "11-21": 9,
            "12-21": 10,
            "01-22": 11,
        }
    )

    # get barcode & namabarang column for grouping on next step
    df_barcode = dfd.drop(
        [
            "kategori",
            "kodekategori",
            "jenis",
            "kodejenis",
            "kodecustomer",
            "qty",
            "satuan",
            "tgl",
        ],
        axis=1,
    )
    df_pred_grouped = df_barcode.groupby("barcode").first().reset_index()
    product_list = df_pred_grouped["barcode"].to_list()
    namabarang = df_pred_grouped["namabarang"].to_list()

    test_monthly = dfd[["tgl", "month_year", "barcode", "qty"]]
    test_monthly = test_monthly.sort_values("tgl").groupby(
        ["month_year", "barcode"], as_index=False
    )
    test_monthly = test_monthly.agg({"qty": ["sum"]})
    test_monthly.columns = ["month_year", "barcode", "qty_cnt"]
    test_monthly = test_monthly.query("qty_cnt >= 0 and qty_cnt <= 400")

    test_monthly["qty_cnt_month"] = (
        test_monthly.sort_values("month_year").groupby(["barcode"])["qty_cnt"].shift(-1)
    )
    test_monthly_series = test_monthly.pivot_table(
        index=["barcode"], columns="month_year", values="qty_cnt", fill_value=0
    ).reset_index()

    selected_data = test_monthly_series[
        test_monthly_series["barcode"].isin(product_list)
    ].reset_index(drop=True)

    selected_data["namabarang"] = namabarang
    selected_data = selected_data[
        ["barcode", "namabarang", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    ]
    print(selected_data.head())
    print(selected_data.shape)

    # drop unnecessary column
    X = selected_data.drop_duplicates(subset=["barcode"])
    X.fillna(0, inplace=True)
    X.drop(["barcode", "namabarang"], axis=1, inplace=True)
    # reshape the predict dataset
    X_reshaped = X.values.reshape((X.shape[0], X.shape[1], 1))
    # predict the dataset
    model_pred = gru_model.predict(X_reshaped)
    # get back the barcode to pairing the prediction
    barcode_pred = selected_data[["barcode", "namabarang"]]
    prediction = pd.DataFrame(
        barcode_pred[["barcode", "namabarang"]], columns=["barcode", "namabarang"]
    )

    prediction["prediction_next_month"] = pd.DataFrame(model_pred.astype(int))

    return prediction


def parseCSV(filePath):
    col_names = [
        "tgl",
        "namabarang",
        "satuan",
        "qty",
        "barcode",
        "kodecustomer",
        "kodejenis",
        "jenis",
        "kodekategori",
        "kategori",
    ]

    csvData = pd.read_csv(filePath, names=col_names)
    for i, row in csvData.iterrows():
        sql = "INSERT IGNORE INTO transactions (tgl, namabarang, satuan, qty, barcode, kodecustomer, kodejenis, jenis, kodekategori, kategori) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        value = (
            row["tgl"],
            row["namabarang"],
            row["satuan"],
            row["qty"],
            row["barcode"],
            row["kodecustomer"],
            row["kodejenis"],
            row["jenis"],
            row["kodekategori"],
            row["kategori"],
        )
        mycursor.execute(sql, value)
        mydb.commit()
        print(
            i,
            row["tgl"],
            row["namabarang"],
            row["satuan"],
            row["qty"],
            row["barcode"],
            row["kodecustomer"],
            row["kodejenis"],
            row["jenis"],
            row["kodekategori"],
            row["kategori"],
        )


if __name__ == "__main__":
    app.run(debug=True)
