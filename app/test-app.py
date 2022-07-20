from lib2to3.pgen2 import driver
import mysql.connector
import pandas as pd
import sqlalchemy as sq
from keras.models import load_model

try:
    connection = mysql.connector.connect(
        host="localhost", user="root", password="", database="db_gru_forecasting"
    )

    gru_model = load_model("gru_model-bs64_hn64_month0_year2017-2021_limit400.h5")

    sql_select_Query = "select * from transactions"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    # get all records
    records = cursor.fetchall()

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
    # print(list)
    df.drop("id", inplace=True, axis=1)
    dfd = df.drop_duplicates()
    dfd["tgl"] = pd.to_datetime(dfd["tgl"])
    dfd = dfd.query("tgl >= '2021-02-01' and tgl <= '2022-01-31'").reset_index(
        drop=True
    )

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

    print(prediction)
    print(df.shape)
    print(dfd.shape)
    # print(dfd.info())
    # print(dfd.head())
    # print(dfd.tail())
    print(f"Min date from data set: {dfd['tgl'].min().date()}")
    print(f"Max date from data set: {dfd['tgl'].max().date()}")
except mysql.connector.Error as e:
    print("Error reading data from MySQL table", e)
finally:
    if connection.is_connected():
        connection.close()
        cursor.close()
        print("MySQL connection is closed")

# con = sq.create_engine("mysql+pymysql://root:@localhost/db_gru_forecasting")
# df = pd.read_sql("transactions", con)
# print(df)

# try:
#     mydb = connection.connect(
#         host="localhost", user="root", password="", database="db_gru_forecasting"
#     )
#     query = "Select * from transactions;"
#     df = pd.read_sql(query, mydb)
#     df = mydb.execute(query)
#     mydb.close()  # close the connection
#     print("Type:", type(df))
#     print(df)
# except Exception as e:
#     mydb.close()
#     print(str(e))
