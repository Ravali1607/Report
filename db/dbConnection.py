# import mysql.connector

# def get_db_connection():
#     return mysql.connector.connect(
#         host="localhost",
#         user="root",
#         password="Ravali@1607",  
#         database="report_db"
#     )
import os
import json
from hdbcli import dbapi
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    try:
        host = os.getenv("HANA_HOST")
        port = int(os.getenv("HANA_PORT", "443"))
        user = os.getenv("HANA_USER")
        password = os.getenv("HANA_PASSWORD")

        if not host or not user:
            
            return dbapi.connect(
            address="a99c4b45-cfe7-4d55-8293-ec6a4181e9f5.hana.prod-us10.hanacloud.ondemand.com",
            port=443,
            user="SBPTECHTEAM",
            password="Sbpcorp@25"
            )
        print("Connecting to HANA:", host)

        return dbapi.connect(
            address=host,
            port=port,
            user=user,
            password=password
        )

    except Exception as e:
        print("DB CONNECTION ERROR:", str(e))
        raise