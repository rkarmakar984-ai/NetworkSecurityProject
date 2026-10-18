from pymongo import MongoClient
uri = "mongodb+srv://rkarmakar984_db_user:Rahul009@cluster0.jzriona.mongodb.net/"
client = MongoClient(uri)
try:
    client.admin.command("ping")
    print("Connected successfully")
    client.close()

except Exception as e:
    raise Exception(
        "The following error occurred: ", e)