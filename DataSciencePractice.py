from pathlib import Path
import kagglehub
import pandas as pd


#IMPORT CHOCOLATE DATA
project_dir = Path(__file__).resolve().parent

#print(project_file)

#creates the path object where we want chocolate_sales.csv to go
data_path = project_dir / "chocolate_sales.csv"
print(data_path)

#if the data path file does not already exist...
if not data_path.exists():
    print("Downloading dataset...")

    #download dataset to kagglehub cache
    path = kagglehub.dataset_download("rtatman/chocolate-bar-ratings")

    print("kagglehub path: ", path)


    #converts the kagglehub path to a path object
    path = Path(path)
    #finds the csv in the kagglehub download folder--searches through all subfolders
    csv_file = next(path.rglob("*.csv"))
    #moves the csv from the kagglehub cache to project dir
    csv_file.replace(data_path)

    print("Dataset downloaded at: ", path)

chocolate_df = pd.read_csv(data_path)

print(f"{chocolate_df.shape[0]} rows, {chocolate_df.shape[1]} vars")


