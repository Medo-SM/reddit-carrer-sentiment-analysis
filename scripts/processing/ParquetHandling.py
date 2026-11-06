import yaml
import os
import pyarrow.parquet as pq   
import pyarrow as pa
import pandas as pd

class ParquetHandling:
#Load the configuration file
    with open("../../config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_data_path = "../../"+config["paths"]["raw_data_dir"] 

    def getParquetFromPath(path):
        """This only takes a single parameter Path"""    
        parquet_file_path = []
        for (dir_path, _, files) in os.walk(path):
            for f in files:
                if f.endswith(".parquet"):
                    parquet_file_path.append(os.path.join(dir_path, f))
        return parquet_file_path

    def toJsonSortByDate(parquet_path, cols=None, startDate = "2018-1-1", endDate = "2022-12-31"):
       for file in parquet_path: 
        file_name = (file.replace("../../data/raw/",""))
        file_name = (file_name.replace(".parquet",""))
        
        startDate = startDate
        endDate = endDate
        
        df = pd.read_parquet(file, columns= cols)
        #print(df.head)
        df_date_sorted = df[(df["created_utc"] >= startDate) & (df["created_utc"]<= endDate)]

            
        return df_date_sorted
        
pObj = ParquetHandling

path = pObj.getParquetFromPath(pObj.raw_data_path) 
df = pObj.toJsonSortByDate(path, None, "2018-1-1", "2018-1-5")
print(df.info())