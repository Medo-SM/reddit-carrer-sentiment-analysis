import re
import yaml
import os
import pyarrow.parquet as pq   
import pyarrow as pa
import pandas as pd
import html

class ParquetHandling:
    def __init__(self, path):
        self.path = path

    def getParquetFromPath(self, path):
        """This only takes a single parameter Path"""    
        parquet_file_path = []
        for (dir_path, _, files) in os.walk(path):
            for f in files:
                if f.endswith(".parquet"):
                    parquet_file_path.append(os.path.join(dir_path, f))
        return parquet_file_path

    def toJsonSortByDate(self, parquet_path, cols=None,
                         startDate = "2018-1-1", endDate = "2022-12-31"):
       for file in parquet_path: 
        file_name = (file.replace("../../data/raw/",""))
        file_name = (file_name.replace(".parquet",""))
        
        startDate = startDate
        endDate = endDate
        
        df = pd.read_parquet(file, columns= cols)
        df_date_sorted = df[(df["created_utc"] >= startDate) & (df["created_utc"]<= endDate)]

        return df_date_sorted
        

class SentimentTextCleaner:
    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        # 1. Decode HTML entities Reddit exports often leave behind (&amp; -> &)
        text = html.unescape(text)
        # 2. Remove URLs (both raw links and markdown links [text](url))
        text = re.sub(r"http\S+|www\.\S+", "", text)
        # 3. Remove Reddit markdown formatting (*, _, >, ~) without deleting punctuation (!, ?)
        text = re.sub(r"[*_~>`#]+", "", text)
        # 4. Collapse multiple spaces/newlines into a single space
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def clean_dataframe(self, df: pd.DataFrame, col: str = "text") -> pd.DataFrame:
        df = df.copy()
        # Drop null, deleted, or bot/moderator rows (note regex=False and | for OR)
        is_noise = (
            df[col].isna()
            | df[col].isin(["[deleted]", "[removed]", ""])
            | df[col].str.contains("I am a bot", regex=False, na=False)
            | df[col].str.contains("[removed by moderator]", regex=False, na=False)
        )
        df = df[~is_noise]
        df["clean_text"] = df[col].apply(self.clean_text)
        return df[df["clean_text"] != ""]    

        
def main():
    #Load the configuration file
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_data_path = config["paths"]["raw_data_dir"] 

    pd.set_option("display.max_colwidth", None)
    pObj = ParquetHandling(raw_data_path)
    cols = ["created_utc", "text"]
    path = pObj.getParquetFromPath(pObj.path)
    df = pObj.toJsonSortByDate(path, cols, "2018-1-1", "2022-1-5")
    print(df[df["text"].str.contains("I am a bot", regex = False)])



if __name__ == "__main__":
    main()