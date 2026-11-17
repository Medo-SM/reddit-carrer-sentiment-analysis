import re
import yaml
import os
import pyarrow.parquet as pq   
import pyarrow as pa
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

class ParquetHandling:

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
        

class TextHandling:
    def removeSpecialSymbols(text):
        # Data cleaning: remove special symbols and links
        text = re.sub(r'http\S+', '', text) # Remove links
        text = re.sub(r'\W+', ' ', text) # Remove special symbols
        return text

    def preprocessText(text):
        # Text preprocessing: convert to lowercase, remove punctuation, and numbers
        text = text.lower()
        text = re.sub(r'\d+', '', text) # Remove numbers
        #text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
        return text

    def tokenizeText(text):
        # Text tokenization: split the text by space
        return text.split()

    def removeStopwords(words_list):
        # Stopwords handling: remove stopwords
        stop_words = set(stopwords.words('english'))
        return [word for word in words_list if word.lower() not in
            stop_words]
    
    def removeShortWords(words_list):
        # Remove words with length less than or equal to 2
        return [word for word in words_list if len(word) > 2]

    def reassembleText(text):
        sentence = " ".join(text)
        return sentence
    
    def cleanModText(self, df: pd.DataFrame):
        filterMask = (
            (df["text"].str.contains("I am a bot"))
            & df["text"].str.contains("[removed by moderator]")
        )
        return df[~filterMask]
        

        
def main():
    #Load the configuration file
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_data_path = config["paths"]["raw_data_dir"] 

    pd.set_option("display.max_colwidth", None)
    pObj = ParquetHandling
    cols = ["created_utc", "text"]
    path = pObj.getParquetFromPath(pObj.raw_data_path) 
    df = pObj.toJsonSortByDate(path, cols, "2018-1-1", "2022-1-5")
    print(df[df["text"].str.contains("removed", regex = False)])



if __name__ == "__main__":
    main()