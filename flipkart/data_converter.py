# flipkart/data_converter.py
# Code to convert Flipkart review data into Document objects that LangChain can work with.Using pandas to read CSV files.
# Assumes the CSV has columns "product_title" and "review" because those are the relevant fields.
## Each review becomes a Document with the review text as content and the product title as metadata.
## This allows for easy integration of Flipkart review data into LangChain applications.
#Imprting document from langchain_core.documents to create Document objects in easy language that means objects that LangChain can work with.
import pandas as pd
from langchain_core.documents import Document

# DataConverter class to handle the conversion of Flipkart review data into Document objects.
class DataConverter:
    #Here we are initializing the class with the file path of the CSV file containing Flipkart review data.
    def __init__(self,file_path:str):
        # Storing the file path for later use in the convert method.
        self.file_path = file_path

    def convert(self):
        # Reading the CSV file using pandas and selecting only the relevant columns: "product_title" and "review".
        df = pd.read_csv(self.file_path)[["product_title","review"]]   
        # Creating a list of Document objects from the DataFrame. Each Document contains the review text and associated product title as metadata.
        # Using list comprehension to iterate over each row in the DataFrame and create Document objects.
        # The page_content is set to the review text, and metadata includes the product title.
        # Returning the list of Document objects.
        docs = [
            Document(page_content=row['review'] , metadata = {"product_name" : row["product_title"]})
            for _, row in df.iterrows()
        ]

        return docs