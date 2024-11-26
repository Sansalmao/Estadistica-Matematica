import pandas as pd


class SingletonMeta(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class CSVReader(metaclass=SingletonMeta):
    _instance = None
    data = list()
    headers = list()
    file = ""

    def __init__(self, path: str):
        try:
            file = pd.read_csv(path)
            self.data = file.to_dict("records")
            self.headers = file.head().columns
            self.file = file
        except FileNotFoundError:
            print("Error: File not found!")
        except pd.errors.ParserError as e:
            print(f"Error parsing CSV: {e}")
        except Exception as e:
            print(f"Error not controlled: {e}")


csv = CSVReader("./src/lib/datos_.csv")

if __name__ == "__main__":

    csv = CSVReader("./src/lib/datos_.csv")
    print(csv.headers)
    print(csv.data)
