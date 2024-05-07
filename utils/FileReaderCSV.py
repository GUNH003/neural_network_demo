import pandas


class FileReaderCSV:
    def __init__(self, filePath: str) -> None:
        self.filePath = filePath

    def readCSV(self) -> pandas.DataFrame:
        return pandas.read_csv(self.filePath, header=None, delimiter=",")
