import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class TitanicDataset(Dataset):

    def __init__(self, csv_file):

        df = pd.read_csv(csv_file)

        df = df[[
            "Survived",
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked"
        ]]

        df["Age"].fillna(df["Age"].median(), inplace=True)
        df["Embarked"].fillna("S", inplace=True)

        df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
        df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

        y = df["Survived"].values
        X = df.drop("Survived", axis=1).values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]