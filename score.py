import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from rdkit import Chem
import mordred
from mordred import Calculator, descriptors

score = 0.0

def smiles_to_mol(smiles):
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
    return m

def featurize(dataset,features):
    df = pd.read_csv(dataset)
    y_test = np.array(df['Solubility'])
    calc = mordred.Calculator()
    calc.register(features)
    df = df['SMILES'].apply(smiles_to_mol)
    X_test = np.array(calc.pandas(df))
    return X_test, y_test

def test(features,model):
    global done, score
    X_test, y_test = featurize('../../Data/Solubility/dataset-E.csv',features)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    if score < 0:
        print('Your score is negative. Something is wrong.')
        return
    done = True
    score = mae
    return mae
