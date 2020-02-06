from categorical import CategoricalFeatures
import pandas as pd 


if __name__ == "__main__":

    df = pd.read_csv("../data/df_training_scholarjet.csv")
    df_test = pd.read_csv("../data/df_holdout_scholarjet.csv")

    train_len = len(df)

    df_test['convert_30'] = -1
    df_test['revenue_30'] = -1
    full_data = pd.concat([df, df_test])

    cols = [c for c in df.columns if c not in ['convert_30', 'revenue_30']]
    cat_feats = CategoricalFeatures(full_data, 
                                    categorical_features=cols, 
                                    encoding_type="ohe",
                                    handle_na=True)
    full_data_transformed = cat_feats.fit_transform()
    
    X = full_data_transformed[:train_len, :]
    X_test = full_data_transformed[train_len:, :]

