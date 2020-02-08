from categorical import CategoricalFeatures
import pandas as pd 


if __name__ == "__main__":

    df = pd.read_csv("data/df_training_scholarjet.csv")
    df_test = pd.read_csv("data/df_holdout_scholarjet.csv")

    train_len = len(df)
    print(train_len)

    df_test['convert_30'] = -1
    df_test['revenue_30'] = -1
    full_data = pd.concat([df, df_test],sort=True)

    cols = [c for c in df.columns if c not in ['convert_30', 'revenue_30']]

    objects_cols = []
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerics_cols = []

    for i in df.columns:
        if df[i].dtype == object:
            objects_cols.append(i)
        if df[i].dtype in numeric_dtypes:
            numerics_cols.append(i)


    # Cate data
    cat_feats = CategoricalFeatures(full_data[objects_cols], 
                                    categorical_features=objects_cols, 
                                    encoding_type="ohe",
                                    handle_na=True)
    full_data_transformed = cat_feats.fit_transform()
    print(numerics_cols)
    X = full_data_transformed[:train_len, :]
    X_test = full_data_transformed[train_len:, :]
