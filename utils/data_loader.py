import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data():
    """
    Load and preprocess the dataset.
    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    # Load the dataset
    df = pd.read_csv('survey_lung_cancer.csv')

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values (if any)
    df = df.dropna()

    # Encode categorical variables
    label_encoder = LabelEncoder()
    if 'GENDER' in df.columns:
        df['GENDER'] = label_encoder.fit_transform(df['GENDER'])
    if 'LUNG_CANCER' in df.columns:
        df['LUNG_CANCER'] = label_encoder.fit_transform(df['LUNG_CANCER'])

    # Replace "YES"/"NO" with 2/1 for consistency
    df.replace({'YES': 2, 'NO': 1}, inplace=True)

    return df