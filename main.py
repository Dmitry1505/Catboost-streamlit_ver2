import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go
import time
import os

# Функция для загрузки данных
def load_data(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1]
    if file_extension == 'csv':
        df = pd.read_csv(uploaded_file)
    elif file_extension in ['xls', 'xlsx']:
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        return None
    return df

# Функция для обучения модели
def train_model(df, features, target_column, test_size=0.2, iterations=1000, learning_rate=0.1, depth=6, l2_leaf_reg=3.0):
    X = df[features]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Определяем тип задачи
    unique_classes = y.nunique()
    if unique_classes == 2:
        model_type = 'Binary'
        eval_metric = 'Logloss'
    else:
        model_type = 'Multiclass'
        eval_metric = 'MultiClass'

    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        eval_metric=eval_metric,
        loss_function=eval_metric,
        verbose=0
    )

    # Измеряем время начала обучения
    start_time = time.time()

    model.fit(X_train, y_train, eval_set=(X_test, y_test), plot=True)

    # Измеряем время конца обучения
    end_time = time.time()
    training_time = end_time - start_time

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    evals_result = model.get_evals_result()
    feature_importances = model.get_feature_importance()

    return model, accuracy, evals_result, feature_importances, eval_metric, training_time

# Функция для загрузки модели
def load_model(model_file):
    with open("uploaded_model.cbm", "wb") as f:
        f.write(model_file.getbuffer())
    model = CatBoostClassifier()
    model.load_model("uploaded_model.cbm")
    return model

# Интерфейс Streamlit
def main():
    st.title("CatBoost Classifier Training Interface with Hyperparameter Tuning")

    mode = st.selectbox("Select mode", ["Train new model", "Load existing model"])

    if mode == "Train new model":
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xls", "xlsx"])
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if df is not None:
                st.write("Data preview:")
                st.write(df.head())

                all_columns = df.columns.tolist()
                target_column = st.selectbox("Select target column", options=all_columns)
                feature_columns = st.multiselect("Select feature columns", options=all_columns, default=[col for col in all_columns if col != target_column])

                test_size = st.slider("Test size", min_value=0.1, max_value=0.5, value=0.2, step=0.1)
                iterations = st.slider("Iterations", min_value=100, max_value=1000, value=1000, step=100)
                learning_rate = st.slider("Learning rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
                depth = st.slider("Depth", min_value=1, max_value=10, value=6, step=1)
                l2_leaf_reg = st.slider("L2 regularization (l2_leaf_reg)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)

                if st.button("Train Model"):
                    model, accuracy, evals_result, feature_importances, eval_metric, training_time = train_model(df, feature_columns, target_column, test_size, iterations, learning_rate, depth, l2_leaf_reg)
                    st.write(f"Model Accuracy: {accuracy}")
                    st.write(f"Training Time: {training_time:.2f} seconds")

                    # Сохранение модели
                    model_path = "catboost_model.cbm"
                    model.save_model(model_path)
                    st.success(f"Model trained and saved as {model_path}")

                    # Добавление кнопки для скачивания модели
                    with open(model_path, "rb") as file:
                        st.download_button(label="Download Model", data=file, file_name=model_path, mime='application/octet-stream')

                    # График потерь
                    train_loss = evals_result['learn'][eval_metric]
                    test_loss = evals_result['validation'][eval_metric]
                    iterations_range = range(1, len(train_loss) + 1)

                    fig_loss = px.line(x=iterations_range, y=[train_loss, test_loss], labels={'x': 'Iterations', 'y': eval_metric}, title='Training and Validation Loss Over Iterations')
                    fig_loss.update_layout(yaxis_title=eval_metric, xaxis_title='Iterations')
                    fig_loss.data[0].name = 'Train Loss'
                    fig_loss.data[1].name = 'Validation Loss'

                    st.plotly_chart(fig_loss)

                    # График значимости признаков
                    feature_importance_df = pd.DataFrame({
                        'Feature': feature_columns,
                        'Importance': feature_importances
                    }).sort_values(by='Importance', ascending=False)

                    fig_importance = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h', title='Feature Importance')
                    fig_importance.update_layout(yaxis_title='Feature', xaxis_title='Importance')

                    st.plotly_chart(fig_importance)

    elif mode == "Load existing model":
        model_file = st.file_uploader("Choose a CatBoost model file", type=["cbm"])
        if model_file is not None:
            model = load_model(model_file)
            st.success("Model loaded successfully")

            prediction_file = st.file_uploader("Choose a CSV or Excel file for predictions", type=["csv", "xls", "xlsx"])
            if prediction_file is not None:
                prediction_data = load_data(prediction_file)
                if prediction_data is not None:
                    st.write("Prediction data preview:")
                    st.write(prediction_data.head())

                    predictions = model.predict(prediction_data)
                    if len(predictions.shape) > 1 and predictions.shape[1] == 1:
                        predictions = predictions.flatten()

                    prediction_data['Predictions'] = predictions

                    st.write("Predictions:")
                    st.write(prediction_data)

                    prediction_data.to_csv("predictions.csv", index=False)
                    st.success("Predictions saved to predictions.csv")

if __name__ == "__main__":
    main()
