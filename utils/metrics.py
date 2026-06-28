def evaluate(self, data):
    data = self.fillna_data(data)
    data = self.transform_data(data)
    y_grade = data['target_grade']
    y_type = data['target_type']

    FEATURES = [
            'program', 'education_level', 'place_type', 'course',
            'absence_status', 'discipline', 'module',
            'exam_type', 'grade_10', 'difficulty_avg_score',
            'exam_type_prev', 'grade_prev', 'difficulty_prev',
            'avg_grade_prev', 'min_prev', 'max_prev'
    ]

    X = data[FEATURES]

    grade_pred = self.grade_model.predict(X)
    type_pred = self.type_model.predict(X)

    mae = mean_absolute_error(y_grade, grade_pred)
    mse = mean_squared_error(y_grade, grade_pred)
    rmse = root_mean_squared_error(y_grade, grade_pred)
    r2 = r2_score(y_grade, grade_pred)

    accuracy = accuracy_score(y_type, type_pred)
    precision = precision_score(y_type, type_pred, average='weighted')
    recall = recall_score(y_type, type_pred, average='weighted')
    f1 = f1_score(y_type, type_pred, average='weighted')

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1}