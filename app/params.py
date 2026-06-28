CATBOOST_HYPERPARAMETRS = {'iterations': 494, 'learning_rate': 0.3788119430934142, 'loss_function': 'MAE', 'depth': 2, 'l2_leaf_reg': 38.39899891347804, 'min_data_in_leaf': 57}
LIGHTGBM_HYPERPARAMETRS = {'learning_rate': 0.05593308405682298, 'num_leaves': 49, 'max_depth': 5, 'n_estimators': 523, 'bagging_fraction': 0.7553386111732491, 'min_child_samples': 177, 'feature_fraction': 0.5700191961275134}

FEATURES = ['program', 'education_level', 'place_type', 'course', 'absence_status', 'discipline', 'module',
       'exam_type', 'grade_10', 'difficulty_avg_score', 'exam_type_prev', 'grade_prev', 'difficulty_prev',
       'avg_grade_prev', 'min_prev', 'max_prev']

RENAME_COLUMNS = {'ID': 'student_id', 'Образовательная программа': 'program', 'Уровень образования': 'education_level',
                    'Учебный год': 'academic_year', 'Тип места': 'place_type', 'Номер курса': 'course', 'Дисциплина': 'discipline',
                    'Модуль': 'module', 'Оценка': 'grade_10', 'Тип сдачи': 'exam_type', 'Причина отсутствия': 'absence_status',
                    'Сложность': 'difficulty_avg_score'}

INV_RENAME_COLUMNS = {'student_id': 'ID', 'discipline': 'Дисциплина', 'grade_10': 'Последняя оценка', 'exam_type': 'Последняя сдача',
                      'predicted_grade': 'Следующая оценка', 'predicted_attempt': 'Следующая попытка', 'status': 'Статус'}