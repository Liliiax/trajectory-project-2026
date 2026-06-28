RENAME_COLUMNS = {'ID': 'student_id', 'Образовательная программа': 'program', 'Уровень образования': 'education_level',
                    'Учебный год': 'academic_year', 'Тип места': 'place_type', 'Номер курса': 'course', 'Дисциплина': 'discipline',
                    'Модуль': 'module', 'Оценка': 'grade_10', 'Тип сдачи': 'exam_type', 'Причина отсутствия': 'absence_status',
                    'Сложность': 'difficulty_avg_score'}

INV_RENAME_COLUMNS = {'student_id': 'ID', 'discipline': 'Дисциплина', 'grade_10': 'Последняя оценка', 'exam_type': 'Последняя сдача',
                      'predicted_grade': 'Следующая оценка', 'predicted_attempt': 'Следующая попытка', 'status': 'Статус'}