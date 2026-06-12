import numpy as np

def fillna_data(data):
        data = data.copy()

        # берем среднее значение оценки сложности по конкретной дисциплине
        data['difficulty_avg_score'] = (data.groupby('discipline')['difficulty_avg_score'].transform(lambda x: x.fillna(x.mean())))
        data['difficulty_avg_score'] = (data['difficulty_avg_score'].fillna(data['difficulty_avg_score'].mean()))
        
        # лаги:
        # тип экзамена: группируем по студентам и заполняем самым частым типом для каждого студента отдельно, если 
        # пропуски остались, заполняем самым частым типом для дисциплины, а потом самым частым типом глобально для всех студентов 
        data['exam_type_prev'] = (data.groupby('student_id')['exam_type_prev'].transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else np.nan)))
        data['exam_type_prev'] = (data.groupby('discipline')['exam_type_prev'].transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else np.nan)))
        data['exam_type_prev'] = (data['exam_type_prev'].fillna(data['exam_type_prev'].mode()[0]))

        # оценка: группируем по студентам и заполняем медианным значением для каждого студента отдельно, если 
        # пропуски остались, заполняем медианой по дисциплине, а потом уже глобальным медианным значением для всех студентов 
        data['grade_prev'] = (data.groupby('student_id')['grade_prev'].transform(lambda x: x.fillna(x.median())))
        data['grade_prev'] = (data.groupby('discipline')['grade_prev'].transform(lambda x: x.fillna(x.median())))
        data['grade_prev'] = (data['grade_prev'].fillna(data['grade_prev'].median()))

        # берем среднее значение оценки сложности по конкретной дисциплине
        data['difficulty_prev'] = (data.groupby('discipline')['difficulty_prev'].transform(lambda x: x.fillna(x.mean())))
        data['difficulty_prev'] = (data['difficulty_prev'].fillna(data['difficulty_prev'].mean()))
        
        return data
