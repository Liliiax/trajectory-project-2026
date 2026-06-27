import numpy as np
import pandas as pd
from params import FEATURES


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


def prepare_data(data):
        data['grade_10'] = pd.to_numeric(data['grade_10'], errors='coerce')
        data['exam_type'] = pd.Categorical(data['exam_type'],
                                         categories=['Первая сдача', 'Пересдача по уважительной причине', 'Пересдача',
                                                     'Пересдача с комиссией'], ordered=True)

        df = data.sort_values(['student_id', 'academic_year', 'module', 'exam_type', 'discipline']).reset_index(drop=True)
        g_student = df.groupby('student_id')
        g_subject = df.groupby(['student_id', 'discipline'])

        df['grade_prev'] = g_subject['grade_10'].shift(1)
        df['difficulty_prev'] = g_subject['difficulty_avg_score'].shift(1)
        df['exam_type_prev'] = g_subject['exam_type'].shift(1)
        df['cnt_prev'] = g_student.cumcount()
        df['sum_prev'] = g_student['grade_10'].transform(lambda x: x.shift().cumsum())
        df['min_prev'] = g_student['grade_10'].transform(lambda x: x.shift().cummin())
        df['max_prev'] = g_student['grade_10'].transform(lambda x: x.shift().cummax())
        df['avg_grade_prev'] = df['sum_prev'] / df['cnt_prev']

        student_avg_grade = df.groupby('student_id')['grade_10'].mean()

        def fill_missing(row):
                if row['cnt_prev'] == 0:
                        avg = student_avg_grade.get(row['student_id'], df['grade_10'].mean())
                        return pd.Series({
                                'grade_prev': avg,
                                'exam_type_prev': 'Первая сдача',
                                'min_prev': avg,
                                'max_prev': avg,
                                'avg_grade_prev': avg
                        })
                return pd.Series({
                        'grade_prev': row['grade_prev'],
                        'exam_type_prev': row['exam_type_prev'],
                        'min_prev': row['min_prev'],
                        'max_prev': row['max_prev'],
                        'avg_grade_prev': row['avg_grade_prev']
                })

        filled = df.apply(fill_missing, axis=1)
        df['grade_prev'] = filled['grade_prev']
        df['exam_type_prev'] = filled['exam_type_prev']
        df['min_prev'] = filled['min_prev']
        df['max_prev'] = filled['max_prev']
        df['avg_grade_prev'] = filled['avg_grade_prev']

        return df[FEATURES]
