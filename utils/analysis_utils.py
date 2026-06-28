import numpy as np
import matplotlib.pyplot as plt

def model_analysis(y_true_grade, y_pred_grade, y_true_type, y_pred_type):
        fig, axs = plt.subplots(2, 2, figsize=(18, 10))
        fig.suptitle('Model analysis', fontweight='bold')
  
        axs[0][0].set_title('Real vs. predicted values')
        axs[0][0].scatter(y_true_grade, y_pred_grade, c=np.abs(y_true_grade - y_pred_grade), cmap='summer', s=20)
        axs[0][0].plot(y_true_grade, y_true_grade, color='red', alpha=0.8)
        axs[0][0].set_xlabel('grade')
        axs[0][0].set_ylabel('prediction')
        axs[0][0].grid(alpha=0.3)
        
        axs[0][1].set_title('Prediction errors (grade)')
        errors = y_true_grade - y_pred_grade
        axs[0][1].hist(errors, bins=10, color='indianred', alpha=0.7)
        axs[0][1].axvline(0, color='red', linestyle='--')
        axs[0][1].set_xlabel('error')
        axs[0][1].set_ylabel('frequency')


        axs[1][0].set_title('Grade comparison')
        bins = np.linspace(min(y_true_grade.min(), y_pred_grade.min()), max(y_true_grade.max(), y_pred_grade.max()), 20)
        axs[1][0].hist(y_true_grade, bins=bins, alpha=0.6, label='true')
        axs[1][0].hist(y_pred_grade, bins=bins, alpha=0.6, label='predicted')
        axs[1][0].legend()
        axs[1][0].set_xlabel('grade')
        axs[1][0].set_ylabel('frequency')

        axs[1][1].set_title('Exam type comparison')
        axs[1][1].hist(y_true_type, bins=10, alpha=0.6, label='true')
        axs[1][1].hist(y_pred_type, bins=10, alpha=0.6, label='predicted')
        axs[1][1].legend()
        axs[1][1].set_ylabel('frequency')

        plt.tight_layout()
        return fig

    
def model_visualization(grades_pred, type_pred):
        fig, axs = plt.subplots(1, 2, figsize=(18, 5))
        
        grades_0 = [grade for grade in grades_pred if grade < 4]
        grades_1 = [grade for grade in grades_pred if grade >= 4]

        axs[0].hist(grades_1, bins=10,  color='mediumaquamarine', label=f'Успешная сдача: {len(grades_1)}')
        axs[0].hist(grades_0, bins=10, color='lightcoral', label=f'Неуспешная сдача: {len(grades_0)}')
        axs[0].set_xticks(np.arange(0, 11))
        axs[0].legend()

        axs[1].hist(type_pred, bins=10, color='cornflowerblue')
        plt.setp(axs[1].get_xticklabels(), fontsize=8)
        axs[1].figure.tight_layout()

        return fig


def discipline_analysis(data, discipline_name):
        if discipline_name not in data['discipline'].unique():
            print("ERROR: the discipline '{discipline_name}' is not in the data")
        
        difficulty = data.groupby(['discipline'])['difficulty_avg_score'].mean().to_dict()[discipline_name]
        median_grade = data.groupby(['discipline'])['grade_10'].median().to_dict()[discipline_name]
        avg_grade = data.groupby(['discipline'])['grade_10'].mean().to_dict()[discipline_name]

        print(f'Средняя сложность: {difficulty}')
        print(f'Средняя оценка: {avg_grade}')
        print(f'Медианная оценка: {median_grade}')
