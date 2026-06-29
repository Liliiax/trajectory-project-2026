import streamlit as st
import os
from model_final import Trajectory
from disciplines import DISCIPLINES
from columns import RENAME_COLUMNS, INV_RENAME_COLUMNS
from data_utils import prepare_data
from datetime import datetime
import pandas as pd
import numpy as np

# текущий год
current_year = datetime.now().year

# текущая директория
current_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(current_dir, "../model/trajectory_model.pkl")
logo_path = os.path.join(current_dir, "img/hse_logo.png")
model_output_path =  os.path.join(current_dir, "img/model_output.png")

model = Trajectory.load(model_path)
st.set_page_config(page_title="Оценка траектории обучения", layout="centered")

# инициализация session_state
if "step" not in st.session_state:
    st.session_state.step = 1
if "data_input_method" not in st.session_state:
    st.session_state.data_input_method = None
if "data" not in st.session_state:
    # введенные данные (в случае ручного ввода)
    st.session_state.data = [
        {"id": 1, "module_num": 1, "grade": 5, "difficulty": 2.5, "attempt": "Первая сдача", "absence_status": "\\N", "min_prev": 4, "max_prev": 8}
    ]

if "year" not in st.session_state:
    st.session_state.year = current_year
if "program" not in st.session_state:
    st.session_state.program = None
if "place_type" not in st.session_state:
    st.session_state.place_type = None
if "education_level" not in st.session_state:
    st.session_state.education_level = None
if "course" not in st.session_state:
    st.session_state.course = None
if "discipline" not in st.session_state:
    st.session_state.discipline = None


st.markdown("""<style>
h1 { text-align: center; }
.center-text { text-align: center; color: #666; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>""", unsafe_allow_html=True)


# лого и название
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    st.image(logo_path, width=1800)

st.markdown("""
<style>
.block-container {
    padding-top: 5rem !important;
    padding-bottom: 5rem !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align: center; color: #1F1F1F; font-size: 32px; font-weight: 600; margin-top: -30px;'>
    Оценка траектории обучения
</h1>
""", unsafe_allow_html=True)

st.markdown('<div style="padding: -20px 0;"></div>', unsafe_allow_html=True)
st.markdown("---")


# приветственная страница
if st.session_state.step == 1:
    st.markdown("""
    <p style='font-size: 16px; font-weight: 500; margin-top: 0;'>
            Модель прогнозирует успешность освоения дисциплины студентом — 
            <span style='color: #1a73e8;'>оценку</span> и 
            <span style='color: #1a73e8;'>попытку сдачи</span>.
    </p>
    
    <div style='display: flex; gap: 15px; margin: 16px 0; flex-wrap: wrap;'>
        <div style='flex: 1; min-width: 180px; background: white; padding: 12px 16px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);'>
            <b>CSV-файл</b>
            <div style='color: #666; font-size: 13px;'>массовый прогноз по группе</div>
        </div>
        <div style='flex: 1; min-width: 180px; background: white; padding: 12px 16px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);'>
            <b>Ручной ввод</b>
            <div style='color: #666; font-size: 13px;'>точечный прогноз для студента</div>
        </div>
    </div>
    
    <div style='background: white; padding: 14px 18px; border-radius: 8px; border: 1px solid #e8eaed; margin: 12px 0;'>
        <b style='font-size: 14px;'>Результаты:</b>
        <ul style='margin: 6px 0 0 0; padding-left: 20px; font-size: 14px; color: #444;'>
            <li>Персональный прогноз (оценка + попытка)</li>
            <li>Групповая статистика: гистограммы оценок, соотношение попыток</li>
            <li>Результаты в формате csv</li>
        </ul>
    </div>
    
    <div style='background: #e8f0fe; padding: 10px 16px; border-radius: 8px; font-size: 16px; color: #174ea6;'>
        Помогает выявлять академические риски и адаптировать учебный процесс.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding: 10px 0;"></div>', unsafe_allow_html=True)
    st.image(model_output_path, width=1990)


    st.header("Входные данные")

    # загрузка готового csv с данными
    st.markdown("""
    <p style='text-align: left; font-size: 16px; color: #666;'>
    Загрузить в формате CSV
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background: #f8f9fa; padding: 16px 20px; border-radius: 10px; margin: 10px 0; border: 1px solid #e0e0e0;'>
        <p style='font-weight: 400; font-size: 13px; margin: 0 0 10px 0; color: #666;'>
            Требуемые входные данные (обязательно с такими названиями колонок):
        </p>
        <ul style='margin: 0; padding-left: 20px; font-size: 14px; color: #444; line-height: 1.8;'>
            <li><b>ID</b> — идентификатор студента</li>
            <li><b>Образовательная программа</b> — название программы обучения</li>
            <li><b>Уровень образования</b> — бакалавриат / магистратура</li>
            <li><b>Учебный год</b></li>
            <li><b>Тип места</b> — бюджетное / коммерческое / целевое или др.</li>
            <li><b>Номер курса</b> — 1–4 (бакалавриат) или 1–2 (магистратура)</li>
            <li><b>Дисциплина</b> — название предмета</li>
            <li><b>Модуль</b> — от 1 до 4</li>
            <li><b>Тип сдачи</b> — первая сдача / пересдача / пересдача с комиссией/ пересдача по уважительной причине</li>
            <li><b>Оценка</b> — оценка за указанный модуль от 0 до 10</li>
            <li><b>Причина отсутствия</b> — уважительная / неуважительная / N (в случае присутствия на экзамене)</li>
            <li><b>Сложность</b> — оценка от 0 до 5 (насколько сложным был предмет для студента)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="Загрузить",
        type=['csv'],
        key="csv_uploader",
        label_visibility='hidden'
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Загрузка завершена ({len(df)} записей)")
            st.dataframe(df.head())

            if st.button("Использовать эти данные", key="use_csv"):
                st.session_state.student_data = df
                st.session_state.data_input_method = 'csv'
                st.session_state.step = 2
                st.rerun()
        except Exception as e:
            st.error(f"Ошибка при чтении файла: {e}")

    # ввод данных вручную
    st.markdown('<div style="padding: 10px 0;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: left; font-size: 16px; color: #666;'>
    Ввести данные вручную
    </p>
    """, unsafe_allow_html=True)

    if st.button("Ввод данных", key="manual_input", use_container_width=True):
        st.session_state.data_input_method = 'manual'
        st.session_state.step = 3
        st.rerun()


# в случае загрузки готового csv-файла
if st.session_state.step == 2:
    df = st.session_state.student_data.copy()
    st.info(f"Данные загружены: {len(df)} записей, {df['ID'].nunique()} студентов")
    st.dataframe(df.head())

    # подготовка данных для модели
    df.rename(columns=RENAME_COLUMNS, inplace=True)
    df['absence_status'] = df['absence_status'].replace({
            'Уважительная': 'valid',
            'Неуважительная': 'invalid',
            'N': '\\N'
        })


    data = prepare_data(df)

    # предсказания
    with st.spinner("Модель обрабатывает данные..."):
        pred_grade, pred_type = model.predict(data, fillna=True)

    data['predicted_grade'] = np.round(pred_grade).astype(int)
    data['predicted_attempt'] = pred_type

    st.markdown('<div style="padding: 10px 0;"></div>', unsafe_allow_html=True)
    st.markdown("### Основные показатели")

    col1, col2, col3 = st.columns(3, gap='small')

    with col1:
        success_rate = len([g for g in pred_grade if g >= 4]) / len(pred_grade) * 100
        st.metric(
            "Успешная сдача",
            f"{success_rate:.1f}%",
            delta=f"{success_rate - 50:.1f}%" if success_rate != 0 else None,
            delta_color="normal"
        )

    with col2:
        avg_grade = np.mean(pred_grade)
        st.metric(
            "Средняя оценка по группе",
            f"{avg_grade:.1f} / 10",
            delta=f"{avg_grade - 5:.1f}" if avg_grade != 0 else None
        )

    with col3:
        min_grade = np.min(pred_grade)
        max_grade = np.max(pred_grade)
        st.metric(
            "Диапазон оценок",
            f"{min_grade:.0f} – {max_grade:.0f}"
        )

    st.markdown('<div style="padding: 10px 0;"></div>', unsafe_allow_html=True)
    st.markdown("### Визуализация результатов")

    fig = model.visualize(pred_grade, pred_type)
    st.pyplot(fig)

    # таблица с результатами
    with st.expander("Подробные результаты по студентам", expanded=False):
        cols = ['student_id', 'discipline', 'grade_10', 'exam_type', 'predicted_grade', 'predicted_attempt']

        data['status'] = data['predicted_grade'].apply(
            lambda x: 'Успешно' if x >= 4 else 'Неуспешно'
        )
        results = data[cols + ['status']].copy()
        results.rename(columns=INV_RENAME_COLUMNS, inplace=True)

        st.dataframe(
            results.style.background_gradient(
                subset=['Следующая оценка'],
                cmap='RdYlGn',
                vmin=0,
                vmax=10
            ),
            height=400
        )

    low_grades = data[data['predicted_grade'] < 4]
    if len(low_grades) > 0:
        st.warning(f"Количество студентов в группе риска: {len(low_grades)}")
    else:
        st.success("Все студенты имеют прогноз выше 4")


    st.markdown('<div style="padding: 10px 0;"></div>', unsafe_allow_html=True)
    st.markdown("### Экспорт результатов")


    col1, col2 = st.columns([3, 5])
    with col1:
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Скачать результаты в CSV",
            data=csv,
            file_name="predictions_results.csv",
            mime="text/csv",
            use_container_width=True
        )


    st.markdown("---")

    col_1, col_2 = st.columns([8, 1])
    with col_2:
        if st.button("←", key="back_arrow"):
            st.session_state.step = 1
            st.rerun()


# в случае ручного ввода данных
if st.session_state.step == 3:
    st.subheader("Введите данные")
    col1, col2 = st.columns(2)

    with col1:
        program = st.selectbox("Образовательная программа", list(DISCIPLINES.keys()), index=None, placeholder="Выберите программу")
        place_type = st.selectbox("Тип поступления",
            ["Бюджетные", "Коммерческие", 'Коммерческие за счет средств вуза',
            'Целевые', 'Коммерческие места для иностранных граждан', 'Без вступительных испытаний'],
            index=None, placeholder="Выберите тип поступления")

    with col2:
        education_level = st.selectbox("Уровень обучения", ["Бакалавриат", "Магистратура"], index=None, placeholder="Выберите уровень обучения")

        if education_level == "Бакалавриат":
            course_options = ["1", "2", "3", "4"]
        elif education_level == "Магистратура":
            course_options = ["1", "2"]
        else:
            course_options = []
        course = st.selectbox("Курс", course_options, index=None, placeholder="Выберите курс обучения")

    if program is None:
        discipline = st.selectbox("Дисциплина", [], index=None, placeholder="Сначала выберите программу", disabled=True)
    else:
        discipline_options = DISCIPLINES.get(program, [])
        discipline = st.selectbox("Дисциплина", discipline_options, index=None, placeholder="Выберите дисциплину")


    submitted = st.button("Далее", type="primary")

    if submitted:
        if program is None:
            st.error("Пожалуйста, выберите образовательную программу")
        elif place_type is None:
            st.error("Пожалуйста, выберите тип поступления")
        elif education_level is None:
            st.error("Пожалуйста, выберите ступень обучения")
        elif course is None:
            st.error("Пожалуйста, выберите курс")
        elif discipline is None:
            st.error("Пожалуйста, выберите дисциплину")
        else:
            st.session_state.step = 4
            st.session_state.discipline = discipline
            st.session_state.program = program
            st.session_state.course = course
            st.session_state.place_type = place_type
            st.session_state.education_level = education_level
            st.rerun()

    st.markdown("---")

    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("←", key="back_arrow"):
            st.session_state.step = 1
            st.rerun()


if st.session_state.step == 4:
    st.subheader(f"Данные по дисциплине: {st.session_state.discipline}")

    st.markdown("""
            <p style='text-align: left; font-size: 13px; color: #666;'>
                * Вы можете вводить свои оценки за контрольные, экзамены или коллоквиумы
            </p>
        """, unsafe_allow_html=True)

    def render_module(module_id, is_first=False):
        col1, col2 = st.columns([5, 1])

        with col1:
            if is_first:
                year = st.slider("Учебный год", min_value=2020, max_value=current_year, value=current_year, key='year')

            module_data = st.session_state.data[module_id - 1]
            module_num_val = st.number_input(
                "Номер учебного модуля",
                min_value=1,
                max_value=4,
                step=1,
                value=module_data.get("module_num", 1),
                key=f'module_num_{module_id}'
            )
            grade = st.slider(
                "Оценка за предмет",
                min_value=0,
                max_value=10,
                value=module_data.get("grade", 5),
                key=f'grade_{module_id}'
            )
            difficulty = st.number_input(
                "Оцените сложность предмета в этом модуле",
                min_value=0.0,
                max_value=5.0,
                step=0.5,
                value=module_data.get("difficulty", 2.0),
                key=f'diff_{module_id}'
            )

            attempt_options = ["Первая сдача", "Пересдача", "Пересдача с комиссией",
                               "Пересдача по уважительной причине"]
            attempt_index = attempt_options.index(module_data.get("attempt", "Первая сдача"))

            attempt = st.selectbox(
                "Попытка сдачи",
                attempt_options,
                index=attempt_index,
                key=f'attempt_{module_id}'
            )

            absence = "\\N"
            if attempt != "Первая сдача":
                absence_options = ["Уважительная", "Неуважительная"]
                absence_ = st.selectbox(
                    "Причина отсутствия (если сдавали не с первой попытки)",
                    absence_options,
                    key=f'absence_{module_id}'
                )
                absence = "valid" if absence_ == "Уважительная" else "invalid"

            min_grade_prev = st.slider(
                "Минимальная оценка за предмет (за всё время до этого модуля)",
                min_value=0,
                max_value=10,
                value=module_data.get("min_prev", 4),
                key=f'min_prev_{module_id}'
            )

            max_grade_prev = st.slider(
                "Максимальная оценка за предмет (за всё время до этого модуля)",
                min_value=0,
                max_value=10,
                value=module_data.get("max_prev", 8),
                key=f'max_prev_{module_id}'
            )

            st.session_state.data[module_id - 1]["module_num"] = module_num_val
            st.session_state.data[module_id - 1]["grade"] = grade
            st.session_state.data[module_id - 1]["difficulty"] = difficulty
            st.session_state.data[module_id - 1]["attempt"] = attempt
            st.session_state.data[module_id - 1]["absence_status"] = absence
            st.session_state.data[module_id - 1]["min_prev"] = min_grade_prev
            st.session_state.data[module_id - 1]["max_prev"] = max_grade_prev

        with col2:
            if not is_first:
                if st.button("×", key=f'remove_{module_id}'):
                    st.session_state.data.pop(module_id - 1)
                    for j, m in enumerate(st.session_state.data, 1):
                        m["id"] = j
                    st.rerun()

    # отображение ввода данных по модулям
    for i, module in enumerate(st.session_state.data, 1):
        if i == 1:
            st.subheader("Первый модуль")
        elif i == 2:
            st.markdown("---")
            st.subheader("Второй модуль")
        elif i == 3:
            st.markdown("---")
            st.subheader("Третий модуль")

        render_module(
            module_id = i,
            is_first = (i == 1)
        )

    if len(st.session_state.data) < 3:
        if st.button("+ (добавить еще один модуль)"):
            new_id = len(st.session_state.data) + 1
            st.session_state.data.append({
                "id": new_id,
                "module_num": new_id,
                "grade": 5,
                "difficulty": 2.5,
                "attempt": "Первая сдача",
                "absence_status": "\\N",
                "min_prev": 4,
                "max_prev": 8
            })
            st.rerun()
    else:
        st.info("Максимальное количество модулей добавлено")


    predict_button = st.button("Получить прогноз", type="primary")

    if predict_button:
        modules, grades, difficulties, attempts, absence_statuses, min_grades, max_grades = [], [], [], [], [], [], []

        for module in st.session_state.data:
            modules.append(module["module_num"])
            grades.append(module["grade"])
            difficulties.append(module["difficulty"])
            attempts.append(module["attempt"])
            absence_statuses.append(module["absence_status"])
            min_grades.append(module["min_prev"])
            max_grades.append(module["max_prev"])

        modules_arr = np.array(modules, dtype=int)
        grades_arr = np.array(grades)
        difficulties_arr = np.array(difficulties)
        attempts_arr = np.array(attempts)
        absence_arr = np.array(absence_statuses)
        min_grades_arr = np.array(min_grades)
        max_grades_arr = np.array(max_grades)

        if len(modules_arr) < 2:
            st.error("Для прогноза нужны данные хотя бы за 2 модуля! Добавьте еще один модуль")
        else:
            # сортируем по номеру модуля
            idx = modules_arr.argsort()
            current_idx = idx[-1]
            prev_idx = idx[-2]
            prevs = idx[:-1]

            input_data = pd.DataFrame([{
                "program": st.session_state.program,
                "education_level": st.session_state.education_level,
                "academic_year": st.session_state.year,
                "place_type": st.session_state.place_type,
                "course": int(st.session_state.course),
                "absence_status": absence_arr[current_idx],
                "discipline": st.session_state.discipline,
                "module": int(modules_arr[current_idx]),
                "exam_type": attempts_arr[current_idx],
                "grade_10": float(grades_arr[current_idx]),
                "difficulty_avg_score": float(difficulties_arr[current_idx]),

                "exam_type_prev": attempts_arr[prev_idx],
                "grade_prev": float(grades_arr[prev_idx]),
                "difficulty_prev": float(difficulties_arr[prev_idx]),
                "avg_grade_prev": float(np.mean(grades_arr[prevs])),
                "min_prev": float(np.min(min_grades_arr)),
                "max_prev": float(np.max(max_grades_arr))
            }])

            pred_grade, pred_type = model.predict(input_data, fillna=False)
            st.success("Прогноз рассчитан")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Предсказанная оценка", f"{int(pred_grade[0])} / 10")
            with col2:
                st.metric("Попытка сдачи", pred_type[0])

    st.markdown("---")

    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("←", key="back_arrow"):
            st.session_state.step = 3
            st.rerun()