import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Условный импорт Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("⚠️ Prophet не установлен. Прогнозы будут недоступны.")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

st.set_page_config(page_title="Анализ товаров", layout="wide")

if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False

st.title("🔍 Анализ товаров: определение кандидатов на снятие")

# === САЙДБАР ===
with st.sidebar:
    st.header("⚙️ Настройки")
    TOP_N = st.slider("Количество топ-артикулов для Prophet", 10, 100, 50)
    
    st.subheader("🎯 Критерии снятия")
    zero_weeks_threshold = st.slider("Недель подряд без продаж", 8, 20, 12)
    min_total_sales = st.slider("Минимальный объем продаж", 1, 50, 5)
    max_store_ratio = st.slider("Макс. доля магазинов без продаж", 0.7, 0.95, 0.85, 0.05)
    
    st.subheader("🤖 Модель ML")
    use_balanced_model = st.checkbox("Использовать балансировку классов", value=True)
    final_threshold = st.slider("Финальный порог для снятия", 0.5, 0.9, 0.7, 0.05)

# === ЗАГРУЗКА ДАННЫХ ===
st.header("📁 Загрузка данных")

# Добавляем примечание о формате файла
st.info("💡 Убедитесь, что ваш Excel файл содержит колонки: дата, артикул, количество, магазин, название")

uploaded_file = st.file_uploader("Выберите Excel файл", type=['xlsx', 'xls'])

df = None
data_loaded = False

if uploaded_file is not None:
    try:
        # Ограничиваем размер файла
        file_size = len(uploaded_file.read())
        uploaded_file.seek(0)  # Возвращаем указатель в начало
        
        if file_size > 50 * 1024 * 1024:  # 50MB лимит
            st.error("❌ Файл слишком большой. Максимальный размер: 50MB")
            st.stop()
        
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names
        
        if len(sheet_names) > 1:
            selected_sheet = st.selectbox("Выберите лист Excel:", sheet_names)
        else:
            selected_sheet = sheet_names[0]
        
        # Ограничиваем количество строк для чтения
        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet, nrows=100000)
        
        if len(df) == 100000:
            st.warning("⚠️ Файл обрезан до 100,000 строк для обработки")
        
        st.success(f"✅ Загружено {len(df)} строк из листа '{selected_sheet}'")
        
        st.subheader("🔍 Проверка структуры данных")
        available_cols = list(df.columns)
        st.write(f"**Найденные колонки:** {', '.join(available_cols[:10])}{'...' if len(available_cols) > 10 else ''}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Основные колонки:**")
            date_col = st.selectbox("Колонка с датой:", available_cols, 
                                  index=next((i for i, col in enumerate(available_cols) 
                                           if any(word in col.lower() for word in ['дат', 'date'])), 0))
            art_col = st.selectbox("Колонка с артикулом:", available_cols,
                                 index=next((i for i, col in enumerate(available_cols) 
                                          if any(word in col.lower() for word in ['арт', 'art'])), 0))
            qty_col = st.selectbox("Колонка с количеством:", available_cols,
                                 index=next((i for i, col in enumerate(available_cols) 
                                          if any(word in col.lower() for word in ['кол', 'qty', 'количеств'])), 0))
        
        with col2:
            st.write("**Дополнительные колонки:**")
            magazin_col = st.selectbox("Колонка с магазином:", available_cols,
                                     index=next((i for i, col in enumerate(available_cols) 
                                              if any(word in col.lower() for word in ['маг', 'magazin', 'магазин'])), 0))
            name_col = st.selectbox("Колонка с названием:", available_cols,
                                  index=next((i for i, col in enumerate(available_cols) 
                                           if any(word in col.lower() for word in ['назв', 'name', 'наименован'])), 0))
            
            # Выбор колонки с сегментом
            segment_col = st.selectbox("Колонка с сегментом (опционально):", 
                                     ['Без сегментации'] + available_cols)
        
        # Переименовываем колонки
        column_mapping = {
            date_col: 'Data',
            art_col: 'Art', 
            qty_col: 'Qty',
            magazin_col: 'Magazin',
            name_col: 'Name'
        }
        
        if segment_col != 'Без сегментации':
            column_mapping[segment_col] = 'Segment'
        
        df = df.rename(columns=column_mapping)
        
        # Проверяем основные колонки
        required_cols = ['Data', 'Art', 'Qty', 'Magazin', 'Name']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Не удалось сопоставить колонки: {missing_cols}")
        else:
            st.success("✅ Все колонки успешно сопоставлены")
            
            # Выбор сегмента для анализа
            if 'Segment' in df.columns:
                st.subheader("🎯 Выбор сегмента для анализа")
                unique_segments = sorted(df['Segment'].dropna().unique())
                
                col1, col2 = st.columns(2)
                with col1:
                    selected_segment = st.selectbox("Выберите сегмент:", 
                                                  ['Все сегменты'] + list(unique_segments))
                with col2:
                    if selected_segment != 'Все сегменты':
                        segment_count = len(df[df['Segment'] == selected_segment])
                        segment_arts = df[df['Segment'] == selected_segment]['Art'].nunique()
                        st.info(f"📊 Записей: {segment_count}, Артикулов: {segment_arts}")
                
                # Фильтруем данные по выбранному сегменту
                if selected_segment != 'Все сегменты':
                    df = df[df['Segment'] == selected_segment].copy()
                    st.success(f"✅ Выбран сегмент: {selected_segment}")
            
            data_loaded = True
            
            with st.expander("📊 Предварительный просмотр данных"):
                st.dataframe(df.head())
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Всего записей", len(df))
                with col2:
                    st.metric("Уникальных артикулов", df['Art'].nunique())
                with col3:
                    try:
                        date_range_min = pd.to_datetime(df['Data'], errors='coerce').min()
                        date_range_max = pd.to_datetime(df['Data'], errors='coerce').max()
                        if pd.notna(date_range_min) and pd.notna(date_range_max):
                            st.metric("Период данных", f"{date_range_min.strftime('%Y-%m-%d')} - {date_range_max.strftime('%Y-%m-%d')}")
                        else:
                            st.metric("Период данных", "Проверьте формат дат")
                    except:
                        st.metric("Период данных", "Ошибка в датах")
            
    except Exception as e:
        st.error(f"❌ Ошибка загрузки файла: {str(e)}")
        st.info("💡 Попробуйте загрузить файл меньшего размера или проверьте его формат")
        st.stop()
else:
    st.info("👆 Загрузите Excel файл для начала работы")

if data_loaded:
    st.header("🚀 Запуск анализа")
    if st.button("▶️ НАЧАТЬ АНАЛИЗ", type="primary", use_container_width=True):
        st.session_state.run_analysis = True
    
    if not st.session_state.get('run_analysis', False):
        st.info("👆 Нажмите кнопку для запуска анализа")
        st.stop()
else:
    st.stop()

# === ОБРАБОТКА ===
try:
    with st.spinner("🔄 Обработка данных..."):
        # Очистка данных с дополнительной проверкой
        df['Data'] = pd.to_datetime(df['Data'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Data'])
        df = df[df['Qty'] >= 0]
        
        # Проверяем, что есть данные после очистки
        if len(df) == 0:
            st.error("❌ Нет валидных данных после очистки")
            st.stop()
        
        df['year_week'] = df['Data'].dt.strftime('%Y-%U')
        
        # Агрегация по неделям с ограничением памяти
        weekly = df.groupby(['Art', 'year_week'])[['Qty']].sum().reset_index()
        unique_weeks = sorted(df['year_week'].unique())
        all_arts = df['Art'].unique()
        
        # Ограничиваем количество артикулов для обработки
        if len(all_arts) > 10000:
            st.warning("⚠️ Слишком много артикулов. Обрабатываем топ-10000 по продажам.")
            top_arts = df.groupby('Art')['Qty'].sum().nlargest(10000).index
            all_arts = top_arts
            df = df[df['Art'].isin(all_arts)]
            weekly = weekly[weekly['Art'].isin(all_arts)]
        
        all_weeks = pd.MultiIndex.from_product([all_arts, unique_weeks], names=['Art', 'year_week'])
        weekly = weekly.set_index(['Art', 'year_week']).reindex(all_weeks, fill_value=0).reset_index()
        
        # ABC-анализ товаров
        abc_analysis = df.groupby('Art').agg({
            'Qty': ['sum', 'mean', 'count'],
            'Data': ['min', 'max']
        }).reset_index()
        
        abc_analysis.columns = ['Art', 'total_qty', 'avg_qty', 'sales_count', 'first_sale', 'last_sale']
        abc_analysis['days_in_catalog'] = (abc_analysis['last_sale'] - abc_analysis['first_sale']).dt.days + 1
        abc_analysis['sales_velocity'] = abc_analysis['total_qty'] / np.maximum(abc_analysis['days_in_catalog'], 1)
        
        abc_analysis = abc_analysis.sort_values('total_qty', ascending=False)
        abc_analysis['cum_qty'] = abc_analysis['total_qty'].cumsum()
        abc_analysis['cum_qty_pct'] = abc_analysis['cum_qty'] / abc_analysis['total_qty'].sum()
        
        def get_abc_category(cum_pct):
            if cum_pct <= 0.8:
                return 'A'
            elif cum_pct <= 0.95:
                return 'B'
            else:
                return 'C'
        
        abc_analysis['abc_category'] = abc_analysis['cum_qty_pct'].apply(get_abc_category)
        
        # Расчет улучшенных фичей
        def compute_enhanced_features(group):
            sorted_group = group.sort_values('year_week')
            qty_series = sorted_group['Qty']
            
            ma_3 = qty_series.rolling(3, min_periods=1).mean().iloc[-1] if len(qty_series) > 0 else 0
            ma_6 = qty_series.rolling(6, min_periods=1).mean().iloc[-1] if len(qty_series) > 0 else 0
            ma_12 = qty_series.rolling(12, min_periods=1).mean().iloc[-1] if len(qty_series) > 0 else 0
            
            consecutive_zeros = 0
            for val in reversed(qty_series.values):
                if val == 0:
                    consecutive_zeros += 1
                else:
                    break
            
            zero_weeks_12 = (qty_series.tail(12) == 0).sum()
            
            if len(qty_series) >= 4:
                try:
                    x = np.arange(len(qty_series))
                    trend = np.polyfit(x, qty_series, 1)[0]
                except:
                    trend = 0
            else:
                trend = 0
            
            volatility = qty_series.std() if len(qty_series) > 1 else 0
            
            return pd.DataFrame({
                'ma_3': [ma_3], 'ma_6': [ma_6], 'ma_12': [ma_12],
                'consecutive_zeros': [consecutive_zeros], 'zero_weeks_12': [zero_weeks_12],
                'trend': [trend], 'volatility': [volatility]
            })
        
        features = weekly.groupby('Art').apply(compute_enhanced_features, include_groups=False).reset_index()
        features = features.drop('level_1', axis=1, errors='ignore')
        
        # Доля магазинов без продаж
        df['has_sales'] = df['Qty'] > 0
        store_sales = df.groupby(['Art', 'Magazin'])['has_sales'].max().reset_index()
        store_ratio = store_sales.groupby('Art')['has_sales'].mean().reset_index()
        store_ratio['no_store_ratio'] = 1 - store_ratio['has_sales']
        
        # Объединение с ABC-анализом
        features = features.merge(abc_analysis[['Art', 'total_qty', 'sales_velocity', 'abc_category', 'days_in_catalog']], on='Art', how='left')
        features = features.merge(store_ratio[['Art', 'no_store_ratio']], on='Art', how='left')
        
        # Создание более сбалансированных лейблов по сегменту
        def create_balanced_labels(row):
            score = 0
            
            if row['abc_category'] == 'C':
                if row['consecutive_zeros'] >= zero_weeks_threshold:
                    score += 3
                elif row['zero_weeks_12'] >= zero_weeks_threshold//2:
                    score += 2
                if row['no_store_ratio'] > max_store_ratio:
                    score += 2
                if row['total_qty'] < min_total_sales:
                    score += 2
                if row['trend'] < -0.1:
                    score += 1
            elif row['abc_category'] in ['A', 'B']:
                if row['consecutive_zeros'] >= zero_weeks_threshold * 1.5:
                    score += 2
                if row['no_store_ratio'] > 0.95:
                    score += 1
            
            return 1 if score >= 4 else 0
        
        features['label'] = features.apply(create_balanced_labels, axis=1)
        
        # Машинное обучение с балансировкой через class_weight
        feature_cols = ['ma_3', 'ma_6', 'ma_12', 'consecutive_zeros', 'zero_weeks_12', 
                       'trend', 'volatility', 'no_store_ratio', 'total_qty', 'sales_velocity']
        
        X = features[feature_cols].fillna(0)
        y = features['label']
        
        st.write(f"**Распределение классов:** Снять: {y.sum()}, Оставить: {len(y) - y.sum()}")
        
        if len(y.unique()) > 1 and y.sum() > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.3)
            
            # Используем только class_weight для балансировки
            if use_balanced_model:
                clf = RandomForestClassifier(
                    n_estimators=50,  # Уменьшаем для экономии памяти
                    random_state=42, 
                    class_weight='balanced',
                    max_depth=10,
                    min_samples_split=5, 
                    min_samples_leaf=2,
                    n_jobs=1  # Ограничиваем количество потоков
                )
            else:
                clf = RandomForestClassifier(
                    n_estimators=50,
                    random_state=42,
                    max_depth=10,
                    min_samples_split=5, 
                    min_samples_leaf=2,
                    n_jobs=1
                )
            
            clf.fit(X_train, y_train)
            features['prob_dying'] = clf.predict_proba(X)[:, 1]
            train_score = clf.score(X_train, y_train)
            test_score = clf.score(X_test, y_test)
            
        else:
            st.warning("⚠️ Недостаточно разнообразных данных для обучения модели")
            features['prob_dying'] = features['label'].astype(float)
            train_score = test_score = 0.0

except Exception as e:
    st.error(f"❌ Ошибка при обработке данных: {str(e)}")
    st.stop()

# === PROPHET ПРОГНОЗЫ ===
forecast_df = pd.DataFrame()

if PROPHET_AVAILABLE:
    try:
        with st.spinner("📈 Создание прогнозов Prophet..."):
            qty_by_art = features.sort_values('total_qty', ascending=False).head(min(TOP_N, 20))['Art']  # Ограничиваем до 20
            forecast_dfs = []
            
            progress_bar = st.progress(0)
            for i, art in enumerate(qty_by_art):
                try:
                    sales = df[df['Art'] == art].groupby('Data')[['Qty']].sum().reset_index()
                    if len(sales) < 8:
                        continue
                    
                    sales.columns = ['ds', 'y']
                    model = Prophet(
                        daily_seasonality=False, 
                        weekly_seasonality=False,  # Отключаем для экономии ресурсов
                        yearly_seasonality=False,
                        changepoint_prior_scale=0.1,
                        interval_width=0.8
                    )
                    model.fit(sales)
                    future = model.make_future_dataframe(periods=30)
                    forecast = model.predict(future)
                    median_30 = forecast.tail(30)['yhat'].median()
                    forecast_dfs.append({'Art': art, 'forecast_30_median': max(0, median_30)})
                except:
                    continue
                
                progress_bar.progress((i + 1) / len(qty_by_art))
            
            forecast_df = pd.DataFrame(forecast_dfs)
    except Exception as e:
        st.warning(f"⚠️ Ошибка создания прогнозов: {str(e)}")
        forecast_df = pd.DataFrame()
else:
    st.info("ℹ️ Prophet недоступен. Прогнозы пропущены.")

# === ФИНАЛЬНАЯ ТАБЛИЦА ===
final = features.merge(forecast_df, on='Art', how='left') if not forecast_df.empty else features.copy()
if 'forecast_30_median' not in final.columns:
    final['forecast_30_median'] = np.nan

final = final.merge(df[['Art', 'Name']].drop_duplicates(), on='Art', how='left')

# Правила и рекомендации
def get_detailed_reason(row):
    reasons = []
    if row['abc_category'] == 'C':
        reasons.append(f"Категория C")
    if row['consecutive_zeros'] >= zero_weeks_threshold:
        reasons.append(f"Без продаж {int(row['consecutive_zeros'])} недель")
    if row['zero_weeks_12'] >= zero_weeks_threshold//2:
        reasons.append(f"Из 12 недель {int(row['zero_weeks_12'])} без продаж")
    if row['no_store_ratio'] > max_store_ratio:
        reasons.append(f"В {(1-row['no_store_ratio'])*100:.0f}% магазинов")
    if row['total_qty'] < min_total_sales:
        reasons.append(f"Малый объем ({row['total_qty']:.1f})")
    if pd.notnull(row.get('forecast_30_median')) and row['forecast_30_median'] < 0.5:
        reasons.append("Прогноз близок к нулю")
    if row['trend'] < -0.1:
        reasons.append("Негативный тренд")
    return "; ".join(reasons) if reasons else "Стабильные продажи"

def get_final_recommendation(row):
    ml_score = row['prob_dying']
    if (row['abc_category'] == 'C' and row['consecutive_zeros'] >= zero_weeks_threshold and 
        row['total_qty'] < min_total_sales):
        return "🚫 Снять"
    if ml_score > final_threshold:
        return "🚫 Снять"
    elif ml_score > final_threshold * 0.7:
        return "⚠️ Наблюдать"
    else:
        return "✅ Оставить"

final['Причина'] = final.apply(get_detailed_reason, axis=1)
final['Рекомендация'] = final.apply(get_final_recommendation, axis=1)

# === РЕЗУЛЬТАТЫ ===
st.header("📊 Результаты анализа")

col1, col2, col3, col4 = st.columns(4)
with col1:
    total_products = len(final)
    st.metric("Всего товаров", total_products)
with col2:
    candidates_remove = len(final[final['Рекомендация'] == "🚫 Снять"])
    st.metric("К снятию", candidates_remove, f"{candidates_remove/total_products*100:.1f}%")
with col3:
    candidates_watch = len(final[final['Рекомендация'] == "⚠️ Наблюдать"])
    st.metric("Наблюдать", candidates_watch, f"{candidates_watch/total_products*100:.1f}%")
with col4:
    if 'test_score' in locals() and test_score > 0:
        st.metric("Точность модели", f"{test_score:.2f}")

# ABC распределение
st.subheader("📈 ABC анализ")
abc_dist = final['abc_category'].value_counts()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Категория A", abc_dist.get('A', 0))
with col2:
    st.metric("Категория B", abc_dist.get('B', 0))
with col3:
    st.metric("Категория C", abc_dist.get('C', 0))

# === ФИЛЬТРЫ ===
st.subheader("🔍 Фильтры")
col1, col2, col3 = st.columns(3)

with col1:
    filter_recommendation = st.selectbox("Рекомендация:", 
                                       ["Все", "🚫 Снять", "⚠️ Наблюдать", "✅ Оставить"])
    filter_abc = st.selectbox("ABC категория:", ["Все", "A", "B", "C"])

with col2:
    min_prob = st.slider("Мин. вероятность снятия", 0.0, 1.0, 0.0, 0.1)
    min_zero_weeks = st.slider("Мин. недель без продаж", 0, 20, 0)

with col3:
    search_art = st.text_input("Поиск по артикулу/названию")

# Применение фильтров
filtered_df = final.copy()

if filter_recommendation != "Все":
    filtered_df = filtered_df[filtered_df['Рекомендация'] == filter_recommendation]

if filter_abc != "Все":
    filtered_df = filtered_df[filtered_df['abc_category'] == filter_abc]

filtered_df = filtered_df[
    (filtered_df['prob_dying'] >= min_prob) &
    (filtered_df['consecutive_zeros'] >= min_zero_weeks)
]

if search_art:
    mask = (filtered_df['Art'].astype(str).str.contains(search_art, case=False, na=False) |
            filtered_df['Name'].astype(str).str.contains(search_art, case=False, na=False))
    filtered_df = filtered_df[mask]

# === ТАБЛИЦА РЕЗУЛЬТАТОВ ===
st.subheader(f"📋 Таблица результатов ({len(filtered_df)} товаров)")

display_columns = ['Art', 'Name', 'abc_category', 'total_qty', 'consecutive_zeros', 
                  'no_store_ratio', 'prob_dying', 'Причина', 'Рекомендация']

if 'forecast_30_median' in filtered_df.columns and not filtered_df['forecast_30_median'].isna().all():
    display_columns.insert(-2, 'forecast_30_median')

display_df = filtered_df[display_columns].copy()

display_df['no_store_ratio'] = display_df['no_store_ratio'].round(2)
if 'forecast_30_median' in display_df.columns:
    display_df['forecast_30_median'] = display_df['forecast_30_median'].round(1)
display_df['prob_dying'] = display_df['prob_dying'].round(3)
display_df['total_qty'] = display_df['total_qty'].round(1)

column_names = ['Артикул', 'Название', 'ABC', 'Общий_объем', 'Недель_без_продаж', 
               'Доля_маг_без_продаж']
if 'forecast_30_median' in display_df.columns:
    column_names.append('Прогноз_30дн')
column_names.extend(['Вероятность_снятия', 'Причина', 'Рекомендация'])

display_df.columns = column_names

st.dataframe(display_df, use_container_width=True)

# === ЭКСПОРТ ===
st.subheader("💾 Экспорт результатов")

if st.button("📥 Подготовить файл для скачивания"):
    try:
        output_columns = ['Art', 'Name', 'abc_category', 'total_qty', 'consecutive_zeros', 
                         'no_store_ratio', 'prob_dying', 'Причина', 'Рекомендация']
        
        if 'forecast_30_median' in final.columns:
            output_columns.insert(-2, 'forecast_30_median')
        
        output = final[output_columns].copy()
        
        from io import BytesIO
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            output.to_excel(writer, sheet_name='Результаты', index=False)
            
            stats = pd.DataFrame({
                'Метрика': ['Всего товаров', 'К снятию', 'Наблюдать', 'Оставить', 'Порог ML'],
                'Значение': [total_products, candidates_remove, candidates_watch, 
                            total_products - candidates_remove - candidates_watch, final_threshold]
            })
            stats.to_excel(writer, sheet_name='Статистика', index=False)
            
            abc_summary = final.groupby(['abc_category', 'Рекомендация']).size().unstack(fill_value=0)
            abc_summary.to_excel(writer, sheet_name='ABC_анализ')
        
        st.download_button(
            label="📥 Скачать Excel",
            data=buffer.getvalue(),
            file_name="product_analysis_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.success("✅ Файл готов к скачиванию!")
        
    except Exception as e:
        st.error(f"❌ Ошибка при создании файла: {str(e)}")

# Добавляем информацию о системе
with st.expander("ℹ️ Информация о системе"):
    st.write("**Статус компонентов:**")
    st.write(f"- Prophet: {'✅ Доступен' if PROPHET_AVAILABLE else '❌ Недоступен'}")
    st.write(f"- Sklearn: ✅ Доступен")
    st.write(f"- Pandas: ✅ Доступен")
    st.write(f"- Обработано артикулов: {len(final) if 'final' in locals() else 0}")
    
    if not PROPHET_AVAILABLE:
        st.warning("⚠️ Для полной функциональности установите Prophet: pip install prophet")
