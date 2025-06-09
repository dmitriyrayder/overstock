import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("⚠️ Prophet не установлен. Прогнозы будут недоступны.")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from io import BytesIO

st.set_page_config(page_title="Анализ товаров", layout="wide")

if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False

st.title("🔍 Анализ товаров: определение кандидатов на снятие")

# === НАСТРОЙКИ ===
with st.sidebar:
    st.header("⚙️ Настройки")
    TOP_N = st.slider("Количество топ-артикулов для Prophet", 10, 50, 20)
    
    st.subheader("🎯 Критерии снятия")
    zero_weeks_threshold = st.slider("Недель подряд без продаж", 8, 20, 12)
    min_total_sales = st.slider("Минимальный объем продаж", 1, 50, 5)
    max_store_ratio = st.slider("Макс. доля магазинов без продаж (%)", 70, 95, 85, 5) / 100
    
    st.subheader("🤖 Модель ML")
    use_balanced_model = st.checkbox("Использовать балансировку классов", value=True)
    final_threshold = st.slider("Финальный порог для снятия (%)", 50, 90, 70, 5) / 100

# === ЗАГРУЗКА ДАННЫХ ===
st.header("📁 Загрузка данных")
st.info("💡 Формат: дата, артикул, количество, магазин, название")

uploaded_file = st.file_uploader("Выберите Excel файл", type=['xlsx', 'xls'])

def load_and_process_data(uploaded_file):
    if uploaded_file is None:
        st.info("👆 Загрузите Excel файл для начала работы")
        return None, False
    
    try:
        file_size = len(uploaded_file.read())
        uploaded_file.seek(0)
        
        if file_size > 50 * 1024 * 1024:
            st.error("❌ Файл слишком большой. Максимум: 50MB")
            return None, False
        
        excel_file = pd.ExcelFile(uploaded_file)
        selected_sheet = st.selectbox("Выберите лист:", excel_file.sheet_names) if len(excel_file.sheet_names) > 1 else excel_file.sheet_names[0]
        
        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet, nrows=100000)
        if len(df) == 100000:
            st.warning("⚠️ Файл обрезан до 100,000 строк")
        
        st.success(f"✅ Загружено {len(df)} строк")
        
        # Сопоставление колонок
        available_cols = list(df.columns)
        col1, col2 = st.columns(2)
        
        with col1:
            date_col = st.selectbox("Дата:", available_cols, index=next((i for i, col in enumerate(available_cols) if any(word in col.lower() for word in ['дат', 'date'])), 0))
            art_col = st.selectbox("Артикул:", available_cols, index=next((i for i, col in enumerate(available_cols) if any(word in col.lower() for word in ['арт', 'art'])), 0))
            qty_col = st.selectbox("Количество:", available_cols, index=next((i for i, col in enumerate(available_cols) if any(word in col.lower() for word in ['кол', 'qty'])), 0))
        
        with col2:
            magazin_col = st.selectbox("Магазин:", available_cols, index=next((i for i, col in enumerate(available_cols) if any(word in col.lower() for word in ['маг', 'magazin'])), 0))
            name_col = st.selectbox("Название:", available_cols, index=next((i for i, col in enumerate(available_cols) if any(word in col.lower() for word in ['назв', 'name'])), 0))
            segment_col = st.selectbox("Сегмент (опционально):", ['Без сегментации'] + available_cols)
        
        # Переименование колонок
        column_mapping = {date_col: 'Data', art_col: 'Art', qty_col: 'Qty', magazin_col: 'Magazin', name_col: 'Name'}
        if segment_col != 'Без сегментации':
            column_mapping[segment_col] = 'Segment'
        
        df = df.rename(columns=column_mapping)
        
        # Проверка обязательных колонок
        required_cols = ['Data', 'Art', 'Qty', 'Magazin', 'Name']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Отсутствуют колонки: {missing_cols}")
            return None, False
        
        # Фильтрация по сегменту
        if 'Segment' in df.columns:
            st.subheader("🎯 Выбор сегмента")
            unique_segments = sorted(df['Segment'].dropna().unique())
            selected_segment = st.selectbox("Сегмент:", ['Все сегменты'] + list(unique_segments))
            
            if selected_segment != 'Все сегменты':
                df = df[df['Segment'] == selected_segment].copy()
                st.success(f"✅ Выбран сегмент: {selected_segment}")
        
        with st.expander("📊 Предварительный просмотр"):
            st.dataframe(df.head())
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Записей", len(df))
            with col2: st.metric("Артикулов", df['Art'].nunique())
            with col3:
                try:
                    date_min = pd.to_datetime(df['Data'], errors='coerce').min()
                    date_max = pd.to_datetime(df['Data'], errors='coerce').max()
                    st.metric("Период", f"{date_min.strftime('%Y-%m-%d')} - {date_max.strftime('%Y-%m-%d')}")
                except:
                    st.metric("Период", "Ошибка дат")
        
        return df, True
        
    except Exception as e:
        st.error(f"❌ Ошибка загрузки: {str(e)}")
        return None, False

df, data_loaded = load_and_process_data(uploaded_file)

if data_loaded:
    st.header("🚀 Запуск анализа")
    if st.button("▶️ НАЧАТЬ АНАЛИЗ", type="primary", use_container_width=True):
        st.session_state.run_analysis = True
    
    if not st.session_state.get('run_analysis', False):
        st.info("👆 Нажмите кнопку для запуска анализа")
        st.stop()
else:
    st.stop()

# === ОСНОВНАЯ ОБРАБОТКА ===
def process_data(df):
    with st.spinner("🔄 Обработка данных..."):
        # Очистка данных
        df['Data'] = pd.to_datetime(df['Data'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Data'])
        df = df[df['Qty'] >= 0]
        
        if len(df) == 0:
            st.error("❌ Нет валидных данных")
            st.stop()
        
        df['year_week'] = df['Data'].dt.strftime('%Y-%U')
        
        # Ограничение артикулов
        all_arts = df['Art'].unique()
        if len(all_arts) > 5000:
            st.warning("⚠️ Обрабатываем топ-5000 артикулов по продажам")
            top_arts = df.groupby('Art')['Qty'].sum().nlargest(5000).index
            all_arts = top_arts
            df = df[df['Art'].isin(all_arts)]
        
        # Агрегация по неделям
        weekly = df.groupby(['Art', 'year_week'])['Qty'].sum().reset_index()
        unique_weeks = sorted(df['year_week'].unique())
        all_weeks = pd.MultiIndex.from_product([all_arts, unique_weeks], names=['Art', 'year_week'])
        weekly = weekly.set_index(['Art', 'year_week']).reindex(all_weeks, fill_value=0).reset_index()
        
        return df, weekly, all_arts, unique_weeks

def calculate_abc_xyz_analysis(df):
    # ABC анализ
    abc_analysis = df.groupby('Art').agg({
        'Qty': ['sum', 'mean', 'std'],
        'Data': ['min', 'max']
    }).reset_index()
    
    abc_analysis.columns = ['Art', 'total_qty', 'avg_qty', 'std_qty', 'first_sale', 'last_sale']
    abc_analysis['days_in_catalog'] = (abc_analysis['last_sale'] - abc_analysis['first_sale']).dt.days + 1
    
    # ABC категории
    abc_analysis = abc_analysis.sort_values('total_qty', ascending=False)
    abc_analysis['cum_qty'] = abc_analysis['total_qty'].cumsum()
    abc_analysis['cum_qty_pct'] = abc_analysis['cum_qty'] / abc_analysis['total_qty'].sum()
    
    def get_abc_category(cum_pct):
        if cum_pct <= 0.8: return 'A'
        elif cum_pct <= 0.95: return 'B'
        else: return 'C'
    
    abc_analysis['abc_category'] = abc_analysis['cum_qty_pct'].apply(get_abc_category)
    
    # XYZ анализ (стабильность спроса)
    abc_analysis['coefficient_variation'] = abc_analysis['std_qty'] / np.maximum(abc_analysis['avg_qty'], 0.01)
    
    def get_xyz_category(cv):
        if cv <= 0.1: return 'X'  # Стабильный спрос
        elif cv <= 0.25: return 'Y'  # Умеренно изменчивый
        else: return 'Z'  # Нестабильный спрос
    
    abc_analysis['xyz_category'] = abc_analysis['coefficient_variation'].apply(get_xyz_category)
    
    return abc_analysis

def calculate_features(weekly, df):
    def compute_features(group):
        sorted_group = group.sort_values('year_week')
        qty_series = sorted_group['Qty']
        
        # Скользящие средние
        ma_3 = qty_series.rolling(3, min_periods=1).mean().iloc[-1] if len(qty_series) > 0 else 0
        ma_6 = qty_series.rolling(6, min_periods=1).mean().iloc[-1] if len(qty_series) > 0 else 0
        
        # Последовательные нули
        consecutive_zeros = 0
        for val in reversed(qty_series.values):
            if val == 0: consecutive_zeros += 1
            else: break
        
        zero_weeks_12 = (qty_series.tail(12) == 0).sum()
        
        # Тренд
        trend = 0
        if len(qty_series) >= 4:
            try:
                x = np.arange(len(qty_series))
                trend = np.polyfit(x, qty_series, 1)[0]
            except: pass
        
        return pd.DataFrame({
            'ma_3': [ma_3], 'ma_6': [ma_6], 'consecutive_zeros': [consecutive_zeros],
            'zero_weeks_12': [zero_weeks_12], 'trend': [trend]
        })
    
    features = weekly.groupby('Art').apply(compute_features, include_groups=False).reset_index()
    features = features.drop('level_1', axis=1, errors='ignore')
    
    # Исправленный расчет доли магазинов без продаж
    total_stores = df['Magazin'].nunique()
    stores_with_sales = df[df['Qty'] > 0].groupby('Art')['Magazin'].nunique().reset_index()
    stores_with_sales.columns = ['Art', 'stores_with_sales']
    stores_with_sales['no_store_ratio'] = 1 - (stores_with_sales['stores_with_sales'] / total_stores)
    
    features = features.merge(stores_with_sales[['Art', 'no_store_ratio']], on='Art', how='left')
    features['no_store_ratio'] = features['no_store_ratio'].fillna(1)
    
    return features

def create_ml_model(features, abc_analysis):
    # Создание меток для обучения
    def create_labels(row):
        score = 0
        if row['abc_category'] == 'C':
            if row['consecutive_zeros'] >= zero_weeks_threshold: score += 3
            elif row['zero_weeks_12'] >= zero_weeks_threshold//2: score += 2
            if row['no_store_ratio'] > max_store_ratio: score += 2
            if row['total_qty'] < min_total_sales: score += 2
            if row['trend'] < -0.1: score += 1
        elif row['abc_category'] in ['A', 'B']:
            if row['consecutive_zeros'] >= zero_weeks_threshold * 1.5: score += 2
            if row['no_store_ratio'] > 0.95: score += 1
        return 1 if score >= 4 else 0
    
    # Объединение данных
    final_features = features.merge(abc_analysis[['Art', 'total_qty', 'abc_category', 'last_sale']], on='Art', how='left')
    final_features['label'] = final_features.apply(create_labels, axis=1)
    
    # Обучение модели
    feature_cols = ['ma_3', 'ma_6', 'consecutive_zeros', 'zero_weeks_12', 'trend', 'no_store_ratio', 'total_qty']
    X = final_features[feature_cols].fillna(0)
    y = final_features['label']
    
    st.write(f"**Распределение:** Снять: {y.sum()}, Оставить: {len(y) - y.sum()}")
    
    if len(y.unique()) > 1 and y.sum() > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.3)
        
        clf = RandomForestClassifier(
            n_estimators=30, random_state=42, 
            class_weight='balanced' if use_balanced_model else None,
            max_depth=8, min_samples_split=5, n_jobs=1
        )
        
        clf.fit(X_train, y_train)
        final_features['prob_dying'] = clf.predict_proba(X)[:, 1] * 100  # В процентах
        test_score = clf.score(X_test, y_test)
    else:
        st.warning("⚠️ Недостаточно данных для ML")
        final_features['prob_dying'] = final_features['label'].astype(float) * 100
        test_score = 0.0
    
    return final_features, test_score

def create_prophet_forecasts(df, features):
    if not PROPHET_AVAILABLE:
        return pd.DataFrame()
    
    try:
        with st.spinner("📈 Прогнозы Prophet..."):
            top_arts = features.nlargest(TOP_N, 'total_qty')['Art']
            forecasts = []
            
            for art in top_arts:
                try:
                    sales = df[df['Art'] == art].groupby('Data')['Qty'].sum().reset_index()
                    if len(sales) < 8: continue
                    
                    sales.columns = ['ds', 'y']
                    model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
                    model.fit(sales)
                    future = model.make_future_dataframe(periods=30)
                    forecast = model.predict(future)
                    median_30 = max(0, forecast.tail(30)['yhat'].median())
                    forecasts.append({'Art': art, 'forecast_30_median': median_30})
                except: continue
            
            return pd.DataFrame(forecasts)
    except:
        return pd.DataFrame()

def get_recommendations(row):
    # Обновленная функция с учетом даты последней продажи
    reasons = []
    if row['abc_category'] == 'C': reasons.append("Категория C")
    if row['consecutive_zeros'] >= zero_weeks_threshold: reasons.append(f"Без продаж {int(row['consecutive_zeros'])} недель")
    if row['zero_weeks_12'] >= zero_weeks_threshold//2: reasons.append(f"Из 12 недель {int(row['zero_weeks_12'])} без продаж")
    if row['no_store_ratio'] > max_store_ratio: reasons.append(f"В {(1-row['no_store_ratio'])*100:.0f}% магазинов")
    if row['total_qty'] < min_total_sales: reasons.append(f"Малый объем ({row['total_qty']:.1f})")
    if row['trend'] < -0.1: reasons.append("Негативный тренд")
    
    # Добавляем дату последней продажи
    if pd.notnull(row.get('last_sale')):
        last_sale_str = row['last_sale'].strftime('%Y-%m-%d')
        reasons.append(f"Последняя продажа: {last_sale_str}")
    
    reason = "; ".join(reasons) if reasons else "Стабильные продажи"
    
    # Рекомендация
    if (row['abc_category'] == 'C' and row['consecutive_zeros'] >= zero_weeks_threshold and row['total_qty'] < min_total_sales):
        return reason, "🚫 Снять"
    elif row['prob_dying'] > final_threshold * 100:
        return reason, "🚫 Снять" 
    elif row['prob_dying'] > final_threshold * 70:
        return reason, "⚠️ Наблюдать"
    else:
        return reason, "✅ Оставить"

# Выполнение анализа
df, weekly, all_arts, unique_weeks = process_data(df)
abc_analysis = calculate_abc_xyz_analysis(df)
features = calculate_features(weekly, df)
final_features, test_score = create_ml_model(features, abc_analysis)
forecast_df = create_prophet_forecasts(df, abc_analysis)

# Финальная таблица
final = final_features.merge(abc_analysis[['Art', 'xyz_category']], on='Art', how='left')
if not forecast_df.empty:
    final = final.merge(forecast_df, on='Art', how='left')
final = final.merge(df[['Art', 'Name']].drop_duplicates(), on='Art', how='left')

# Получение рекомендаций
recommendations = final.apply(get_recommendations, axis=1)
final['Причина'] = [rec[0] for rec in recommendations]
final['Рекомендация'] = [rec[1] for rec in recommendations]

# === РЕЗУЛЬТАТЫ ===
st.header("📊 Результаты анализа")

total_products = len(final)
candidates_remove = len(final[final['Рекомендация'] == "🚫 Снять"])
candidates_watch = len(final[final['Рекомендация'] == "⚠️ Наблюдать"])

col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Всего товаров", total_products)
with col2: st.metric("К снятию", candidates_remove, f"{candidates_remove/total_products*100:.1f}%")
with col3: st.metric("Наблюдать", candidates_watch, f"{candidates_watch/total_products*100:.1f}%")
with col4: st.metric("Точность модели", f"{test_score:.2f}" if test_score > 0 else "N/A")

# ABC/XYZ распределение
st.subheader("📈 ABC/XYZ анализ")
abc_dist = final['abc_category'].value_counts()
xyz_dist = final['xyz_category'].value_counts()

col1, col2 = st.columns(2)
with col1:
    st.write("**ABC категории:**")
    st.write(f"A: {abc_dist.get('A', 0)}, B: {abc_dist.get('B', 0)}, C: {abc_dist.get('C', 0)}")
with col2:
    st.write("**XYZ категории:**")
    st.write(f"X: {xyz_dist.get('X', 0)}, Y: {xyz_dist.get('Y', 0)}, Z: {xyz_dist.get('Z', 0)}")

# === ФИЛЬТРЫ И ТАБЛИЦА ===
st.subheader("🔍 Фильтры")
col1, col2, col3 = st.columns(3)

with col1:
    filter_recommendation = st.selectbox("Рекомендация:", ["Все", "🚫 Снять", "⚠️ Наблюдать", "✅ Оставить"])
    filter_abc = st.selectbox("ABC:", ["Все", "A", "B", "C"])
with col2:
    min_prob = st.slider("Мин. вероятность (%)", 0, 100, 0)
    filter_xyz = st.selectbox("XYZ:", ["Все", "X", "Y", "Z"])
with col3:
    min_zero_weeks = st.slider("Мин. недель без продаж", 0, 20, 0)
    search_art = st.text_input("Поиск артикула/названия")

# Применение фильтров
filtered_df = final.copy()
if filter_recommendation != "Все":
    filtered_df = filtered_df[filtered_df['Рекомендация'] == filter_recommendation]
if filter_abc != "Все":
    filtered_df = filtered_df[filtered_df['abc_category'] == filter_abc]
if filter_xyz != "Все":
    filtered_df = filtered_df[filtered_df['xyz_category'] == filter_xyz]

filtered_df = filtered_df[
    (filtered_df['prob_dying'] >= min_prob) &
    (filtered_df['consecutive_zeros'] >= min_zero_weeks)
]

if search_art:
    mask = (filtered_df['Art'].astype(str).str.contains(search_art, case=False, na=False) |
            filtered_df['Name'].astype(str).str.contains(search_art, case=False, na=False))
    filtered_df = filtered_df[mask]

# Таблица результатов
st.subheader(f"📋 Результаты ({len(filtered_df)} товаров)")

display_columns = ['Art', 'Name', 'abc_category', 'xyz_category', 'total_qty', 'consecutive_zeros', 'no_store_ratio', 'prob_dying', 'Причина', 'Рекомендация']
if 'forecast_30_median' in filtered_df.columns:
    display_columns.insert(-2, 'forecast_30_median')

display_df = filtered_df[display_columns].copy()
display_df['no_store_ratio'] = (display_df['no_store_ratio'] * 100).round(1)  # В процентах
display_df['prob_dying'] = display_df['prob_dying'].round(1)  # Уже в процентах

column_names = ['Артикул', 'Название', 'ABC', 'XYZ', 'Объем', 'Недель_без_продаж', 'Магазины_без_продаж_%', 'Вероятность_снятия_%']
if 'forecast_30_median' in display_df.columns:
    column_names.append('Прогноз_30дн')
column_names.extend(['Причина', 'Рекомендация'])

display_df.columns = column_names
st.dataframe(display_df, use_container_width=True)

# === ЭКСПОРТ ===
st.subheader("💾 Экспорт")
if st.button("📥 Подготовить Excel"):
    try:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            output_cols = ['Art', 'Name', 'abc_category', 'xyz_category', 'total_qty', 'consecutive_zeros', 'no_store_ratio', 'prob_dying', 'Причина', 'Рекомендация']
            if 'forecast_30_median' in final.columns:
                output_cols.insert(-2, 'forecast_30_median')
            
            final[output_cols].to_excel(writer, sheet_name='Результаты', index=False)
            
            stats = pd.DataFrame({
                'Метрика': ['Всего', 'Снять', 'Наблюдать', 'Оставить', 'Порог_ML_%'],
                'Значение': [total_products, candidates_remove, candidates_watch, 
                           total_products - candidates_remove - candidates_watch, final_threshold*100]
            })
            stats.to_excel(writer, sheet_name='Статистика', index=False)
        
        st.download_button("📥 Скачать Excel", buffer.getvalue(), "analysis_results.xlsx", 
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.success("✅ Готово!")
    except Exception as e:
        st.error(f"❌ Ошибка: {str(e)}")

with st.expander("ℹ️ Информация"):
    st.write(f"**Статус:** Prophet {'✅' if PROPHET_AVAILABLE else '❌'}, Обработано: {len(final) if 'final' in locals() else 0}")
    if not PROPHET_AVAILABLE:
        st.warning("⚠️ Установите Prophet: pip install prophet")
