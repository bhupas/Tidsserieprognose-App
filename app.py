# -*- coding: utf-8 -*-
# Importer nødvendige biblioteker
import streamlit as st
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy import array
from torch import nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from datetime import timedelta
import gc
import plotly.figure_factory as ff
import traceback # Til detaljeret fejlfinding

# --- Sidekonfiguration ---
st.set_page_config(layout="wide", page_title="Automatisk Tidsserieprognose")

# --- Opsætning af Enhed (CPU/GPU) ---
if torch.cuda.is_available():
    device = torch.device('cuda')
    st.sidebar.success("GPU (CUDA) er tilgængelig og vil blive brugt!", icon="✅")
    torch.cuda.empty_cache() # Ryd cache ved start
else:
    device = torch.device('cpu')
    st.sidebar.warning("GPU (CUDA) ikke tilgængelig, bruger CPU. Træning kan være langsom.", icon="⚠️")

# --- Model Definition (LSTM) ---
# (Modelkoden forbliver på engelsk for standardisering)
class LSTMForecasting(nn.Module):
    def __init__(self, input_feature_size, lstm_hidden_size, linear_hidden_size, lstm_num_layers, linear_num_layers, output_size):
        super(LSTMForecasting, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        self.lstm = nn.LSTM(input_feature_size, lstm_hidden_size, lstm_num_layers, batch_first=True)

        # --- Dynamiske Lineære Lag ---
        self.linear_layers = nn.ModuleList()
        current_linear_size = lstm_hidden_size
        if linear_num_layers == 1:
            target_linear_size = output_size
        else:
            target_linear_size = linear_hidden_size

        # Første lineære lag
        self.linear_layers.append(nn.Linear(current_linear_size, target_linear_size))
        if linear_num_layers > 1:
            self.linear_layers.append(nn.ReLU())
        current_linear_size = target_linear_size

        # Efterfølgende skjulte lineære lag (reducerende størrelse)
        num_intermediate_layers = max(0, linear_num_layers - 2)
        if num_intermediate_layers > 0 and linear_num_layers > 1:
            min_intermediate_size = max(output_size * 2, 16)
            final_hidden_size_target = max(min_intermediate_size, int(linear_hidden_size * 0.5))

            if current_linear_size > final_hidden_size_target:
                 size_reduction_factor = (final_hidden_size_target / current_linear_size)**(1/num_intermediate_layers)
                 size_reduction_factor = max(0.5, min(0.95, size_reduction_factor))
            else:
                 size_reduction_factor = 1.0

            for i in range(num_intermediate_layers):
                next_size = max(min_intermediate_size, int(current_linear_size * size_reduction_factor))
                self.linear_layers.append(nn.Linear(current_linear_size, next_size))
                self.linear_layers.append(nn.ReLU())
                current_linear_size = next_size

        # Sidste output-lag (kun hvis linear_num_layers > 1)
        if linear_num_layers > 1:
             self.fc = nn.Linear(current_linear_size, output_size)
        else:
             self.fc = None # Håndteres i forward-pass

    def forward(self, x):
        h0 = torch.zeros(self.lstm_num_layers, x.size(0), self.lstm_hidden_size, device=x.device)
        c0 = torch.zeros(self.lstm_num_layers, x.size(0), self.lstm_hidden_size, device=x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = lstm_out[:, -1, :]

        for layer in self.linear_layers:
             out = layer(out)

        if self.fc is not None:
             out = self.fc(out)
        return out

# --- Datahåndteringsfunktioner ---
@st.cache_data # Cache indlæste data
def load_data(uploaded_file):
    """Indlæser data fra uploadet CSV-fil."""
    if uploaded_file is None:
        return None
    try:
        # Prøv almindelige separatorer
        try:
            # Forsøg at udlede separatoren automatisk
            df = pd.read_csv(uploaded_file, sep=None, engine='python', on_bad_lines='warn')
            # Hvis automatisk detektion giver få kolonner, prøv specifikke separatorer
            if df.shape[1] <= 1:
                 uploaded_file.seek(0) # Nulstil filpointer
                 df = pd.read_csv(uploaded_file, sep='[,;|]', engine='python', on_bad_lines='warn')
        except Exception:
            uploaded_file.seek(0) # Nulstil filpointer
            df = pd.read_csv(uploaded_file, sep='[,;|]', engine='python', on_bad_lines='warn')

        st.success(f"CSV-fil indlæst succesfuldt! (Størrelse: {df.shape[0]} rækker, {df.shape[1]} kolonner)", icon="✅")
        return df
    except Exception as e:
        st.error(f"Fejl ved indlæsning af CSV: {e}. Kontroller, at filen er en gyldig CSV med standardseparator (komma, semikolon, pipe).", icon="🚨")
        return None

@st.cache_data # Cache forbehandlingsresultater
def preprocess_data(_df, date_col, input_cols, output_col):
    """Forbehandler de valgte data."""
    if _df is None:
        st.warning("Ingen data indlæst.", icon="⚠️")
        return None
    if not date_col or not output_col or not input_cols:
        st.warning("Vælg venligst Dato-, Input- og Output-kolonner.", icon="⚠️")
        return None

    if output_col in input_cols:
        st.error(f"Output-kolonnen '{output_col}' kan ikke også være en Input-kolonne.", icon="🚨")
        return None

    try:
        process_df = _df.copy() # Arbejd på en kopi

        # Sikre unikke kolonner, bevar rækkefølge, output til sidst
        all_cols = [date_col] + [col for col in input_cols if col != output_col] + [output_col]
        missing_cols = [col for col in all_cols if col not in process_df.columns]
        if missing_cols:
            st.error(f"Valgte kolonner blev ikke fundet i de uploadede data: {missing_cols}", icon="🚨")
            return None

        process_df = process_df[list(dict.fromkeys(all_cols))]

        # --- Håndtering af Datokolonne ---
        try:
            process_df[date_col] = pd.to_datetime(process_df[date_col])
        except Exception:
            try:
                 process_df[date_col] = pd.to_datetime(process_df[date_col], infer_datetime_format=True)
            except (ValueError, TypeError) as e:
                 st.error(f"Fejl ved parsing af datokolonne '{date_col}': {e}. Sikr, at formatet er standard (f.eks. YYYY-MM-DD, MM/DD/YYYY HH:MM:SS). Tjek for uregelmæssigheder.", icon="🚨")
                 return None

        # Fjern rækker hvor datoparsing fejlede (NaT)
        initial_rows = len(process_df)
        process_df.dropna(subset=[date_col], inplace=True)
        dropped_date_rows = initial_rows - len(process_df)
        if dropped_date_rows > 0:
             st.warning(f"Fjernede {dropped_date_rows} rækker pga. fejl ved datoparsing.", icon="⚠️")

        # Sorter efter dato *før* behandling af numeriske kolonner
        process_df.sort_values(by=date_col, inplace=True)

        # --- Håndtering af Numeriske Kolonner ---
        feature_cols = process_df.columns.drop(date_col)
        initial_rows = len(process_df) # Nulstil rækkeantal efter dato-rensning

        for col in feature_cols:
             # Forsøg numerisk konvertering, ignorer fejl (bliver NaN)
             process_df[col] = pd.to_numeric(process_df[col], errors='coerce')

        # Fjern rækker hvor *output*-kolonnen blev NaN efter konvertering
        process_df.dropna(subset=[output_col], inplace=True)
        dropped_target_rows = initial_rows - len(process_df)
        if dropped_target_rows > 0:
             st.warning(f"Fjernede {dropped_target_rows} rækker, hvor målvariablen ('{output_col}') ikke kunne konverteres til numerisk.", icon="⚠️")

        if process_df.empty:
            st.error("Ingen gyldige data tilbage efter håndtering af datoer eller ikke-numeriske målvariable.", icon="🚨")
            return None

        # Imputer resterende NaNs i numeriske kolonner (både input og evt. output)
        numeric_cols_to_impute = [col for col in feature_cols if pd.api.types.is_numeric_dtype(process_df[col])]
        if not numeric_cols_to_impute:
             st.warning("Ingen numeriske inputkolonner fundet efter konvertering. Tjek valget af inputkolonner.", icon="⚠️")
        else:
            imputed_counts = {}
            for col in numeric_cols_to_impute:
                 initial_nans = process_df[col].isnull().sum()
                 if initial_nans > 0:
                      process_df[col] = process_df[col].interpolate(method='linear', limit_direction='both', axis=0)
                      process_df[col] = process_df[col].ffill().bfill() # Fyld evt. resterende i start/slut
                      final_nans = process_df[col].isnull().sum()
                      if initial_nans - final_nans > 0:
                         imputed_counts[col] = initial_nans - final_nans
                 if process_df[col].isnull().any():
                     st.warning(f"Kolonnen '{col}' indeholder stadig manglende værdier efter imputation. Overvej at gennemgå kildedata.", icon="⚠️")

            if imputed_counts:
                 impute_msg = ", ".join([f"{col} ({count})" for col, count in imputed_counts.items()])
                 st.info(f"Imputerede manglende værdier i: {impute_msg}", icon="ℹ️")

        process_df.reset_index(drop=True, inplace=True)

        st.success("Dataforbehandling fuldført!", icon="✅")
        return process_df

    except KeyError as e:
        st.error(f"Kolonnefejl under forbehandling: '{e}'. Tjek at valgte kolonner findes i data.", icon="🚨")
        return None
    except Exception as e:
        st.error(f"En uventet fejl opstod under forbehandling: {e}", icon="🚨")
        st.error(traceback.format_exc()) # Log detaljeret traceback for debugging
        return None

def check_date_frequency(date_series):
    """Bestemmer den dominerende tidsfrekvens og returnerer den tilsvarende periode."""
    if len(date_series) < 2: return None
    dates = pd.to_datetime(date_series).sort_values().unique()
    if len(dates) < 2: return None
    differences = pd.Series(dates[1:]) - pd.Series(dates[:-1])
    if differences.empty: return None
    mode_diff = differences.mode()
    if mode_diff.empty or pd.isna(mode_diff[0]):
        st.info("Kunne ikke bestemme en enkelt dominerende tidsforskel. Data kan være uregelmæssigt fordelt.", icon="ℹ️")
        return None
    dominant_diff = mode_diff[0]

    # Sammenlign med almindelige perioder
    one_day = pd.Timedelta(days=1); one_week = pd.Timedelta(weeks=1)
    one_hour = pd.Timedelta(hours=1); one_minute = pd.Timedelta(minutes=1)
    if dominant_diff == one_day:
        st.info("Registrerede Daglig frekvens (Periode=365 brugt til dekomponering).", icon="ℹ️")
        return 365
    elif dominant_diff == one_week:
        st.info("Registrerede Ugentlig frekvens (Periode=52 brugt til dekomponering).", icon="ℹ️")
        return 52
    elif pd.Timedelta(days=28) <= dominant_diff <= pd.Timedelta(days=31):
        st.info("Registrerede ~Månedlig frekvens (Periode=12 brugt til dekomponering).", icon="ℹ️")
        return 12
    elif dominant_diff == one_hour:
        st.info("Registrerede Time-frekvens (Periode=24*7=168 brugt til dekomponering - ugentligt mønster).", icon="ℹ️")
        return 24*7
    elif dominant_diff == one_minute:
         st.info("Registrerede Minut-frekvens (Periode=60*24=1440 brugt til dekomponering - dagligt mønster).", icon="ℹ️")
         return 60*24
    else:
        st.info(f"Kunne ikke matche dominerende forskel ({dominant_diff}) til en standardfrekvens (daglig, ugentlig, månedlig, time, minut). Dekomponering kan være mindre præcis eller fejle.", icon="ℹ️")
        return None

@st.cache_data # Cache dekomponeringsresultater
def get_seasonal_decomposition(_processed_data, date_col, output_col):
    """Udfører og plotter sæsondekomponering."""
    if _processed_data is None or date_col not in _processed_data.columns or output_col not in _processed_data.columns:
        return None, None, None
    if _processed_data[output_col].isnull().any():
         st.warning("Output-kolonnen indeholder NaNs selv efter forbehandling. Skipper dekomponering.", icon="⚠️")
         return None, None, None

    try:
        data_for_decomp = _processed_data.set_index(date_col).sort_index()[output_col]
        if data_for_decomp.index.duplicated().any():
            st.warning("Dublerede datoer fundet i tidsserieindekset. Aggregerer ved at tage gennemsnittet.", icon="⚠️")
            data_for_decomp = data_for_decomp.groupby(data_for_decomp.index).mean()

        period = check_date_frequency(data_for_decomp.index)

        if period is None or period <= 1:
            st.warning(f"Kan ikke udføre sæsondekomponering: Bestemt periode er {period}. Kræver periode > 1.", icon="⚠️")
            return None, None, None
        if len(data_for_decomp) < 2 * period:
             st.warning(f"Ikke nok data til sæsondekomponering med periode {period} (Kræver mindst {2*period} punkter, har {len(data_for_decomp)}). Skipper.", icon="⚠️")
             return None, None, None

        result = seasonal_decompose(data_for_decomp, model='additive', period=period, extrapolate_trend='freq')

        # --- Opret Plots ---
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal.values, mode='lines', name='Sæson', line=dict(color='orange')))
        fig_s.update_layout(title='Sæsonkomponent', xaxis_title='Dato', yaxis_title='Værdi', height=300, margin=dict(t=30, b=10, l=10, r=10))

        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=result.trend.index, y=result.trend.values, mode='lines', name='Trend', line=dict(color='blue')))
        fig_t.update_layout(title='Trendkomponent', xaxis_title='Dato', yaxis_title='Værdi', height=300, margin=dict(t=30, b=10, l=10, r=10))

        fig_r = go.Figure()
        resid_vals = result.resid.dropna()
        fig_r.add_trace(go.Scatter(x=resid_vals.index, y=resid_vals.values, mode='markers', name='Residualer', marker=dict(color='green', size=4)))
        fig_r.update_layout(title='Residualkomponent', xaxis_title='Dato', yaxis_title='Værdi', height=300, margin=dict(t=30, b=10, l=10, r=10))

        return fig_t, fig_s, fig_r

    except ValueError as ve:
         st.error(f"Værdifejl under sæsondekomponering: {ve}. Kan ske ved utilstrækkelige datapunkter for den detekterede periode.", icon="🚨")
         return None, None, None
    except Exception as e:
        st.error(f"Fejl under sæsondekomponering: {e}", icon="🚨")
        st.error(traceback.format_exc())
        return None, None, None

#@st.cache_data # Cache af figure factory kan give problemer, deaktiveret hvis nødvendigt
def get_correlation_heatmap(_processed_data):
    """ Beregner og plotter korrelationsheatmap """
    if _processed_data is None: return None
    try:
        corr_df = _processed_data.select_dtypes(include=np.number)
        if corr_df.shape[1] < 2:
            st.info("Kræver mindst 2 numeriske kolonner for en korrelationsmatrix.", icon="ℹ️")
            return None

        corr_df = corr_df.loc[:, corr_df.std() > 1e-6] # Fjern kolonner uden varians
        if corr_df.shape[1] < 2:
            st.info("Kræver mindst 2 numeriske kolonner *med varians* for en korrelationsmatrix.", icon="ℹ️")
            return None

        correlation_matrix = corr_df.corr()
        fig_corr = ff.create_annotated_heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns.tolist(),
            y=correlation_matrix.index.tolist(),
            annotation_text=correlation_matrix.round(2).values,
            colorscale='Viridis', showscale=True, hoverinfo='z'
        )
        fig_corr.update_layout(
            title='Korrelationsmatrix for Numeriske Features',
            margin=dict(t=50, l=20, r=20, b=20), xaxis_tickangle=-45
        )
        return fig_corr
    except Exception as e:
        st.error(f"Kunne ikke generere korrelationsmatrix: {e}", icon="🚨")
        st.error(traceback.format_exc())
        return None

def split_sequences(features, target, n_steps_in, n_steps_out):
    """Opdeler multivariat tidsserie i input/output-sekvenser."""
    X, y = list(), list()
    n_features = features.shape[1]
    n_target_features = target.shape[1]
    if len(features) != len(target): raise ValueError("Features og target skal have samme længde.")

    total_samples = len(features)
    required_len_for_one_sample = n_steps_in + n_steps_out
    if total_samples < required_len_for_one_sample:
         st.error(f"Kan ikke opdele sekvenser: Total antal prøver ({total_samples}) er mindre end krævet længde for én sekvens (forsinkelse {n_steps_in} + prognose {n_steps_out} = {required_len_for_one_sample}).", icon="🚨")
         return torch.empty(0, n_steps_in, n_features).float(), torch.empty(0, n_steps_out).float()

    for i in range(total_samples):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > total_samples: break
        seq_x = features[i:end_ix, :]
        seq_y = target[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)

    if not X:
        st.error(f"Sekvensopdeling resulterede i nul prøver. Tjek forsinkelse ({n_steps_in}) / prognose ({n_steps_out}) trin relativt til datalængde ({total_samples}).", icon="🚨")
        return torch.empty(0, n_steps_in, n_features).float(), torch.empty(0, n_steps_out).float()

    X = array(X); y = array(y)
    if n_target_features == 1 and y.ndim == 3: y = y.reshape(y.shape[0], y.shape[1])
    return torch.from_numpy(X).float(), torch.from_numpy(y).float()

# --- Streamlit App Layout ---
st.title("📈 Tidsserieprognose med LSTM")
st.markdown("""
Upload dine tidsseriedata (CSV), vælg features, konfigurer LSTM-modellen,
træn den, og visualiser prognoserne mod de faktiske værdier.
""")

# --- Placeholders til dynamisk indhold ---
data_preview_placeholder = st.empty()
preprocessing_vis_placeholder = st.empty()
training_results_placeholder = st.empty()
prediction_plot_placeholder = st.empty()

# --- Sidebar til Kontroller ---
st.sidebar.header("⚙️ Kontrolpanel & Parametre")

# 1. Fil Upload
st.sidebar.subheader("1. Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV-fil:", type=['csv', 'txt'], help="Upload en CSV-fil med tidsseriedata. Sørg for, at den indeholder en datokolonne og mindst én numerisk kolonne.")

# --- Initialiser Session State variabler ---
if 'df_raw' not in st.session_state: st.session_state.df_raw = None
if 'processed_data' not in st.session_state: st.session_state.processed_data = None
if 'last_uploaded_filename' not in st.session_state: st.session_state.last_uploaded_filename = None
if 'ui_key_prefix' not in st.session_state: st.session_state.ui_key_prefix = 0 # Til at nulstille widgets

# Tjek om en ny fil er uploadet
new_upload = False
if uploaded_file is not None:
    if st.session_state.last_uploaded_filename != uploaded_file.name:
        st.session_state.last_uploaded_filename = uploaded_file.name
        st.session_state.df_raw = None
        st.session_state.processed_data = None
        st.cache_data.clear() # Ryd @st.cache_data
        st.session_state.ui_key_prefix += 1 # Inkrementer prefix for at nulstille widget states
        new_upload = True
        st.success("Ny fil registreret. Tidligere tilstand og valg nulstillet.", icon="🔄")

# Indlæs data kun hvis de ikke allerede er indlæst eller ved ny upload
if uploaded_file is not None and (st.session_state.df_raw is None or new_upload):
    st.session_state.df_raw = load_data(uploaded_file)

# Vis rå data preview hvis data er indlæst
raw_df_rows = 0; raw_df_cols = 0
if st.session_state.df_raw is not None:
    raw_df_rows = st.session_state.df_raw.shape[0]
    raw_df_cols = st.session_state.df_raw.shape[1]
    with data_preview_placeholder.container():
        with st.expander("Vis Rå Data (Første 5 Rækker)", expanded=False):
             st.dataframe(st.session_state.df_raw.head(), use_container_width=True)
             st.info(f"Uploadede data har {raw_df_rows} rækker og {raw_df_cols} kolonner.")
else:
    st.sidebar.info("Upload en CSV-fil for at begynde.")
    data_preview_placeholder.empty()

# --- Hovedområde ---
col1, col2 = st.columns([3, 2]) # Justeret kolonneforhold for mere plads til analyse

with col1:
    st.subheader("📊 Dataanalyse & Forbehandling")
    with st.container(border=True, height=800): # Begrænset højde med scroll hvis nødvendigt
        if st.session_state.df_raw is not None:
            st.markdown("**Feature Valg:**")
            all_columns = st.session_state.df_raw.columns.tolist()
            key_prefix = str(st.session_state.ui_key_prefix) # Brug prefix til nøgler

            date_col = st.selectbox(
                "Vælg Dato/Tids Kolonne:", options=all_columns, index=0 if all_columns else None,
                key=f"{key_prefix}_date_col_select",
                help="Vælg kolonnen, der indeholder dato- eller tidsinformation."
                )

            available_features = [col for col in all_columns if col != date_col]

            input_cols = st.multiselect(
                "Inputkolonner (X):", options=available_features,
                key=f"{key_prefix}_input_cols_select",
                help="Vælg de kolonner, der skal bruges som input til modellen (features)."
                )

            available_output = [col for col in available_features if col not in input_cols]
            output_col = st.selectbox(
                "Outputkolonne (Y - Mål):", options=available_output, index=0 if available_output else None,
                key=f"{key_prefix}_output_col_select",
                help="Vælg den kolonne, modellen skal forudsige (målvariablen)."
                )

            st.divider()
            run_preprocessing = st.button(
                'Forbehandl & Analyser Data', type="primary", key=f"{key_prefix}_preprocess_button",
                disabled=(st.session_state.df_raw is None or not date_col or not input_cols or not output_col),
                help="Klik her for at rense data, håndtere manglende værdier og udføre indledende analyse."
                )

            # --- Udfør Forbehandling ---
            if run_preprocessing:
                with st.spinner("Forbehandler data..."):
                    st.session_state.processed_data = preprocess_data(st.session_state.df_raw, date_col, input_cols, output_col)

            # --- Vis Forbehandlingsresultater ---
            if st.session_state.processed_data is not None:
                 with preprocessing_vis_placeholder.container():
                    st.success("Data er succesfuldt forbehandlet!")
                    with st.expander("Vis Forbehandlede Data & Analyse", expanded=True):
                        st.metric("Rækker efter forbehandling:", st.session_state.processed_data.shape[0])
                        st.metric("Kolonner efter forbehandling:", st.session_state.processed_data.shape[1])
                        st.dataframe(st.session_state.processed_data.head(), use_container_width=True)
                        st.divider()

                        st.write("**Korrelationsmatrix:**")
                        fig_corr = get_correlation_heatmap(st.session_state.processed_data)
                        if fig_corr: st.plotly_chart(fig_corr, use_container_width=True)

                        st.divider()
                        st.write("**Sæsondekomponering:**")
                        fig_t, fig_s, fig_r = get_seasonal_decomposition(st.session_state.processed_data, date_col, output_col)
                        if fig_t and fig_s and fig_r:
                           st.plotly_chart(fig_t, use_container_width=True)
                           st.plotly_chart(fig_s, use_container_width=True)
                           st.plotly_chart(fig_r, use_container_width=True)

            elif run_preprocessing: # Hvis knappen blev trykket, men processering fejlede
                preprocessing_vis_placeholder.error("Forbehandling fejlede. Tjek fejlmeddelelser ovenfor og dine data/valg.", icon="🚨")

        else:
            st.info("Upload data og vælg kolonner for at aktivere forbehandling.")
            preprocessing_vis_placeholder.empty()


with col2:
    st.subheader("🧠 LSTM Model Træning")
    with st.container(border=True, height=800): # Begrænset højde
        if st.session_state.processed_data is not None:
            current_processed_data = st.session_state.processed_data
            key_prefix = str(st.session_state.ui_key_prefix) # Brug prefix til nøgler
            min_required_rows = 10

            if current_processed_data.shape[0] < min_required_rows:
                st.error(f"Utilstrækkelige data til træning efter forbehandling ({current_processed_data.shape[0]} rækker). Kræver mindst {min_required_rows}.", icon="🚨")
                train_button_disabled = True
            else:
                 # --- Sekvensparametre ---
                 st.markdown("**Sekvens Definition:**")
                 max_data_points = current_processed_data.shape[0]
                 default_forecast = min(50, max(1, int(max_data_points * 0.1)))
                 default_lag = min(max_data_points - default_forecast, max(1, default_forecast * 2))
                 default_lag = max(1, default_lag)

                 forecast_steps = st.number_input(
                     'Forudsigelsestrin (N skridt frem):', min_value=1, max_value=max(1, max_data_points - 1),
                     value=default_forecast, step=1, key=f"{key_prefix}_forecast_steps",
                     help="Hvor mange fremtidige tidspunkter skal modellen forudsige?"
                     )
                 max_allowable_lag = max(1, max_data_points - forecast_steps)
                 lag_steps = st.number_input(
                     'Forsinkelsestrin (N skridt tilbage):', min_value=1, max_value=max_allowable_lag,
                     value=min(default_lag, max_allowable_lag), step=1, key=f"{key_prefix}_lag_steps",
                     help="Hvor mange tidligere tidspunkter skal bruges som input til hver forudsigelse?"
                     )

                 effective_train_len_for_split = max_data_points - forecast_steps
                 num_train_samples = max(0, effective_train_len_for_split - lag_steps + 1)

                 train_button_disabled = False
                 if forecast_steps >= max_data_points:
                     st.error(f"Forudsigelsestrin ({forecast_steps}) skal være mindre end det samlede antal datapunkter ({max_data_points}).", icon="🚨")
                     train_button_disabled = True
                 elif effective_train_len_for_split < lag_steps:
                     st.error(f"Træningsdatasætets størrelse efter split ({effective_train_len_for_split} rækker) er mindre end de krævede Forsinkelsestrin ({lag_steps}). Reducer trin eller skaf mere data.", icon="🚨")
                     train_button_disabled = True
                 elif num_train_samples == 0 :
                      st.error(f"Nuværende indstillinger (Forsinkelse={lag_steps}, Prognose={forecast_steps}) resulterer i 0 træningsprøver fra de tilgængelige {effective_train_len_for_split} træningsdatapunkter. Juster trin.", icon="🚨")
                      train_button_disabled = True
                 else:
                     st.info(f"Effektive data til træningssekvenser: {effective_train_len_for_split} rækker. Testdatasæt: {forecast_steps} rækker. Estimeret antal træningsprøver: {num_train_samples}.", icon="ℹ️")

                 st.divider()

                 # --- Modelarkitektur ---
                 st.markdown("**Model Arkitektur:**")
                 lstm_layers = st.slider('Antal LSTM Lag:', min_value=1, max_value=5, value=2, key=f"{key_prefix}_lstm_layers", disabled=train_button_disabled, help="Antal stablede LSTM-lag.")
                 lstm_neurons = st.slider('LSTM Neuroner (pr. lag):', min_value=16, max_value=512, value=100, step=8, key=f"{key_prefix}_lstm_neurons", disabled=train_button_disabled, help="Antal neuroner (hidden units) i hvert LSTM-lag.")
                 st.divider()
                 linear_hidden_layers = st.slider('Antal Tætte Skjulte Lag (efter LSTM):', min_value=1, max_value=5, value=1, key=f"{key_prefix}_linear_layers", disabled=train_button_disabled, help="Antal fuldt forbundne (dense) lag efter LSTM-lagene.")
                 min_linear_neurons = max(forecast_steps, 16)
                 default_linear_neurons = max(min_linear_neurons, min(1024, lstm_neurons))
                 linear_hidden_neurons = st.slider('Neuroner i Første Tætte Skjulte Lag:', min_value=min_linear_neurons, max_value=1024, value=default_linear_neurons, step=16, key=f"{key_prefix}_linear_neurons", disabled=train_button_disabled, help="Antal neuroner i det første tætte lag. Efterfølgende lag vil typisk have færre.")
                 st.caption("Efterfølgende tætte lag vil aftage i størrelse mod output.")
                 st.divider()

                 # --- Træningsparametre ---
                 st.markdown("**Træningsparametre:**")
                 n_epochs = st.number_input('Antal Epochs:', min_value=1, max_value=1000, value=50, step=1, key=f"{key_prefix}_epochs", disabled=train_button_disabled, help="Antal gange hele træningsdatasættet gennemgås under træning.")
                 default_batch_size = min(64, max(1, num_train_samples // 10 if num_train_samples > 0 else 1))
                 max_batch_size = max(1, num_train_samples)
                 batch_size = st.slider('Batch Størrelse:', min_value=1, max_value=max_batch_size, value=min(default_batch_size, max_batch_size), key=f"{key_prefix}_batch_size", disabled=train_button_disabled or max_batch_size <=1, help="Antal prøver der behandles ad gangen i hver træningsiteration.")
                 learning_rate = st.number_input('Learning Rate:', min_value=1e-6, max_value=1e-1, value=0.001, step=1e-4, format="%f", key=f"{key_prefix}_lr", disabled=train_button_disabled, help="Størrelsen af skridt taget under optimering (Adam optimizer).")

                 st.divider()

                 # --- Træningsknap ---
                 run_training = st.button(
                     'Træn LSTM Model', type="primary", key=f"{key_prefix}_train_button",
                     disabled=train_button_disabled,
                     help="Start træningen af LSTM-modellen med de valgte parametre."
                     )

                 # --- Træningsudførelse ---
                 if run_training and not train_button_disabled:
                     # Hent nødvendige kolonner fra session state (valgt i col1)
                     # Sikrer vi bruger de samme kolonner som blev forbehandlet
                     if f"{key_prefix}_input_cols_select" not in st.session_state or \
                        f"{key_prefix}_output_col_select" not in st.session_state or \
                        f"{key_prefix}_date_col_select" not in st.session_state:
                           st.error("Kolonnevalg mangler i session state. Genindlæs siden eller vælg kolonner igen.")
                           st.stop()

                     input_cols_train = st.session_state[f"{key_prefix}_input_cols_select"]
                     output_col_train = st.session_state[f"{key_prefix}_output_col_select"]
                     date_col_train = st.session_state[f"{key_prefix}_date_col_select"]

                     if not input_cols_train or not output_col_train or not date_col_train:
                          st.error("Input/Output/Dato kolonnevalg er ugyldige. Vælg venligst igen i forbehandlingstrinnet.")
                          st.stop()

                     try:
                         with training_results_placeholder.container():
                             st.info("Starter træningsprocessen...", icon="⏳")
                             progress_bar = st.progress(0, text="Initialiserer...")
                             status_text = st.empty()

                             # 1. Klargør Data til LSTM
                             status_text.text("Opdeler data i træning/test...")
                             progress_bar.progress(5, text="Opdeler data...")
                             df_train = current_processed_data[:-forecast_steps].copy()
                             df_test = current_processed_data[-forecast_steps:].copy()

                             if df_train.empty or df_test.empty:
                                 st.error("Trænings- eller testdatasæt er tomt efter opdeling. Kan ikke træne.", icon="🚨")
                                 st.stop()

                             status_text.text("Skalerer data...")
                             progress_bar.progress(8, text="Skalerer data...")
                             scaler_X = MinMaxScaler(feature_range=(0, 1))
                             scaler_y = MinMaxScaler(feature_range=(0, 1))
                             scaled_X_train_features = scaler_X.fit_transform(df_train[input_cols_train])
                             scaled_y_train = scaler_y.fit_transform(df_train[[output_col_train]])

                             # 2. Opdel i sekvenser
                             status_text.text("Opdeler data i sekvenser...")
                             progress_bar.progress(10, text="Opdeler i sekvenser...")
                             X_train_seq, y_train_seq = split_sequences(scaled_X_train_features, scaled_y_train, lag_steps, forecast_steps)

                             if X_train_seq.nelement() == 0 or y_train_seq.nelement() == 0:
                                 st.error("Stop træning pga. problemer med sekvensopdeling (se fejl ovenfor).")
                                 progress_bar.empty(); status_text.empty()
                                 st.stop()

                             X_train_seq = X_train_seq.to(device)
                             y_train_seq = y_train_seq.to(device)

                             st.write(f"Inputsekvens form: `{X_train_seq.shape}` `(Prøver, Forsinkelsestrin, Features)`")
                             st.write(f"Outputsekvens form: `{y_train_seq.shape}` `(Prøver, Forudsigelsestrin)`")

                             # 3. Opret DataLoader
                             status_text.text("Opretter DataLoader...")
                             progress_bar.progress(15, text="Opretter DataLoader...")
                             dataset = torch.utils.data.TensorDataset(X_train_seq, y_train_seq)
                             effective_batch_size = min(batch_size, len(dataset))
                             if effective_batch_size <= 0:
                                 st.error(f"Batch størrelse ({batch_size}) er ugyldig for datasætstørrelse ({len(dataset)}).")
                                 st.stop()
                             dataloader = torch.utils.data.DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)

                             # 4. Initialiser Model, Tab, Optimizer
                             status_text.text("Initialiserer model...")
                             progress_bar.progress(20, text="Initialiserer model...")
                             input_feature_count = X_train_seq.shape[2]
                             model = LSTMForecasting(
                                 input_feature_size=input_feature_count, lstm_hidden_size=lstm_neurons,
                                 lstm_num_layers=lstm_layers, linear_num_layers=linear_hidden_layers,
                                 linear_hidden_size=linear_hidden_neurons, output_size=forecast_steps
                             ).to(device)

                             criterion = nn.MSELoss()
                             optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                             st.write("**Model Arkitektur Detaljer:**")
                             st.code(str(model))

                             # 5. Træningsloop
                             status_text.text("Starter Træningsloop...")
                             progress_bar.progress(25, text="Starter Træning...")
                             loss_text = st.empty()
                             start_time = time.time()
                             total_batches = len(dataloader)
                             losses = []

                             model.train()
                             for epoch in range(1, n_epochs + 1):
                                 epoch_loss = 0.0
                                 epoch_start_time = time.time()
                                 for i, (inputs, labels) in enumerate(dataloader):
                                     inputs, labels = inputs.to(device), labels.to(device)
                                     optimizer.zero_grad()
                                     outputs = model(inputs)
                                     loss = criterion(outputs, labels)
                                     loss.backward()
                                     optimizer.step()
                                     epoch_loss += loss.item()

                                 avg_epoch_loss = epoch_loss / total_batches
                                 losses.append(avg_epoch_loss)
                                 progress = 25 + int(70 * (epoch / n_epochs)) # 70% af bar til træning
                                 epoch_time = time.time() - epoch_start_time
                                 progress_text = f"Epoch {epoch}/{n_epochs} | Tab: {avg_epoch_loss:.6f} | Tid: {epoch_time:.2f}s"
                                 progress_bar.progress(progress, text=progress_text)
                                 # status_text.text(f"Epoch {epoch}/{n_epochs} Fuldført") # Mindre vigtigt nu med tekst i progress bar
                                 # loss_text.text(f"Epoch {epoch} Gns. Tab: {avg_epoch_loss:.6f}")

                             end_time = time.time()
                             training_time = end_time - start_time
                             status_text.success(f"Træning Fuldført på {training_time:.2f} sekunder!", icon="✅")
                             progress_bar.progress(95, text="Træning Fuldført. Plotter tab...")
                             # loss_text.empty() # Kan beholdes for at vise sidste tab
                             st.balloons()

                             # --- Plot Træningstab ---
                             fig_loss = go.Figure()
                             fig_loss.add_trace(go.Scatter(x=list(range(1, n_epochs + 1)), y=losses, mode='lines'))
                             fig_loss.update_layout(title='Træningstab pr. Epoch (MSE)', xaxis_title='Epoch', yaxis_title='MSE Tab', height=300, margin=dict(t=30, b=10))
                             st.plotly_chart(fig_loss, use_container_width=True)

                             # 6. Prediktion / Evaluering
                             status_text.text("Evaluerer på Testdatasæt...")
                             progress_bar.progress(97, text="Evaluerer...")
                             model.eval()
                             with torch.no_grad():
                                 last_train_features_scaled = scaled_X_train_features[-lag_steps:]
                                 if last_train_features_scaled.shape[0] < lag_steps:
                                     st.error(f"Kunne ikke hente nok ({lag_steps}) features fra træningsdata til prediktionsinput. Fik kun {last_train_features_scaled.shape[0]}.")
                                     st.stop()

                                 test_input_sequence = torch.from_numpy(last_train_features_scaled).float().unsqueeze(0).to(device)

                                 if test_input_sequence.shape[1] != lag_steps:
                                      st.error(f"Inputsekvens til prediktion har forkert lag-dimension ({test_input_sequence.shape[1]}), forventet {lag_steps}.", icon="🚨")
                                      st.stop()
                                 else:
                                     prediction_scaled = model(test_input_sequence)
                                     prediction_unscaled = scaler_y.inverse_transform(prediction_scaled.cpu().numpy())[0]
                                     actual_values = df_test[output_col_train].values

                                     # --- Længde Tjek ---
                                     if len(prediction_unscaled) != len(actual_values):
                                         st.warning(f"Prediktionslængde ({len(prediction_unscaled)}) matcher ikke faktisk testlængde ({len(actual_values)}). Sammenligning trimmes.", icon="⚠️")
                                         min_len = min(len(prediction_unscaled), len(actual_values))
                                         prediction_unscaled = prediction_unscaled[:min_len]
                                         actual_values = actual_values[:min_len]
                                         df_test_for_plot = df_test.iloc[:min_len]
                                     else:
                                          df_test_for_plot = df_test

                                     # Beregn Metrikker
                                     if len(actual_values) > 0:
                                          test_rmse = np.sqrt(mean_squared_error(actual_values, prediction_unscaled))
                                          test_mae = mean_absolute_error(actual_values, prediction_unscaled)
                                          final_training_loss = losses[-1]
                                          st.metric("Endeligt Træningstab (MSE):", f"{final_training_loss:.6f}")
                                          st.metric("Testdatasæt RMSE:", f"{test_rmse:.4f}")
                                          st.metric("Testdatasæt MAE:", f"{test_mae:.4f}")
                                     else:
                                          st.warning("Ingen faktiske værdier tilgængelige for sammenligning efter længdejustering.", icon="⚠️")

                                     # 7. Visualiser Prediktion
                                     with prediction_plot_placeholder.container():
                                         st.subheader(f"📊 Testdatasæt: Faktisk vs. Forudsagt ({output_col_train})")
                                         fig_pred = go.Figure()
                                         dates_test = df_test_for_plot[date_col_train]

                                         if len(dates_test) == len(actual_values) and len(dates_test) == len(prediction_unscaled):
                                             plot_steps = len(dates_test)
                                             fig_pred.add_trace(go.Scatter(x=dates_test, y=actual_values, mode='lines+markers', name='Faktisk', line=dict(color='blue'), marker=dict(size=6)))
                                             fig_pred.add_trace(go.Scatter(x=dates_test, y=prediction_unscaled, mode='lines+markers', name='Forudsagt', line=dict(color='orange', dash='dash'), marker=dict(size=6)))
                                             fig_pred.update_layout(
                                                 title=f'Faktisk vs Forudsagt - Næste {plot_steps} Trin',
                                                 xaxis_title='Dato', yaxis_title='Værdi', legend_title="Forklaring", height=450)
                                             st.plotly_chart(fig_pred, use_container_width=True)

                                             # Download Knapper
                                             col_btn1, col_btn2 = st.columns(2)
                                             with col_btn1:
                                                 try:
                                                     img_bytes = fig_pred.to_image(format="png", scale=2)
                                                     st.download_button(label="Download Prognose Plot (PNG)", data=img_bytes,
                                                                        file_name=f'prognose_plot_{output_col_train}_{plot_steps}trin.png', mime='image/png')
                                                 except Exception as img_e:
                                                     st.warning(f"Kunne ikke generere plot-billede: {img_e}. Kræver måske 'kaleido'.", icon="⚠️")
                                             with col_btn2:
                                                 pred_df = pd.DataFrame({date_col_train: dates_test, 'Faktisk': actual_values, 'Forudsagt': prediction_unscaled})
                                                 try:
                                                      csv_bytes = pred_df.to_csv(index=False).encode('utf-8')
                                                      st.download_button(label="Download Prognose Data (CSV)", data=csv_bytes,
                                                                          file_name=f'prognose_data_{output_col_train}_{plot_steps}trin.csv', mime='text/csv')
                                                 except Exception as csv_e:
                                                       st.warning(f"Kunne ikke generere CSV-data: {csv_e}", icon="⚠️")
                                         else:
                                             st.error("Uoverensstemmelse i længder mellem datoer, faktiske og forudsagte værdier forhindrer plotning.")

                         # Ryd op i hukommelse
                         del model, X_train_seq, y_train_seq, dataset, dataloader, prediction_scaled, test_input_sequence
                         gc.collect()
                         if torch.cuda.is_available(): torch.cuda.empty_cache()
                         status_text.empty()
                         progress_bar.progress(100, text="Fuldført!")
                         time.sleep(1) # Giver tid til at se "Fuldført"
                         progress_bar.empty()

                     except Exception as e:
                          training_results_placeholder.error(f"En fejl opstod under træning eller prediktion: {e}", icon="🚨")
                          st.error(traceback.format_exc())
                          gc.collect()
                          if torch.cuda.is_available(): torch.cuda.empty_cache()
                          if 'progress_bar' in locals(): progress_bar.empty()
                          if 'status_text' in locals(): status_text.empty()
                          if 'loss_text' in locals(): loss_text.empty()

        else:
            # Vis denne besked hvis ingen forbehandlede data er tilgængelige
            st.info("Forbehandl data ved hjælp af kontrollerne til venstre for at aktivere modeltræning.")
            training_results_placeholder.empty()
            prediction_plot_placeholder.empty()

# --- Footer ---
st.markdown("---")
st.caption("Udviklet med Streamlit, PyTorch, Statsmodels, Plotly og Pandas.")