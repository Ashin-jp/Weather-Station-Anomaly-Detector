"""Streamlit interface for a historical, non-destructive weather quality audit."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.data_generator import ensure_dataset, apply_fault, make_upload_test_dataset, FAULTS
from src.features import extract_features, validate_data
from src.model import load_model, audit_data
from src.evaluator import evaluate

st.set_page_config(page_title='Weather Station Data Quality Anomaly Detector', page_icon='🌦️', layout='wide')
st.markdown('''<style>
[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    color: var(--text-color);
    border: 1px solid var(--text-color);
    border-color: color-mix(in srgb, var(--text-color) 20%, transparent);
    padding: 16px;
    border-radius: 12px;
}
/* Leave room for Streamlit's fixed toolbar above the first line. */
.block-container {padding-top: 5rem;}
</style>''', unsafe_allow_html=True)
st.markdown('**WEATHER INTELLIGENCE / HISTORICAL QUALITY ASSURANCE**')
st.title('Weather Station Data Quality Anomaly Detector')
st.write('Find suspicious telemetry. Preserve every reading. Keep a human in the review loop.')

@st.cache_resource
def bootstrap():
    return ensure_dataset(), load_model()

with st.spinner('Preparing six years of weather and the calibrated 240-tree detector…'):
    baseline, artifact = bootstrap()

st.sidebar.title('Audit controls')
st.sidebar.caption('Temperatures: °C · Rainfall: mm/day')
custom = st.sidebar.toggle('Use custom sensitivity', value=False)
threshold = st.sidebar.slider('Alert probability threshold', .01, 1., float(artifact['threshold']), .01, disabled=not custom)
if not custom:
    threshold = artifact['threshold']
st.sidebar.info(f'Active threshold: {threshold:.3f}\n\nLower thresholds produce more alerts.')
st.sidebar.caption('Batch auditing only. Synthetic NY demonstration data; no live station feed.')
overview, lab, evidence, learn = st.tabs(['Station Overview', 'Fault Laboratory', 'Model Evidence', 'Learn & Present'])


def chart(frame, title):
    fig = go.Figure()
    for station, g in frame.groupby('station_id'):
        for field, dash in [('tmax', 'solid'), ('tmin', 'dot')]:
            fig.add_trace(go.Scatter(x=g.date, y=g[field], name=f'{station} · {field}', mode='lines',
                                     line={'width': 1.5, 'dash': dash}))
    flagged = frame[frame.final_alert] if 'final_alert' in frame else frame.iloc[:0]
    if len(flagged):
        # Missing-only records still get a visible marker without changing their raw values.
        y = flagged.tmax.fillna(flagged.tmin).fillna(0)
        fig.add_trace(go.Scatter(x=flagged.date, y=y, mode='markers', name='Review alert',
                                marker={'size': 9, 'color': ['#d94b46' if v else '#d89b21' for v in flagged.rule_flag]},
                                text=flagged.diagnostic_clues, customdata=flagged.rf_fault_prob,
                                hovertemplate='%{x}<br>%{text}<br>Fault probability: %{customdata:.3f}<extra></extra>'))
    fig.update_layout(title=title, height=410, yaxis_title='Temperature (°C)',
                      legend={'orientation': 'h', 'y': -0.25}, margin={'t': 45, 'b': 90})
    return fig

with overview:
    st.subheader('A clear view of station health')
    source = st.radio('Data source', ['Built-in NY Dataset', 'Custom CSV Upload'], horizontal=True)
    raw = baseline
    if source == 'Custom CSV Upload':
        st.download_button('Download CSV template', baseline.head(12).to_csv(index=False), 'weather_template.csv', 'text/csv')
        test_data, answer_key = make_upload_test_dataset()
        sample_col, key_col = st.columns(2)
        sample_col.download_button('Download anomaly test CSV', test_data.to_csv(index=False),
                                   'ny_anomaly_upload_test.csv', 'text/csv')
        key_col.download_button('Download test answer key', answer_key.to_csv(index=False),
                                'ny_anomaly_answer_key.csv', 'text/csv')
        st.caption('Synthetic 2023 sample: 4 stations, all 6 fault types at NY_CENTRAL_PARK. '
                   'Upload the test CSV; use the separate answer key to check results. Some faults may be missed.')
        upload = st.file_uploader('Daily CSV: date, station_id, tmax, tmin, rainfall', type=['csv'])
        raw = None
        if upload is not None:
            try:
                candidate = pd.read_csv(upload)
                # Internal simulation identifiers cannot be supplied via the upload route.
                if 'scenario_id' in candidate:
                    raise ValueError('scenario_id is reserved; remove it from your upload.')
                raw = validate_data(candidate)
            except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                st.error(str(exc))
    if raw is not None:
        audited = audit_data(raw, artifact, threshold)
        left, right = st.columns([1, 2])
        stations = left.multiselect('Stations', sorted(raw.station_id.unique()), default=sorted(raw.station_id.unique())[:1])
        dates = right.date_input('Date range', value=(raw.date.min().date(), raw.date.max().date()),
                                 min_value=raw.date.min().date(), max_value=raw.date.max().date())
        if len(dates) == 2:
            view = audited[audited.station_id.isin(stations) & audited.date.between(pd.Timestamp(dates[0]), pd.Timestamp(dates[1]))]
            columns = st.columns(5)
            values = [len(view), int(view.final_alert.sum()), int(view.rule_flag.sum()), int(view.ml_flag.sum()),
                      f'{100*(1-view.final_alert.mean()):.1f}%' if len(view) else '—']
            for col, label, value in zip(columns, ['Total Records', 'Total Alerts', 'Rule-Triggered Alerts', 'ML-Triggered Alerts', 'Clean Record %'], values):
                col.metric(label, value)
            st.caption('Rule and ML counts may overlap. Filters affect display/export; all available peers contribute to the audit. “Clean” means not flagged, not proven fault-free.')
            st.plotly_chart(chart(view, 'Daily temperatures & review alerts'), width='stretch')
            st.dataframe(view, hide_index=True, width='stretch')
            st.download_button('Download Analysed CSV', view.to_csv(index=False), 'weather_audit.csv', 'text/csv')
        else:
            st.info('Choose both ends of the date range.')

with lab:
    st.subheader('Put the detector to the test')
    st.caption('Each experiment starts from a fresh baseline. All four stations remain available as peers.')
    c1, c2, c3 = st.columns(3)
    station = c1.selectbox('Target station', sorted(baseline.station_id.unique()))
    mode = c2.selectbox('Fault mode', FAULTS, format_func=lambda s: s.replace('_', ' ').title())
    start = c3.date_input('Fault start', pd.Timestamp('2023-07-10').date(), min_value=baseline.date.min().date(),
                           max_value=pd.Timestamp('2023-12-16').date())
    magnitude = st.number_input('Magnitude (°C; °C/day for drift; negative for negative rain)', value=.4 if mode == 'drift' else 25., step=.1)
    duration = st.slider('Duration in days (spikes use one day)', 1, 15, 5)
    if st.button('Inject Fault & Run Detector', type='primary'):
        window = baseline[baseline.date.between(pd.Timestamp(start)-pd.Timedelta(days=35), pd.Timestamp(start)+pd.Timedelta(days=30))].reset_index(drop=True)
        altered = apply_fault(window, station, start, mode, magnitude, duration)
        before, after = audit_data(window, artifact, threshold), audit_data(altered, artifact, threshold)
        st.session_state['experiment'] = (window, altered, before, after, station)
    if 'experiment' in st.session_state:
        window, altered, before, after, target = st.session_state['experiment']
        a, b = st.columns(2)
        a.plotly_chart(chart(before[before.station_id == target], 'Before injection'), width='stretch')
        b.plotly_chart(chart(after[after.station_id == target], 'After injection'), width='stretch')
        changed = altered.is_fault.eq(1)
        st.dataframe(after.loc[changed, ['date', 'rf_fault_prob', 'rule_reasons', 'final_alert', 'diagnostic_clues']], hide_index=True)
        delta = extract_features(altered)-extract_features(window)
        st.write('Mean feature changes over injected rows')
        st.dataframe(delta.loc[changed].mean().rename('Mean delta').to_frame(), width='stretch')
        st.caption('Results belong to the last submitted experiment; press the button to apply changed controls.')

with evidence:
    st.subheader('Evidence from an unseen year')
    st.write('Train: **2018–2021** → Calibrate: **2022** → Test: **2023**')
    st.warning('All evaluation labels are synthetically injected. These scores do not establish performance on real sensor faults.')
    result = evaluate(audit_data(artifact['test_data'], artifact, threshold))
    for col, (label, value) in zip(st.columns(4), result['metrics'].items()):
        col.metric(label, f'{value:.3f}' if value is not None else 'N/A')
    st.caption(f'Hybrid alerts at threshold {threshold:.3f}. ROC-AUC uses RF probabilities. Clean-row false positive rate: {result["clean_false_positive_rate"]:.2%}.')
    a, b = st.columns(2)
    fig = px.imshow(result['confusion'], x=['Predicted clean', 'Predicted fault'], y=['Actually clean', 'Actually fault'],
                    text_auto=True, color_continuous_scale='Blues', title='2023 confusion matrix')
    a.plotly_chart(fig, width='stretch')
    b.dataframe(result['faults'], hide_index=True, width='stretch')
    st.bar_chart(pd.Series(artifact['model'].feature_importances_, index=artifact['features']).sort_values(), horizontal=True)

with learn:
    st.subheader('Context makes the difference')
    for col, title, body in zip(st.columns(4), ['01 · Preserve', '02 · Describe', '03 · Detect', '04 · Review'],
                               ['Validate the CSV and retain original readings.', 'Calculate 17 temporal, spatial and missingness features.',
                                'Combine a 240-tree forest with physical rules.', 'Explain suspicious records and export an annotated copy.']):
        col.info(f'**{title}**\n\n{body}')
    comparison = st.radio('Compare weather and sensor behavior', ['Regional heatwave', 'Single-station spike'], horizontal=True)
    sample = baseline[baseline.date.between('2023-07-01', '2023-07-31')].copy().reset_index(drop=True)
    if comparison == 'Single-station spike':
        sample = apply_fault(sample, 'NY_CENTRAL_PARK', '2023-07-15', 'spike', 25, 1)
    comparison_audit = audit_data(sample, artifact, threshold)
    st.plotly_chart(chart(comparison_audit, comparison), width='stretch')
    st.caption('The baseline contains a shared July 12–18 heatwave. Peer agreement helps distinguish it from a localized fault; predictions can still be wrong.')
    questions = [
        ('Why not flag every hot day?', 'Regional extremes are plausible weather. Spatial agreement and temporal behavior provide context.'),
        ('Why Random Forest?', 'It handles nonlinear feature interactions and mixed feature scales without standardization.'),
        ('Why a temporal split?', 'Training on earlier years and testing on a later year avoids random-split leakage between neighboring days.'),
        ('How is the threshold chosen?', '2022 RF probabilities select the threshold maximizing binary F1. 2023 never selects the threshold.'),
        ('Are missing readings replaced?', 'Only the ML feature copy is imputed using local medians, seasonal medians and fixed fallbacks. Audit values remain missing.'),
        ('What if only one station is uploaded?', 'Neighbor maximum differences fall back to local rolling residuals; minimum differences use zero. Spatial evidence is weaker.'),
        ('What do diagnostic clues mean?', 'They rank feature deviations from clean training statistics weighted by global importance. They are heuristic clues, not causal explanations or SHAP values.'),
        ('Can this run in real time?', 'No. Forward-day differences use future observations, so this implementation is for retrospective batch audits.'),
        ('What are the main limitations?', 'Synthetic labels, a New York climatology and correlated peers limit generalization. Early drift and the first flatline day may be missed.'),
        ('What should happen after an alert?', 'A reviewer checks metadata, neighboring observations and maintenance logs. No original reading is automatically changed or deleted.')]
    for question, answer in questions:
        with st.expander(question):
            st.write(answer)
