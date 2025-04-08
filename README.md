# LSTM Streamlit - Tidsserieprognose App
https://bhupas-tidsserieprognose-app-app-js0pas.streamlit.app/
## Understøttede Prognosetyper
- ✅ Univariat Flertrins-Prognose
- ✅ Multivariat Flertrins-Prognose

## Understøttede Modeller
- ✅ LSTM (Baseret på PyTorch, Understøtter både GPU & CPU)

## Indbyggede Funktioner
- 🛠️ Dataforbehandling:
  - Lineær Interpolation
  - Baglæns Opfyldning (Backward Fill)
- 🔢 Korrelationsmatrix
- 🔄 Sæsondekomponering:
  - Trend
  - Sæsonalitet
- 🧠 Brugerdefineret Modelkonfiguration
- 📈 Plotly-baserede grafer
- 💾 Download Prognosegraf

## Understøttede Datatyper
- 📆 Dato - `datetime`
- 📥 Input Features (Inputkolonner) - `int`, `float`
- 📤 Output Feature (Målkolonne) - `int`, `float`

## LSTM Model Egenskaber
- 🔙 Forsinkelsestrin (Lag Steps)
- 🔜 Forudsigelsestrin (Forecast Steps)
- Brugerdefinerede LSTM- og Tætte Lag (Dense Layers)
- ⏱️ Antal Epochs
- 📦 Batch Størrelse
