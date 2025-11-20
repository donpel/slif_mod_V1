# slif_mod_V1

## Apžvalga
Repozitorijoje yra du Python 3 scenarijai, skirti faneros briaunos juostiniam šlifavimui modeliuoti ir analizuoti naudojant pateiktus eksperimentinius duomenis. 
- `plywood_sanding.py` – paprastas regresinis modelis ir scenarijus, skirtas pamatiniams ryšiams tarp MRR ir normaliosios jėgos nustatyti.
- `detailed_sanding_model.py` – keturių būstavų ODE modelis (z, T, W, Ra) su kampo, abrazyvo grūdėtumo, kietumo ir nusidėvėjimo poveikiais bei automatinėmis parametrų pritaikymo procedūromis.

## Priklausomybės
Abi bylos naudoja standartinį mokslinį Python paketą:
- `numpy`
- `pandas`
- `matplotlib` (naudoja neinteraktyvų `Agg` išvesčių tipą)
- `scipy`

## Eksperimentiniai duomenys
Abu scenarijai turi vidinėje kodo dalyje užfaiksuotus eksperimentinius šlifavimo duomenis (P180 abrazyvas, 5 mm nušlifavimo gylis). Stulpeliai:
- `weight_g` – papilomai pridėtas svoris (g)
- `time_s` – laikas (s) nušlifuoti 5 mm
- `Ft_N`, `FN_N` – pjovimo ir normaliąją jėgą (N)
- `speed_mm_min` – šalinimo greitis (mm/min)
Išvestiniai stulpeliai apima `MRR_mm_per_min`, `MRR_mm_per_s` ir jėgų santykį `ratio_Ft_to_FN`.

## Ką daro kiekvienas scenarijus
### `plywood_sanding.py`
- Sukuria `DataFrame` su išvestiniais MRR ir Ft/FN santykiais.
- Atlieka linijinę regresiją `MRR = a * FN + b` ir išveda R².
- Braižo ir saugo PNG grafikus: `mrr_vs_FN.png`, `Ft_vs_FN.png`, `ratio_vs_FN.png`.
- Modeliuoja 30 s šlifavimo scenarijus dviem normaliųjų jėgų reikšmėms (18.78615 N ir 21.69972 N), pateikia nušlifuotą storį, tūrį ir masę.

### `detailed_sanding_model.py`
- Aprašo keturių būstavų ODE (z, T, W, Ra) su kampo, grūdėtumo, kietumo ir nusidėvėjimo funkcijomis.
- Atlieka dvi pritaikymo pakopas: linijinė MRR(FN) regresija ir nelinijinis `k_m`, `a_FN`, `a_v` derinimas naudojant `curve_fit`.
- Skaičiuoja kokybinius rodiklius (MAE, RMSE, MAPE, R²) tiek regresijai, tiek ODE pagrindu paremtam MRR.
- Simuliuoja 30 s scenarijus 90° ir 45° briaunoms toms pačioms jėgoms; kiekvienam scenarijui pateikiamas nušlifuotas storis, tūris, masė, temperaturos ir nusidėvėjimo būstai.
- Saugo laikinių eilučių grafikus (formatas `right_angle_FN_<force>.png`, `bevel_edge_FN_<force>.png`) bei MRR grafikus `mrr_vs_FN.png` ir `mrr_validation.png`.

## Paleidimo instrukcijos
1. Įdiekite priklausomybes (pvz., `pip install numpy pandas matplotlib scipy`).
2. Pagrindiniai paleidimai:
   - `python plywood_sanding.py` – suformuoja duomenų lentelę, išveda regresijos koeficientus, išsaugo 3 PNG ir parodo 30 s simuliacijos rezultatus.
   - `python detailed_sanding_model.py` – atspausdina duomenis, regresijos ir ODE pritaikymo parametrus su metrikomis, sugeneruoja MRR grafikus, paleidžia 90° ir 45° scenarijų simuliacijas bei išsaugo laikinių eilučių PNG failus.

## Pritaikymas ir plėtra
- Pagrindinės konstantos ir pradiniai būstai yra aiškiai nurodyti kiekvieno failo viršuje; modifikuokite juos, jei reikia kalibracijos.
- `detailed_sanding_model.py` turi dataklases `ModelParameters` ir `Inputs`, kad būtų lengviau keisti modelio parametrus ir įvestis ateityje (pvz., temperatūros ar ruplėtumo modelio plėtra).
