import { useEffect, useMemo, useState } from "react";

import { extractInputs, getAntennaCatalog, getModelInfo, predictPoint, predictSweep } from "./api/client";
import NumberField from "./components/NumberField";
import ParagraphInput from "./components/ParagraphInput";
import S11Chart from "./components/S11Chart";

const INITIAL_VALUES = {
  gap: "39.47",
  surface_width: "75",
  surface_length: "75",
  epsilon_r: "5",
  frequency: "2.4",
  start_frequency: "0.5",
  end_frequency: "4",
  points: "301",
  threshold_db: "-10",
};

const FIELD_LABELS = {
  antenna_family_id: "grande famille",
  antenna_variant: "antenne / variante",
  gap: "gap",
  surface_width: "largeur du substrat",
  surface_length: "longueur du substrat",
  epsilon_r: "permittivité relative",
  frequency: "fréquence",
  start_frequency: "fréquence de début",
  end_frequency: "fréquence de fin",
  points: "nombre de points",
  threshold_db: "seuil",
};

const WORKFLOW_STEPS = [
  { number: 1, label: "Antenne" },
  { number: 2, label: "Paramètres" },
  { number: 3, label: "Résultat" },
  { number: 4, label: "Courbe S11" },
];

const RANGE_NAMES = {
  gap: "gap",
  surface_width: "largeur",
  surface_length: "longueur",
  epsilon_r: "epsilon r",
  FREQUENCY: "fréquence",
};

const asNumber = (value) => Number(value);
const geometryPayload = (values) => ({
  gap: asNumber(values.gap),
  surface_width: asNumber(values.surface_width),
  surface_length: asNumber(values.surface_length),
  epsilon_r: asNumber(values.epsilon_r),
});

function findRangeWarnings(ranges, candidates) {
  if (!ranges) return [];

  return Object.entries(candidates).flatMap(([feature, featureValues]) => {
    const range = ranges[feature];
    if (!range) return [];

    const outside = featureValues.some(
      (value) => Number.isFinite(value) && (value < range.min || value > range.max),
    );
    return outside
      ? [`${RANGE_NAMES[feature]} hors de [${range.min}, ${range.max}]`]
      : [];
  });
}

export default function App() {
  const [values, setValues] = useState(INITIAL_VALUES);
  const [inputMode, setInputMode] = useState("form");
  const [paragraph, setParagraph] = useState("");
  const [extraction, setExtraction] = useState(null);
  const [selection, setSelection] = useState({ familyId: "", antenna: "" });
  const [activeStep, setActiveStep] = useState(1);
  const [modelInfo, setModelInfo] = useState(null);
  const [antennaCatalog, setAntennaCatalog] = useState(null);
  const [pointResult, setPointResult] = useState(null);
  const [sweepResult, setSweepResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState("");

  useEffect(() => {
    Promise.all([getModelInfo(), getAntennaCatalog()])
      .then(([model, catalog]) => {
        setModelInfo(model);
        setAntennaCatalog(catalog);
      })
      .catch((requestError) => setError(requestError.message));
  }, []);

  const antennaFamilies = antennaCatalog?.families || [];
  const selectedFamily = antennaFamilies.find(
    (family) => family.id === selection.familyId,
  );
  const hasAntenna = Boolean(selectedFamily && selection.antenna);
  const selectedAntennaLabel = hasAntenna
    ? `${selectedFamily.name} · ${selection.antenna}`
    : "";

  const ranges = modelInfo?.training_ranges;
  const geometryCandidates = useMemo(
    () => ({
      gap: [asNumber(values.gap)],
      surface_width: [asNumber(values.surface_width)],
      surface_length: [asNumber(values.surface_length)],
      epsilon_r: [asNumber(values.epsilon_r)],
    }),
    [values.epsilon_r, values.gap, values.surface_length, values.surface_width],
  );
  const pointWarnings = useMemo(
    () =>
      findRangeWarnings(ranges, {
        ...geometryCandidates,
        FREQUENCY: [asNumber(values.frequency)],
      }),
    [geometryCandidates, ranges, values.frequency],
  );
  const sweepWarnings = useMemo(
    () =>
      findRangeWarnings(ranges, {
        ...geometryCandidates,
        FREQUENCY: [
          asNumber(values.start_frequency),
          asNumber(values.end_frequency),
        ],
      }),
    [geometryCandidates, ranges, values.end_frequency, values.start_frequency],
  );

  const highestStep = !hasAntenna ? 1 : pointResult ? 4 : 2;

  const missingExtractedFields = extraction
    ? Object.keys(FIELD_LABELS).filter((name) => {
        if (name === "antenna_family_id") return !selection.familyId;
        if (name === "antenna_variant") return !selection.antenna;
        return values[name] === "";
      })
    : [];

  function changeInputMode(mode) {
    setInputMode(mode);
    setError("");
  }

  async function handleExtraction(event) {
    event.preventDefault();
    setError("");
    setLoading("extract");
    try {
      const result = await extractInputs(paragraph.trim());
      const fields = result.fields;
      setValues(Object.fromEntries(
        Object.keys(INITIAL_VALUES).map((name) => [
          name, fields[name] == null ? "" : String(fields[name]),
        ]),
      ));
      setSelection({
        familyId: fields.antenna_family_id || "",
        antenna: fields.antenna_variant || "",
      });
      resetPredictions();
      setExtraction(result);
      setInputMode("form");
      setActiveStep(fields.antenna_family_id && fields.antenna_variant ? 2 : 1);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading("");
    }
  }

  function resetPredictions() {
    setPointResult(null);
    setSweepResult(null);
    setError("");
  }

  function handleFamilyChange(event) {
    setSelection({ familyId: event.target.value, antenna: "" });
    resetPredictions();
    setActiveStep(1);
  }

  function handleAntennaChange(event) {
    setSelection((current) => ({ ...current, antenna: event.target.value }));
    resetPredictions();
    setActiveStep(1);
  }

  function handleSelectionSubmit(event) {
    event.preventDefault();
    if (!hasAntenna) return;
    setError("");
    setActiveStep(2);
  }

  function handleChange(event) {
    const { name, value } = event.target;
    setValues((current) => ({ ...current, [name]: value }));
    setError("");

    if (["gap", "surface_width", "surface_length", "epsilon_r", "frequency"].includes(name)) {
      setPointResult(null);
      setSweepResult(null);
    } else {
      setSweepResult(null);
    }
  }

  async function handlePointPrediction(event) {
    event.preventDefault();
    setError("");
    setLoading("point");
    try {
      const result = await predictPoint({
        antenna_family_id: selection.familyId,
        antenna_variant: selection.antenna,
        ...geometryPayload(values),
        frequency: asNumber(values.frequency),
      });
      setPointResult(result);
      setSweepResult(null);
      setActiveStep(3);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading("");
    }
  }

  async function handleSweep(event) {
    event.preventDefault();
    setError("");
    setLoading("sweep");
    try {
      setSweepResult(
        await predictSweep({
          antenna_family_id: selection.familyId,
          antenna_variant: selection.antenna,
          ...geometryPayload(values),
          start_frequency: asNumber(values.start_frequency),
          end_frequency: asNumber(values.end_frequency),
          points: asNumber(values.points),
          threshold_db: asNumber(values.threshold_db),
        }),
      );
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading("");
    }
  }

  function navigateToStep(step) {
    if (step > highestStep) return;
    setError("");
    setActiveStep(step);
  }

  const hint = (feature, unit = "") =>
    ranges?.[feature]
      ? `Plage ML : ${ranges[feature].min}–${ranges[feature].max}${unit}`
      : "Chargement de la plage...";

  return (
    <main>
      <header className="hero shell">
        <div>
          <p className="eyebrow"><span /> ANTENNA INTELLIGENCE PLATFORM</p>
          <h1>Explorez la réponse <em>S11</em> de votre antenne.</h1>
          <p className="hero__copy">
            Choisissez une antenne, définissez sa géométrie puis visualisez sa
            réponse fréquentielle avec le modèle XGBoost.
          </p>
        </div>
        <div className="model-status">
          <span className={modelInfo && antennaCatalog ? "status-dot online" : "status-dot"} />
          <div>
            <strong>{modelInfo && antennaCatalog ? "Service opérationnel" : "Connexion au service"}</strong>
            <small>
              {modelInfo
                ? `${modelInfo.effective_rows.toLocaleString("fr-FR")} entrées uniques · ${modelInfo.data_quality.geometry_count} géométries`
                : "Veuillez patienter"}
            </small>
          </div>
        </div>
      </header>

      <div className="input-mode shell" role="group" aria-label="Mode de saisie">
        <button type="button" aria-pressed={inputMode === "form"} disabled={Boolean(loading)} onClick={() => changeInputMode("form")}>
          Remplir le formulaire
        </button>
        <button type="button" aria-pressed={inputMode === "paragraph"} disabled={Boolean(loading)} onClick={() => changeInputMode("paragraph")}>
          Décrire avec un paragraphe
        </button>
      </div>

      {inputMode === "form" && <WorkflowStepper
        currentStep={activeStep}
        highestStep={highestStep}
        onStepChange={navigateToStep}
      />}

      <div className="workflow-content shell">
        {inputMode === "paragraph" ? (
          <ParagraphInput
            paragraph={paragraph}
            onChange={(text) => { setParagraph(text); setError(""); }}
            onSubmit={handleExtraction}
            loading={loading === "extract"}
            error={error}
          />
        ) : <>
        {extraction && (
          <div className="extraction-review" role="status">
            <Notice kind="info" title="Paramètres extraits — vérifiez les valeurs">
              {missingExtractedFields.length
                ? `À compléter dans les étapes Antenne, Paramètres ou Courbe S11 : ${missingExtractedFields.map((name) => FIELD_LABELS[name]).join(", ")}.`
                : "Tous les champs sont renseignés. Vous pouvez les corriger avant de lancer les calculs."}
            </Notice>
            {extraction.warnings.length > 0 && (
              <Notice kind="warning" title="Points à vérifier">{extraction.warnings.join(" ")}</Notice>
            )}
          </div>
        )}
        {activeStep === 1 && (
          <form className="panel workflow-panel antenna-panel" onSubmit={handleSelectionSubmit}>
            <PanelHeading
              title="Choisissez votre antenne"
              subtitle={antennaCatalog
                ? `${antennaCatalog.family_count} familles et ${antennaCatalog.antenna_count} variantes chargées depuis BOT.xlsx`
                : "Chargement du catalogue BOT.xlsx..."}
            />

            <div className="antenna-fields">
              <label className="field" htmlFor="antenna-family">
                <span className="field__label">Grande famille</span>
                <select
                  id="antenna-family"
                  value={selection.familyId}
                  onChange={handleFamilyChange}
                  required
                >
                  <option value="">Sélectionner une famille</option>
                  {antennaFamilies.map((family) => (
                    <option key={family.id} value={family.id}>
                      {family.name}
                    </option>
                  ))}
                </select>
                <span className="field__hint">Première étape obligatoire</span>
              </label>

              <label className="field" htmlFor="antenna-variant">
                <span className="field__label">Antenne / variante</span>
                <select
                  id="antenna-variant"
                  value={selection.antenna}
                  onChange={handleAntennaChange}
                  disabled={!selectedFamily}
                  required
                >
                  <option value="">
                    {selectedFamily ? "Sélectionner une variante" : "Choisissez d'abord une famille"}
                  </option>
                  {selectedFamily?.antennas.map((antenna) => (
                    <option key={antenna} value={antenna}>
                      {antenna}
                    </option>
                  ))}
                </select>
                <span className="field__hint">
                  {selectedFamily
                    ? `${selectedFamily.antennas.length} variante(s) disponible(s)`
                    : "La liste dépend de la famille"}
                </span>
              </label>
            </div>

            {hasAntenna && (
              <div className="selection-preview">
                <span>Antenne sélectionnée</span>
                <strong>{selection.antenna}</strong>
                <small>{selectedFamily.name}</small>
              </div>
            )}

            <Notice kind="info" title="Périmètre du modèle actuel">
              {modelInfo?.antenna_scope?.verified
                ? "Le modèle chargé est associé à une antenne vérifiée. Les autres variantes seront refusées par l'API."
                : "La famille du dataset actuel n'est pas encore identifiée. Les estimations restent internes et doivent être confirmées par CST ou par mesure."}
            </Notice>

            {error && <Notice kind="error" title="Erreur de connexion">{error}</Notice>}

            <button className="primary-button" type="submit" disabled={!hasAntenna}>
              Continuer vers les paramètres
            </button>
          </form>
        )}

        {activeStep === 2 && (
          <form className="panel workflow-panel controls" onSubmit={handlePointPrediction}>
            <SelectedAntenna
              label={selectedAntennaLabel}
              onChange={() => navigateToStep(1)}
            />
            <PanelHeading
              title="Paramètres du modèle"
              subtitle="Renseignez la géométrie et la fréquence de prédiction"
            />

            <div className="field-grid">
              <NumberField label="Gap" name="gap" value={values.gap} onChange={handleChange} step="any" min="0.000001" required hint={hint("gap")} />
              <NumberField label="Largeur substrat" name="surface_width" value={values.surface_width} onChange={handleChange} step="any" min="0.000001" required hint={hint("surface_width")} />
              <NumberField label="Longueur substrat" name="surface_length" value={values.surface_length} onChange={handleChange} step="any" min="0.000001" required hint={hint("surface_length")} />
              <NumberField label="Permittivité relative" name="epsilon_r" value={values.epsilon_r} onChange={handleChange} step="any" min="0.000001" required hint={hint("epsilon_r")} />
            </div>

            <div className="divider" />
            <PanelHeading
              title="Fréquence précise"
              subtitle="Valeur utilisée pour la première estimation S11"
              compact
            />
            <NumberField label="Fréquence" name="frequency" value={values.frequency} onChange={handleChange} step="any" min="0.000001" required hint={hint("FREQUENCY", " GHz")} />

            {pointWarnings.length > 0 && (
              <Notice kind="warning" title="Avertissement de domaine">
                {pointWarnings.join(" · ")}
              </Notice>
            )}
            {error && <Notice kind="error" title="Erreur">{error}</Notice>}

            <div className="step-actions">
              <button className="ghost-button" type="button" onClick={() => navigateToStep(1)}>
                Retour
              </button>
              <button className="primary-button" type="submit" disabled={Boolean(loading)}>
                {loading === "point" ? "Calcul en cours..." : "Prédire S11"}
              </button>
            </div>
          </form>
        )}

        {activeStep === 3 && pointResult && (
          <section className="panel workflow-panel results" aria-live="polite">
            <SelectedAntenna
              label={selectedAntennaLabel}
              onChange={() => navigateToStep(1)}
            />
            <PanelHeading
              title="Résultat de la prédiction"
              subtitle="Estimation instantanée du modèle"
            />

            <div className="reading">
              <span>S11 estimé</span>
              <strong>{pointResult.s11_db.toFixed(4)}</strong>
              <b>dB</b>
            </div>
            <p className="reading__caption">
              Prédiction calculée à {asNumber(values.frequency).toFixed(4)} GHz
            </p>

            <div className="quality-grid">
              <div>
                <span>Erreur absolue P95</span>
                <strong>
                  {pointResult.estimated_absolute_error_p95_db != null
                    ? `${pointResult.estimated_absolute_error_p95_db.toFixed(3)} dB`
                    : "Non disponible"}
                </strong>
              </div>
              <div>
                <span>Géométrie déjà entraînée</span>
                <strong>{pointResult.known_training_geometry ? "Oui" : "Non"}</strong>
              </div>
              <div>
                <span>Validation</span>
                <strong>{pointResult.model_scope_verified ? "Périmètre vérifié" : "Interne uniquement"}</strong>
              </div>
            </div>

            {(pointWarnings.length > 0 || pointResult.warnings?.length > 0) && (
              <Notice kind="warning" title="Avertissement de validité">
                {[...new Set([...(pointResult.warnings || []), ...pointWarnings])].join(" · ")}
              </Notice>
            )}

            <div className="step-actions">
              <button className="ghost-button" type="button" onClick={() => navigateToStep(2)}>
                Modifier les paramètres
              </button>
              <button className="primary-button" type="button" onClick={() => navigateToStep(4)}>
                Continuer vers la courbe
              </button>
            </div>
          </section>
        )}

        {activeStep === 4 && pointResult && (
          <section className="panel sweep">
            <SelectedAntenna
              label={selectedAntennaLabel}
              onChange={() => navigateToStep(1)}
            />
            <div className="sweep__header">
              <PanelHeading
                title="Sweep fréquentiel"
                subtitle="Analyse de S11 sur une plage continue"
              />
              {sweepResult && (
                <div className="minimum-card">
                  <span>Minimum prédit</span>
                  <strong>{sweepResult.minimum.s11.toFixed(3)} dB</strong>
                  <small>{sweepResult.minimum.frequency.toFixed(4)} GHz</small>
                </div>
              )}
            </div>

            <form className="sweep__controls" onSubmit={handleSweep}>
              <NumberField label="Début (GHz)" name="start_frequency" value={values.start_frequency} onChange={handleChange} step="any" min="0.000001" required />
              <NumberField label="Fin (GHz)" name="end_frequency" value={values.end_frequency} onChange={handleChange} step="any" min="0.000001" required />
              <NumberField label="Points" name="points" value={values.points} onChange={handleChange} step="1" min="10" max="2001" required />
              <NumberField label="Seuil (dB)" name="threshold_db" value={values.threshold_db} onChange={handleChange} step="any" min="-100" max="0" required />
              <button className="secondary-button" type="submit" disabled={Boolean(loading)}>
                {loading === "sweep" ? "Génération..." : "Générer la courbe"}
              </button>
            </form>

            {(sweepWarnings.length > 0 || sweepResult?.warnings?.length > 0) && (
              <Notice kind="warning" title="Avertissement de validité">
                {[...new Set([...(sweepResult?.warnings || []), ...sweepWarnings])].join(" · ")}
              </Notice>
            )}
            {error && <Notice kind="error" title="Erreur">{error}</Notice>}

            {sweepResult && (
              <div className="band-summary">
                <div>
                  <span>Seuil analysé</span>
                  <strong>{sweepResult.threshold_db.toFixed(1)} dB</strong>
                </div>
                <div>
                  <span>Bandes détectées</span>
                  <strong>{sweepResult.bands.length}</strong>
                </div>
                <div>
                  <span>Bande passante totale</span>
                  <strong>{sweepResult.total_bandwidth.toFixed(4)} GHz</strong>
                </div>
              </div>
            )}

            <S11Chart
              curve={sweepResult?.curve || []}
              thresholdDb={sweepResult?.threshold_db ?? asNumber(values.threshold_db)}
            />

            <div className="step-actions step-actions--end">
              <button className="ghost-button" type="button" onClick={() => navigateToStep(3)}>
                Retour au résultat
              </button>
            </div>
          </section>
        )}
        </>}
      </div>

      <footer className="shell">
        <span>Antenna Lab</span>
        <p>Les prédictions hors des plages d'entraînement sont signalées.</p>
      </footer>
    </main>
  );
}

function WorkflowStepper({ currentStep, highestStep, onStepChange }) {
  return (
    <nav className="workflow-stepper shell" aria-label="Étapes de la simulation">
      <ol>
        {WORKFLOW_STEPS.map((step) => {
          const isActive = step.number === currentStep;
          const isComplete = step.number < currentStep;
          const isAvailable = step.number <= highestStep;

          return (
            <li
              className={`${isActive ? "active" : ""} ${isComplete ? "complete" : ""}`}
              key={step.number}
            >
              <button
                type="button"
                onClick={() => onStepChange(step.number)}
                disabled={!isAvailable}
                aria-current={isActive ? "step" : undefined}
                aria-label={`Étape ${step.number} : ${step.label}`}
              >
                <span>{step.number}</span>
                <small>{step.label}</small>
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

function SelectedAntenna({ label, onChange }) {
  return (
    <div className="selected-antenna">
      <div>
        <span>Antenne sélectionnée</span>
        <strong>{label}</strong>
      </div>
      <button type="button" onClick={onChange}>Changer</button>
    </div>
  );
}

function PanelHeading({ title, subtitle, compact = false }) {
  return (
    <div className={`panel__heading${compact ? " compact" : ""}`}>
      <div><h2>{title}</h2><p>{subtitle}</p></div>
    </div>
  );
}

function Notice({ kind, title, children }) {
  return <div className={kind}><strong>{title}</strong><p>{children}</p></div>;
}
