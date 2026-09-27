export default function ParagraphInput({ paragraph, onChange, onSubmit, loading, error }) {
  return (
    <form className="panel workflow-panel" onSubmit={onSubmit} aria-busy={loading}>
      <div className="panel__heading">
        <div>
          <h2>Décrivez votre antenne</h2>
          <p>Gemini extrait les paramètres. Vous pourrez les vérifier et les compléter avant la prédiction.</p>
        </div>
      </div>
      <label className="field" htmlFor="antenna-paragraph">
        <span className="field__label">Votre paragraphe</span>
        <textarea
          id="antenna-paragraph"
          value={paragraph}
          onChange={(event) => onChange(event.target.value)}
          minLength={10}
          maxLength={6000}
          rows={7}
          required
          disabled={loading}
          aria-describedby="paragraph-help"
          placeholder="Je souhaite étudier un dipôle demi-onde de la famille des antennes filaires, avec un gap de 39,47, un substrat de largeur 75 et de longueur 75, une permittivité relative de 5 et une fréquence de 2,4 GHz. Pour la courbe, balayer de 0,5 à 4 GHz sur 301 points avec un seuil de -10 dB."
        />
        <span className="field__hint" id="paragraph-help">
          Précisez la famille, la variante et les unités de fréquence. {paragraph.length}/6000 caractères.
        </span>
      </label>
      <p className="paragraph-disclosure">En cliquant sur « Extraire les paramètres », ce texte sera envoyé à Google Gemini.</p>
      {error && <div className="error" role="alert">{error}</div>}
      <button className="primary-button" type="submit" disabled={loading || paragraph.trim().length < 10}>
        {loading ? "Extraction en cours…" : "Extraire les paramètres"}
      </button>
    </form>
  );
}
