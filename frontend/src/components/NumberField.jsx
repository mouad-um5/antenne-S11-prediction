export default function NumberField({ label, name, value, onChange, hint, ...inputProps }) {
  return (
    <label className="field" htmlFor={name}>
      <span className="field__label">{label}</span>
      <input
        id={name}
        name={name}
        type="number"
        value={value}
        onChange={onChange}
        {...inputProps}
      />
      {hint && <span className="field__hint">{hint}</span>}
    </label>
  );
}
