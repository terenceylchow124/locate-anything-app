import { IMAGE_ATTRIBUTIONS, MODEL_LICENSES } from "../licenses";

export function LicensePanel() {
  return (
    <footer className="license-panel">
      <p>
        Portfolio/research demo only -- not for commercial use. Model and engine licenses:
      </p>
      <ul>
        {MODEL_LICENSES.map((m) => (
          <li key={m.name}>
            {m.name} -- {m.license}
          </li>
        ))}
      </ul>
      <p>Sample scene images (Wikimedia Commons):</p>
      <ul>
        {IMAGE_ATTRIBUTIONS.map((img) => (
          <li key={img.url}>
            {img.scene}:{" "}
            <a href={img.url} target="_blank" rel="noreferrer">
              {img.title}
            </a>{" "}
            -- {img.author ?? "uncredited uploader"}, {img.license}
          </li>
        ))}
      </ul>
    </footer>
  );
}
