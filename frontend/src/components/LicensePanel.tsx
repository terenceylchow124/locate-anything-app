import { IMAGE_ATTRIBUTIONS, MODEL_LICENSES } from "../licenses";

export function LicensePanel() {
  return (
    <footer className="mx-auto mt-8 max-w-3xl border-t border-border px-4 py-6 text-xs text-text sm:px-6">
      <p className="mb-1 mt-2 first:mt-0">
        Portfolio/research demo only -- not for commercial use. Model and engine licenses:
      </p>
      <ul className="list-inside list-disc space-y-0.5 pl-1">
        {MODEL_LICENSES.map((m) => (
          <li key={m.name}>
            {m.name} -- {m.license}
          </li>
        ))}
      </ul>
      <p className="mb-1 mt-2 first:mt-0">Sample scene images (Wikimedia Commons):</p>
      <ul className="list-inside list-disc space-y-0.5 pl-1">
        {IMAGE_ATTRIBUTIONS.map((img) => (
          <li key={img.url}>
            {img.scene}:{" "}
            <a
              href={img.url}
              target="_blank"
              rel="noreferrer"
              className="text-inherit underline decoration-border underline-offset-2 transition-colors hover:text-accent hover:decoration-accent"
            >
              {img.title}
            </a>{" "}
            -- {img.author ?? "uncredited uploader"}, {img.license}
          </li>
        ))}
      </ul>
    </footer>
  );
}
