# GitHub Pages

## Why the README appeared

The source homepage is `src/web/overview.html`. Local Python routes map `/` to
that file and provide `/api/report`. Publishing the repository root through
the default Pages/Jekyll flow does not run those routes. Without a root-level
`index.html`, it can publish the README instead.

Renaming a file alone is insufficient: root-relative asset paths would miss the
repository subpath, and `/api/report` would not exist on a static host.

## Deploy

1. Commit and push `.github/workflows/pages.yml`, `scripts/build_pages.py`, the
   updated homepage JavaScript, and `src/web/sandbox-static.html` with its styles.
2. Open the repository's Settings, then Pages.
3. Under Build and deployment, change Source to **GitHub Actions**, not
   **Deploy from a branch**. No custom domain is required.
4. In Actions, open **Deploy research site to Pages**. If necessary, select
   **Run workflow** on `main` after changing the Pages setting.
5. Wait for a successful deploy step and follow its returned Pages URL.

The expected default URL is:

```text
https://mikawu73.github.io/Dota2-BP-Predection/
```

Official setup reference:
https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site

## Build Contract

`python -m scripts.build_pages` uses only Python's standard library. It builds
`dist/pages` from the shared overview, a local-sandbox notice page, display
scripts/styles, the Lucide bundle/license and the public report. The homepage
becomes `index.html`; links become relative; `data-report-url` changes the data
source from the local API to `./results.json`. A `.nojekyll` marker is included.

The builder refuses directories containing unexpected files, rather than
uploading their contents or deleting them. Use a dedicated output directory.
No raw matches, checkpoints, credentials, team catalog, Python source, or
interactive sandbox JavaScript are included. GitHub Actions uploads only the
generated directory, never the repository root.

The public sandbox link leads to `sandbox.html`, which clearly states that
inference is local and requires a checkpoint. It does not simulate predictions
or send visitors' drafts to a nonexistent backend. The existing local
`python -m scripts.serve_demo` and `/sandbox` route are unchanged.

## Preview and Updates

```bash
python -m scripts.build_pages
python -m http.server 8766 --bind 127.0.0.1 --directory dist
```

Open `http://127.0.0.1:8766/pages/` to verify the site under a subpath.
Stop the preview with Ctrl+C. The generated files are ignored by Git; Actions
rebuilds them from source on the runner. No training dependencies are installed
by the Pages workflow.

The report is a recorded snapshot. Updating weights or raw match files does not
update public figures automatically. Regenerate the aggregate report with
`python -m scripts.export_portfolio_report`, review it, then commit/push it.

If the README still appears, check Source, the latest workflow's commit and
deployment status before trying a browser hard refresh. If configuration fails,
verify that Pages is enabled and that you have permission to configure it.
Never fix missing assets by uploading raw data or an entire working directory.
