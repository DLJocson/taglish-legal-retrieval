# Frontend

UI for PH Legal AI Search, served by FastAPI from the repo root.

## Layout

```
frontend/
├── index.html          →  GET /
├── analytics.html      →  GET /analytics
└── assets/
    ├── css/app.css     →  GET /assets/css/app.css
    └── js/
        ├── core.js     →  API client, toasts, shared helpers
        ├── search.js   →  Search, compare, filters, modal
        └── analytics.js
```

## Development

From the repo root:

```bash
python run_dev.py
```

With `DEV=1`, the server sends no-cache headers for HTML/CSS/JS. **Save a file and refresh the browser** — no server restart needed for frontend changes.

For Python/API changes:

```bash
python run_dev.py --reload-backend
```
