// Runtime configuration. This is the local-development copy, used when the frontend is
// served alongside the FastAPI backend (npm run dev).
//
// scripts/build.mjs overwrites this file in dist/ with mode: 'static' for the GitHub Pages
// deployment, where there is no backend and the model runs in the browser.
window.WILTCAST_CONFIG = {
    // 'server'  -> call the FastAPI backend; all modes including Historical Validation
    // 'static'  -> score the exported GBM in-browser; network mode only
    mode: 'server',

    // Absolute so `npm run frontend` (static server on :8080) still reaches the backend
    // on :8000. Same-origin setups can set this to '' instead.
    apiBaseUrl: 'http://localhost:8000',

    modelUrl: 'model/gbm_pressure.json'
};
