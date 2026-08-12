// Verify the browser simulation reproduces the FastAPI one.
//
// Reads scenarios from stdin as JSON, runs each through simulation.js, prints results to
// stdout. scripts/parity_check.py drives this and diffs against inference_server.py.

import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const SRC = join(ROOT, 'prediction_system', 'web_interface');

// simulation.js and gbm.js are plain browser scripts that publish onto window; give them
// one and evaluate them here rather than maintaining a parallel module build.
globalThis.window = globalThis;

const evalBrowserScript = (name) => {
    const code = readFileSync(join(SRC, name), 'utf8');
    new Function(code)();
};

evalBrowserScript('gbm.js');
evalBrowserScript('simulation.js');

const spec = JSON.parse(readFileSync(join(SRC, 'model', 'gbm_pressure.json'), 'utf8'));
const model = new globalThis.GBMModel(spec);

const scenarios = JSON.parse(readFileSync(0, 'utf8'));
const results = [];
for (const scenario of scenarios) {
    results.push(await globalThis.WiltcastSim.runNetworkSimulation(model, scenario));
}

console.log(JSON.stringify(results));
