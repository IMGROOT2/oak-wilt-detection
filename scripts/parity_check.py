"""Diff the browser simulation against the FastAPI one on randomly generated forests.

The static site reimplements /api/network_simulation in JavaScript. This asserts the two
agree exactly: same trees infected, in the same months. Weather is always passed explicitly
so neither side calls NASA POWER and the comparison stays deterministic.

    python scripts/parity_check.py
"""

import asyncio
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'prediction_system'))

from inference_server import run_network_simulation, SimulationRequest  # noqa: E402

N_SCENARIOS = 40
# Austin, in the area the model was trained on.
CENTER_LAT, CENTER_LON = 30.2672, -97.7431


def make_scenario(rng):
    """A random forest of infected and healthy trees at realistic oak-wilt spacing."""
    n_infected = rng.randint(1, 5)
    n_healthy = rng.randint(5, 40)

    # ~0.0005 deg is roughly 180 ft, so trees land in and around transmission range.
    spread = rng.uniform(0.0002, 0.0012)
    lat0 = CENTER_LAT + rng.uniform(-0.05, 0.05)
    lon0 = CENTER_LON + rng.uniform(-0.05, 0.05)

    trees = []
    for _ in range(n_infected):
        trees.append({
            "lat": lat0 + rng.uniform(-spread, spread),
            "lon": lon0 + rng.uniform(-spread, spread),
            "type": "infected"
        })
    for _ in range(n_healthy):
        trees.append({
            "lat": lat0 + rng.uniform(-spread * 3, spread * 3),
            "lon": lon0 + rng.uniform(-spread * 3, spread * 3),
            "type": "healthy"
        })

    return {
        "trees": trees,
        "start_date": f"{rng.randint(2020, 2026)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        "months": rng.choice([6, 12, 24, 36]),
        # Explicit weather keeps both engines off the network.
        "custom_temp": round(rng.uniform(2, 38), 1),
        "custom_precip": round(rng.uniform(0, 200), 1),
        "custom_humidity": round(rng.uniform(20, 95), 1),
        "custom_wind_speed": round(rng.uniform(0, 12), 1),
    }


def normalize(result):
    """Reduce a result to the parts that must match: who got infected, and when."""
    return [
        {"month": ev["month"], "date": ev["date"], "new_cases": sorted(ev["new_cases"])}
        for ev in result["timeline"]
    ]


async def main():
    rng = random.Random(1234)
    scenarios = [make_scenario(rng) for _ in range(N_SCENARIOS)]

    py_results = []
    for s in scenarios:
        py_results.append(await run_network_simulation(SimulationRequest(**s)))

    proc = subprocess.run(
        ['node', str(ROOT / 'scripts' / 'parity_check.mjs')],
        input=json.dumps(scenarios),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit("Node parity harness failed")

    js_results = json.loads(proc.stdout)

    mismatches = 0
    total_infections = 0
    for i, (s, py, js) in enumerate(zip(scenarios, py_results, js_results)):
        py_norm, js_norm = normalize(py), normalize(js)
        total_infections += sum(len(ev["new_cases"]) for ev in py_norm)

        if py_norm != js_norm:
            mismatches += 1
            print(f"\nMISMATCH in scenario {i} "
                  f"({len(s['trees'])} trees, {s['months']} months, start {s['start_date']})")
            print(f"  python: {py_norm}")
            print(f"  js:     {js_norm}")

    print(f"\n{N_SCENARIOS} scenarios, {total_infections} simulated infections total.")
    if mismatches:
        raise SystemExit(f"{mismatches} scenario(s) diverged between Python and JavaScript")
    print("Python and JavaScript simulations agree exactly.")


if __name__ == '__main__':
    asyncio.run(main())
