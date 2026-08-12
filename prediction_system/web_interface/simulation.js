// In-browser port of the /api/network_simulation endpoint.
//
// Mirrors run_network_simulation() in prediction_system/inference_server.py step for step:
// same pressure field, same incubation rule, same feature vector, same thresholds. The
// FastAPI server stays the reference implementation for local research work; this exists so
// the public site can run without a backend. Keep the two in sync when either changes.

const EARTH_RADIUS_FT = 3959 * 5280;

// Seasonal averages for Austin (C, mm/month, %, m/s), used when NASA POWER is unreachable.
const DEFAULT_WEATHER = { temp: 25.0, precip: 50.0, humidity: 65.0, wind: 3.0 };

// Root graft transmission rarely reaches beyond ~150 ft.
const MAX_TRANSMISSION_FT = 150;
const LOCAL_DENSITY_RADIUS_FT = 100;
const INCUBATION_MONTHS = 3;
const INFECTION_THRESHOLD = 0.50;

/** Great-circle distance in feet. */
function haversineDist(lat1, lon1, lat2, lon2) {
    const toRad = Math.PI / 180;
    const phi1 = lat1 * toRad;
    const phi2 = lat2 * toRad;
    const dphi = (lat2 - lat1) * toRad;
    const dlambda = (lon2 - lon1) * toRad;
    const a = Math.sin(dphi / 2) ** 2 + Math.cos(phi1) * Math.cos(phi2) * Math.sin(dlambda / 2) ** 2;
    return EARTH_RADIUS_FT * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

// Dates are handled in UTC so that adding fixed 30-day steps matches Python's naive
// datetime arithmetic exactly, with no daylight-saving shifts.
function parseDateUTC(dateStr) {
    return new Date(`${dateStr}T00:00:00Z`);
}

function formatCompact(date) {
    const y = date.getUTCFullYear();
    const m = String(date.getUTCMonth() + 1).padStart(2, '0');
    const d = String(date.getUTCDate()).padStart(2, '0');
    return { ymd: `${y}${m}${d}`, yearMonth: `${y}-${m}` };
}

/**
 * Fetch 30-day averaged weather from NASA POWER.
 * Called straight from the browser: power.larc.nasa.gov sends Access-Control-Allow-Origin: *.
 * Returns null on any failure so callers fall back to defaults.
 */
async function fetchRealNasaWeather(lat, lon, startDate) {
    try {
        // NASA POWER has a ~5-day reporting lag, so the 30-day window ends 5 days before sim start.
        const simStart = parseDateUTC(startDate);
        const endDt = new Date(simStart.getTime() - 5 * 864e5);
        const startDt = new Date(endDt.getTime() - 30 * 864e5);

        const params = new URLSearchParams({
            parameters: 'T2M,PRECTOTCORR,RH2M,WS2M',
            community: 'AG',
            longitude: String(lon),
            latitude: String(lat),
            start: formatCompact(startDt).ymd,
            end: formatCompact(endDt).ymd,
            format: 'JSON'
        });

        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 10000);
        let data;
        try {
            const res = await fetch(
                `https://power.larc.nasa.gov/api/temporal/daily/point?${params}`,
                { signal: controller.signal }
            );
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            data = await res.json();
        } finally {
            clearTimeout(timer);
        }

        const props = (data.properties && data.properties.parameter) || {};
        const avg = (obj) => {
            // NASA POWER uses -999 to mark missing/invalid daily values.
            const vals = Object.values(obj || {}).filter(v => v !== -999);
            return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
        };

        return {
            temp: avg(props.T2M),
            precip: avg(props.PRECTOTCORR),
            humidity: avg(props.RH2M),
            wind: avg(props.WS2M)
        };
    } catch (e) {
        console.warn('NASA API fetch failed:', e);
        return null;
    }
}

/**
 * Run a month-by-month simulation of infection spread.
 * Takes and returns the same shapes as POST /api/network_simulation.
 */
async function runNetworkSimulation(model, request) {
    const {
        trees,
        start_date: startDate,
        months = 24,
        custom_temp = null,
        custom_precip = null,
        custom_humidity = null,
        custom_wind_speed = null
    } = request;

    const currentDate = parseDateUTC(startDate);

    // User-supplied overrides take precedence; missing fields are filled from NASA POWER.
    const overrides = {
        temp: custom_temp,
        precip: custom_precip,
        humidity: custom_humidity,
        wind: custom_wind_speed
    };

    if (Object.values(overrides).some(v => v === null) && trees.length > 0) {
        const cLat = trees.reduce((s, t) => s + t.lat, 0) / trees.length;
        const cLon = trees.reduce((s, t) => s + t.lon, 0) / trees.length;
        const nasa = await fetchRealNasaWeather(cLat, cLon, startDate);

        if (nasa) {
            // Falsy readings fall through to the defaults below, matching the server.
            if (overrides.temp === null && nasa.temp) overrides.temp = round1(nasa.temp);
            // NASA returns mm/day; scale up to a monthly estimate for the model.
            if (overrides.precip === null && nasa.precip) overrides.precip = round1(nasa.precip * 30);
            if (overrides.humidity === null && nasa.humidity) overrides.humidity = round1(nasa.humidity);
            if (overrides.wind === null && nasa.wind) overrides.wind = round1(nasa.wind);
        }
    }

    const cTemp = overrides.temp !== null ? overrides.temp : DEFAULT_WEATHER.temp;
    const cPrecip = overrides.precip !== null ? overrides.precip : DEFAULT_WEATHER.precip;
    const cHumidity = overrides.humidity !== null ? overrides.humidity : DEFAULT_WEATHER.humidity;
    const cWind = overrides.wind !== null ? overrides.wind : DEFAULT_WEATHER.wind;

    const forest = trees.map((t, i) => ({
        id: i,
        lat: t.lat,
        lon: t.lon,
        status: t.type,
        infectionMonth: t.type === 'infected' ? 0 : -1
    }));

    const timeline = [];

    for (let month = 1; month <= months; month++) {
        const stepDate = new Date(currentDate.getTime() + 30 * month * 864e5);
        const monthNum = stepDate.getUTCMonth() + 1;
        const mSin = Math.sin(2 * Math.PI * monthNum / 12);
        const mCos = Math.cos(2 * Math.PI * monthNum / 12);

        // A newly infected tree needs 3 months of incubation before it can transmit.
        const infectious = forest.filter(t =>
            t.status === 'infected' &&
            (t.infectionMonth === 0 || (month - t.infectionMonth) >= INCUBATION_MONTHS)
        );

        if (infectious.length === 0) {
            // Nothing can transmit yet; stop only if the outbreak is genuinely empty.
            if (!forest.some(t => t.status === 'infected')) break;
            continue;
        }

        const healthy = forest.filter(t => t.status === 'healthy');
        const newlyInfected = [];

        // Pressure field: for each healthy tree, aggregate inverse-square contributions from
        // every currently infectious tree. The 1000 / d^2 scaling matches train_model.py so
        // feature magnitudes line up with what the GBM was trained on.
        for (const h of healthy) {
            let minDist = Infinity;
            let pressure = 0;
            let nearbyCount = 0;

            for (const i of infectious) {
                const d = haversineDist(h.lat, h.lon, i.lat, i.lon);
                // Clamp to 1 ft so colocated trees do not blow up the inverse-square term.
                const dSafe = Math.max(d, 1.0);
                pressure += 1000.0 / (dSafe * dSafe);

                if (d < minDist) minDist = d;
                if (d < LOCAL_DENSITY_RADIUS_FT) nearbyCount++;
            }

            if (minDist > MAX_TRANSMISSION_FT) continue;

            // Order must match feature_names in the exported model.
            const prob = model.predictProba([
                Math.log1p(pressure),
                Math.log1p(minDist),
                nearbyCount,
                mSin,
                mCos,
                cTemp,
                cPrecip / 30.0,
                cHumidity,
                cWind
            ]);

            // 0.50 gives roughly balanced precision/recall on the training distribution.
            if (prob > INFECTION_THRESHOLD) newlyInfected.push(h);
        }

        if (newlyInfected.length > 0) {
            for (const t of newlyInfected) {
                t.status = 'infected';
                t.infectionMonth = month;
            }
            timeline.push({
                month,
                date: formatCompact(stepDate).yearMonth,
                new_cases: newlyInfected.map(t => t.id)
            });
        }
    }

    return {
        timeline,
        total_months: months,
        environment: { temp: cTemp, precip: cPrecip, humidity: cHumidity, wind: cWind }
    };
}

function round1(x) {
    return Math.round(x * 10) / 10;
}

window.WiltcastSim = { runNetworkSimulation, haversineDist, fetchRealNasaWeather };
