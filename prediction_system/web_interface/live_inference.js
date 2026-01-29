// Global State
let map;
let layers = new L.LayerGroup();
let markers = []; // {lat, lon, type, id, markerObj}
let mode = 'network';
let activeTool = 'healthy';
let scenarioData = null; // For historical

// Config: Use localhost if on file://, otherwise relative path
const API_BASE_URL = window.location.protocol === 'file:' ? 'http://localhost:8000' : '';

// Animation State
let timelineEvents = []; // [{month: 1, new_cases: [id, id]}]

// --- Status Check ---
async function checkServerStatus() {
    const dot = document.getElementById('status-dot');
    const txt = document.getElementById('status-text');
    if (!dot || !txt) return;

    try {
        const res = await fetch(`${API_BASE_URL}/health`);
        if (res.ok) {
            dot.style.background = '#2ecc71'; // Green
            txt.innerText = "Online";
        } else {
            throw new Error();
        }
    } catch (e) {
        dot.style.background = '#e74c3c'; // Red
        txt.innerText = "Offline";
    }
}
setInterval(checkServerStatus, 5000); // Check every five s

let currentAnimFrame = 0;
let animInterval = null;
let totalMonths = 12;

document.addEventListener('DOMContentLoaded', () => {
    checkServerStatus(); // Initial check
    initMap();
    selectMode('network');
});

// --- Map ---
function initMap() {
    map = L.map('map', { maxZoom: 19 }).setView([30.2672, -97.7431], 13);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
        attribution: '© OpenStreetMap, © CARTO',
        maxZoom: 20
    }).addTo(map);
    layers.addTo(map);
    
    map.on('click', (e) => {
        if (document.getElementById('step-2').classList.contains('hidden')) return;
        if (mode === 'network') addManualTree(e.latlng);
    });
}

function clearMap() {
    layers.clearLayers();
    markers = [];
    updateCounts();
}

// --- Navigation ---
function nextStep(step) {
    document.querySelectorAll('.content').forEach(el => el.classList.add('hidden')); // Keep this, I'll add .content class to sections
    // Or just target IDs which is safer
    document.getElementById('step-1').classList.add('hidden');
    document.getElementById('step-2').classList.add('hidden');
    document.getElementById('step-3').classList.add('hidden');
    document.getElementById(`step-${step}`).classList.remove('hidden');
    
    // Reset all steps
    [1, 2, 3].forEach(i => {
        const el = document.getElementById(`step-ind-${i}`);
        el.classList.remove('border-slate-800', 'text-slate-900');
        el.classList.add('border-transparent', 'text-slate-400');
    });
    
    // Activate current
    const active = document.getElementById(`step-ind-${step}`);
    active.classList.remove('border-transparent', 'text-slate-400');
    active.classList.add('border-slate-800', 'text-slate-900');

    if (step === 2) setupConfig();
}

function selectMode(m) {
    mode = m;
    // Reset styles
    ['network', 'historical'].forEach(t => {
        const el = document.getElementById(`mode-${t}`);
        el.classList.remove('ring-1', 'ring-slate-800', 'bg-slate-50');
    });
    
    // Apply Active
    document.getElementById(`mode-${m}`).classList.add('ring-1', 'ring-slate-800', 'bg-slate-50');
}

function setupConfig() {
    document.querySelectorAll('.cfg-panel').forEach(el => el.classList.add('hidden'));
    document.getElementById(`cfg-${mode}`).classList.remove('hidden');
    
    if (mode === 'historical' && !scenarioData) {
        // Auto load first time
        loadScenario();
    }
}

// --- Tooling ---
function setTool(t) {
    activeTool = t;
    ['healthy', 'infected'].forEach(tool => {
        const el = document.getElementById(`tool-${tool}`);
        // Reset to inactive: bg-white text-slate-600 border-slate-200
        el.classList.remove('bg-slate-800', 'text-white', 'border-slate-800');
        el.classList.add('bg-white', 'text-slate-600', 'border-slate-200');
    });
    
    // Active: bg-slate-800 text-white border-slate-800
    const active = document.getElementById(`tool-${t}`);
    active.classList.remove('bg-white', 'text-slate-600', 'border-slate-200');
    active.classList.add('bg-slate-800', 'text-white', 'border-slate-800');
}

function addManualTree(latlng) {
    const color = activeTool === 'healthy' ? '#27ae60' : '#4B0082'; // Dark Purple
    const m = L.circleMarker(latlng, {
        color: color, fillColor: color, fillOpacity: 0.8, radius: 6
    }).addTo(layers);
    
    markers.push({
        id: markers.length,
        lat: latlng.lat,
        lon: latlng.lng,
        type: activeTool,
        marker: m
    });
    updateCounts();
}

function updateCounts() {
    const h = markers.filter(t => t.type === 'healthy').length;
    const i = markers.filter(t => t.type === 'infected').length;
    document.getElementById('count-h').innerText = h;
    document.getElementById('count-i').innerText = i;
}


// --- Historical Mode ---
async function loadScenario() {
    const btn = document.getElementById('btn-load-scenario');
    btn.innerText = "Loading...";
    btn.disabled = true;
    
    try {
        clearMap();
        // Try to pick an eligible cluster using precomputed simulated spread rates (simple: array of numeric IDs)
        // Eligible: 20 <= spread_ft_per_yr <= 200
        let scenario = null;
        try {
            // If API_BASE_URL is set (file://), fetch CSV via server; otherwise use relative path
            const csvUrl = API_BASE_URL ? `${API_BASE_URL}/data/simulated_spread_rates.csv` : 'data/simulated_spread_rates.csv';
            const csvRes = await fetch(csvUrl);
            if (csvRes.ok) {
                const text = await csvRes.text();
                const lines = text.split(/\r?\n/).filter(l => l.trim() !== '');
                // Expect header then rows: cluster_id,spread_ft_per_yr
                const header = lines.shift().split(',').map(h => h.trim());
                const idIdx = header.indexOf('cluster_id');
                const valIdx = header.indexOf('spread_ft_per_yr');
                if (idIdx !== -1 && valIdx !== -1) {
                    const eligibleIds = [];
                    const allIds = [];
                    for (const l of lines) {
                        const cols = l.split(',').map(c => c.trim());
                        const id = Number(cols[idIdx]);
                        const val = parseFloat(cols[valIdx]);
                        if (!Number.isNaN(id)) allIds.push(id);
                        if (!Number.isNaN(id) && !Number.isNaN(val) && val >= 20 && val <= 200) {
                            eligibleIds.push(id);
                        }
                    }

                    const excludedIds = allIds.filter(i => !eligibleIds.includes(i));
                    console.log('simulated_spread_rates: eligible count=', eligibleIds.length, 'excluded count=', excludedIds.length);
                    console.log('Eligible cluster IDs:', eligibleIds);
                    console.log('Excluded cluster IDs:', excludedIds);

                    if (eligibleIds.length > 0) {
                        const pick = eligibleIds[Math.floor(Math.random() * eligibleIds.length)];
                        console.log('Picked cluster ID from CSV:', pick);
                        const res = await fetch(`${API_BASE_URL}/api/historical_scenario?cluster_id=${encodeURIComponent(pick)}`);
                        if (res.ok) scenario = await res.json();
                    } else {
                        // Fallback: use cluster features if simulated CSV is missing or empty
                        try {
                            console.log('simulated_spread_rates empty — falling back to cluster features');
                            const featuresUrl = API_BASE_URL ? `${API_BASE_URL}/data/oak_wilt_cluster_features.csv` : 'data/oak_wilt_cluster_features.csv';
                            const fRes = await fetch(featuresUrl);
                            if (fRes.ok) {
                                const fText = await fRes.text();
                                const fLines = fText.split(/\r?\n/).filter(l => l.trim() !== '');
                                const fHeader = fLines.shift().split(',').map(h => h.trim());
                                const fidIdx = fHeader.indexOf('cluster_id');
                                const frateIdx = fHeader.indexOf('spread_rate_km_per_year');
                                // Fallback parsing is brittle; prefer server endpoint if available
                                try {
                                    if (API_BASE_URL) {
                                        const eligRes = await fetch(`${API_BASE_URL}/api/eligible_clusters`);
                                        if (eligRes.ok) {
                                            const elig = await eligRes.json();
                                            console.log('Eligible clusters (server):', elig.eligible);
                                            console.log('Excluded clusters (server):', elig.excluded);
                                            if (elig.eligible && elig.eligible.length > 0) {
                                                const pick2 = elig.eligible[Math.floor(Math.random() * elig.eligible.length)];
                                                console.log('Picked cluster ID from eligible_clusters endpoint:', pick2);
                                                const res2 = await fetch(`${API_BASE_URL}/api/historical_scenario?cluster_id=${encodeURIComponent(pick2)}`);
                                                if (res2.ok) scenario = await res2.json();
                                            }
                                        }
                                    } else {
                                        // legacy CSV-based fallback (best effort)
                                        const fallbackIds = [];
                                        if (fidIdx !== -1 && frateIdx !== -1) {
                                            for (const l2 of fLines) {
                                                const cols2 = l2.split(',').map(c => c.trim());
                                                const id2 = Number(cols2[fidIdx]);
                                                const rateKm = parseFloat(cols2[frateIdx]);
                                                if (!Number.isNaN(id2) && !Number.isNaN(rateKm)) {
                                                    const rateFtPerYr = rateKm * 3280.84; // km/yr -> ft/yr
                                                    if (rateFtPerYr >= 20 && rateFtPerYr <= 200) fallbackIds.push(id2);
                                                }
                                            }
                                        }
                                        console.log('Fallback eligible count=', fallbackIds.length, 'IDs=', fallbackIds);
                                        if (fallbackIds.length > 0) {
                                            const pick2 = fallbackIds[Math.floor(Math.random() * fallbackIds.length)];
                                            console.log('Picked cluster ID from features fallback:', pick2);
                                            const res2 = await fetch(`${API_BASE_URL}/api/historical_scenario?cluster_id=${encodeURIComponent(pick2)}`);
                                            if (res2.ok) scenario = await res2.json();
                                        }
                                    }
                                } catch (fe) {
                                    console.warn('Features fallback failed', fe);
                                }
                            }
                        } catch (fe) {
                            console.warn('Features fallback failed', fe);
                        }
                    }
                }
            }
        } catch (e) {
            console.warn('Failed to load simulated_spread_rates.csv or pick eligible cluster — falling back', e);
        }

        // If CSV-based selection didn't yield a valid scenario, fall back to server random scenario

        // Fallback: request a random historical scenario from the API
        if (!scenario) {
            const res = await fetch(`${API_BASE_URL}/api/historical_scenario`);
            if (!res.ok) throw new Error("API Error");
            scenario = await res.json();
        }

        const data = scenario;
        scenarioData = data;
        
        // Render Map
        // 1. Past Infections (Known)
        data.past_infection.forEach(p => {
            // Dark purple for existing infections
            L.circleMarker([p.lat, p.lon], {
                color: '#4B0082', fillColor: '#4B0082', fillOpacity:0.6, radius:5
            }).addTo(layers).bindPopup("Existing Infection");
            // Treat as infected input
            markers.push({id: markers.length, lat: p.lat, lon: p.lon, type: 'infected', marker: null}); 
        });

        // 2. Candidates (Healthy Input + Future Hidden)
        data.candidates.forEach(c => {
             const m = L.circleMarker([c.lat, c.lon], {
                color: '#bdc3c7', fillColor: '#bdc3c7', fillOpacity:0.5, radius:4 
            }).addTo(layers);
            
            markers.push({
                id: markers.length,
                lat: c.lat,
                lon: c.lon,
                type: 'healthy', // Reset to healthy for simulation
                real_future: c.is_future_infection, // Hidden truth
                infection_date: c.infection_date, // <--- CRITICAL FIX: Pass date through
                marker: m
            });
        });

        map.setView([data.center.lat, data.center.lon], 15);
        
        // UI
        document.getElementById('hist-meta').classList.remove('hidden');
        document.getElementById('meta-id').innerText = data.cluster_id;
        document.getElementById('meta-date').innerText = data.cutoff_date;
        document.getElementById('meta-counts').innerText = data.candidates.length + " Trees";

    } catch (e) {
        alert("Failed to load: " + e.message);
    } finally {
        btn.innerText = "Load New Scenario";
        btn.disabled = false;
    }
}


// --- Execution ---
async function runAnalysis() {
    const btn = document.getElementById('btn-run');
    btn.innerText = "Simulating...";
    btn.disabled = true;

    try {
        let payload = {};
        let url = '';

        if (mode === 'network' || mode === 'historical') {
            url = `${API_BASE_URL}/api/network_simulation`;
            // Use 24 months (2 years) for historical validation to capture more ground truth data
            const months = mode === 'network' ? parseInt(document.getElementById('net-months').value) : 24;
            
            // For historical, we use the scenario date, else today
            const startC = mode === 'historical' ? scenarioData.cutoff_date : new Date().toISOString().split('T')[0];
            
            // Check for overrides (Network mode only)
            let customTemp = null;
            let customPrecip = null;
            let customHumidity = null;
            let customWind = null;

            if (mode === 'network') {
                const tempVal = document.getElementById('custom-temp').value;
                const precipVal = document.getElementById('custom-precip').value;
                const humidVal = document.getElementById('custom-humidity').value;
                const windVal = document.getElementById('custom-wind').value;
                
                if (tempVal !== "") customTemp = parseFloat(tempVal);
                if (precipVal !== "") customPrecip = parseFloat(precipVal);
                if (humidVal !== "") customHumidity = parseFloat(humidVal);
                if (windVal !== "") customWind = parseFloat(windVal);
            }

            payload = {
                trees: markers.map(m => ({lat: m.lat, lon: m.lon, type: m.type})),
                start_date: startC,
                months: months,
                custom_temp: customTemp,
                custom_precip: customPrecip,
                custom_humidity: customHumidity,
                custom_wind_speed: customWind
            };
        }

        const res = await fetch(url, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("Simulation Failed");
        const data = await res.json();
        
        processResults(data);
        
        // Populate overrides if they were empty
        if (data.environment && mode === 'network') {
            const tempInput = document.getElementById('custom-temp');
            if (tempInput.value === "") tempInput.value = data.environment.temp || "";
            
            const precipInput = document.getElementById('custom-precip');
            if (precipInput.value === "") precipInput.value = data.environment.precip || "";

            const humidInput = document.getElementById('custom-humidity');
            if (humidInput.value === "") humidInput.value = data.environment.humidity || "";

            const windInput = document.getElementById('custom-wind');
            if (windInput.value === "") windInput.value = data.environment.wind || "";
        }

        nextStep(3);

    } catch(e) {
        alert(e.message);
    } finally {
        btn.innerText = "Run Analysis";
        btn.disabled = false;
    }
}

function processResults(data) {
    // data.timeline = [{month, new_cases: [ids]}]
    timelineEvents = data.timeline || [];
    currentAnimFrame = 0;
    totalMonths = data.total_months;
    
    // UI Update
    const totalInfected = timelineEvents.reduce((acc, ev) => acc + ev.new_cases.length, 0);
    
    // --- Calc Expansion Rate (Robust) ---
    // and provide a more representative "Frontline" expansion rate.
    let yearlyRate = 0;
    const originInfected = markers.filter(m => m.type === 'infected');
    
    if (originInfected.length > 0) {
        // Centroid of original infection
        const cLat = originInfected.reduce((sum, m) => sum + m.lat, 0) / originInfected.length;
        const cLon = originInfected.reduce((sum, m) => sum + m.lon, 0) / originInfected.length;
        const centroid = L.latLng(cLat, cLon);
        
        // Helper: Get Robust "Frontline" Radius (seven five) Percentile)
        // We use the seventyfifth percentile (Upper Quartile) to track the main infection front
        // while ignoring stochastic "spark" outliers that inflate the spread rate logic.
        function getEffectiveRadius(idList) {
            if (idList.length === 0) return 0;
            const dists = idList.map(id => {
                const m = markers.find(mk => mk.id === id);
                return m ? centroid.distanceTo(L.latLng(m.lat, m.lon)) : 0;
            }).sort((a, b) => a - b);
            
            // seventyfifth Percentile index (Upper Quartile)
            console.log("dists length:", dists.length);
            const k = Math.floor(dists.length * 0.9);
            return dists[k];
        }

        // Initial Set IDs
        const initialIds = originInfected.map(m => m.id);
        const r0 = getEffectiveRadius(initialIds);
        
        // Final Set IDs (Initial + All New)
        const finalIds = [...initialIds];
        timelineEvents.forEach(ev => ev.new_cases.forEach(id => finalIds.push(id)));
        const r1 = getEffectiveRadius(finalIds);
        
        // Rate Calculation
        // If simulation is short, we project to yearly
        const deltaMonths = Math.max(1, totalMonths); // Avoid div/0
        const growthFt = (r1 - r0) * 3.28084; // meters to feet
        
        if (growthFt > 0) {
            yearlyRate = (growthFt / deltaMonths) * 12;
        } else {
            yearlyRate = 0;
        }
    }

    let html = `
        <div class="metric-row">
            <span>New Infections:</span>
            <span class="badge" style="background:${totalInfected>0?'#ffebee':'#e8f5e9'}; color:${totalInfected>0?'#c62828':'#2e7d32'}">
                ${totalInfected} Trees
            </span>
        </div>
        <div class="metric-row" title="Projects the radial growth of the infection center to a full year">
            <span>Est. Spread Rate:</span>
            <span class="metric-val">${Math.round(yearlyRate)} ft/yr</span>
        </div>
    `;

    if (mode === 'historical') {
        // Validation Logic
        // Calculate accuracy against 'real_future' markers
        let truePositives = 0;
        let falsePositives = 0;
        let missed = 0;
        let outOfScope = 0;

        // Get all IDs predicted to be infected
        const allPredictedIds = new Set();
        timelineEvents.forEach(e => e.new_cases.forEach(id => allPredictedIds.add(id)));
        
        // Simulation Time Window
        // We only "fail" the model if it misses a tree that got infected WITHIN the simulation period (12 months).
        // If the tree got infected 2 years later, the model was correct to say "safe for now".
        const simStartDate = new Date(scenarioData.cutoff_date);
        const simEndDate = new Date(simStartDate);
        simEndDate.setMonth(simEndDate.getMonth() + totalMonths);

        markers.forEach(m => {
            if (m.real_future) {
                // Check if this future infection actually falls within our 1-year window
                const infDate = new Date(m.infection_date);
                const isRelevant = infDate <= simEndDate;

                if (isRelevant) {
                    if (allPredictedIds.has(m.id)) truePositives++;
                    else missed++;
                } else {
                    outOfScope++;
                }
            } else {
                if (allPredictedIds.has(m.id) && m.type === 'healthy') falsePositives++;
            }
        });

        // Dynamic Label for Years
        const scopeYears = (totalMonths / 12).toFixed(1).replace('.0', '');

        html += `
            <div style="margin-top:10px; padding:10px; background:#f8f9fa; border-radius:6px;">
                <div class="metric-row"><span>Captured Infections:</span> <strong>${truePositives}</strong></div>
                <div class="metric-row" title="Safe areas flagged as risky (Safety Buffer)"><span>High Pressure Zones:</span> <strong>${falsePositives}</strong></div>
                <div class="metric-row" title="Actual infected trees we missed (within ${scopeYears} yr window)"><span>Unpredicted Infections:</span> <strong>${missed}</strong></div>
                <div class="metric-row" style="color:#7f8c8d; font-size:0.9em"><span>Out of Scope (>${scopeYears} yrs):</span> <strong>${outOfScope}</strong></div>
            </div>
            <p style="font-size:0.8rem; color:#666; margin-top:5px;">
                * Comparison of "Gravity Model" prediction vs. Reality.
            </p>
        `;
        
        // Show Ground Truth 
        markers.forEach(m => {
            if (m.real_future) {
                const infDate = new Date(m.infection_date);
                const isRelevant = infDate <= simEndDate;
                
                // USER REQUEST 3: Just don't show out of scope cases at all
                // Only render if relevant (in scope)
                
                if (isRelevant) {
                    const color = '#f1c40f'; // Yellow for future "truth" to compare
                    const style = 'solid';
                    
                    // USER REQUEST 4: Don't show yellow circle for original infections...
                    // "markers" loop includes healthy candidates which have real_future=true.
                    // Original infections were pushed as type='infected' and usually don't have real_future set?
                    // Let's check init logic. 
                    // Past infections: type='infected', marker:null.
                    // Candidates: type='healthy', real_future=true/false.
                    // So this loop only affects candidates.
                    
                    L.circleMarker([m.lat, m.lon], {
                       radius: 8, color: color, fill: false, weight: 2
                    }).addTo(layers).bindPopup(`Infected: ${m.infection_date} (In Scope)`);
                }
            }
        });
    }

    document.getElementById('results-container').innerHTML = html;
    
    // Setup Animation
    document.getElementById('media-controls').classList.remove('hidden');
    const slider = document.getElementById('anim-slider');
    slider.max = totalMonths;
    slider.value = 0;
    slider.oninput = (e) => showFrame(parseInt(e.target.value));
    
    // Start animation automatically
    if (!animInterval) togglePlay();
}

// --- Animation ---
function showFrame(monthIndex) {
    document.getElementById('anim-label').innerText = monthIndex === 0 ? "Start" : `Month ${monthIndex}`;
    
    // Reset all predicted to base state
    markers.forEach(m => {
        if (m.type === 'infected' && m.marker) return; // Originally infected stays red
        if (!m.marker) return;

        // Check if this marker was infected by this month
        let infectionMonth = -1;
        for (let t of timelineEvents) {
            if (t.new_cases.includes(m.id)) {
                infectionMonth = t.month;
                break;
            }
        }
        
        const el = m.marker.getElement();
        const wasInfected = infectionMonth !== -1 && infectionMonth <= monthIndex;
        
        if (wasInfected) {
            // USER REQUEST 3: 
            // - If newly infected (THIS month), turn Orange.
            // - If infected previously (BEFORE this month), turn Red.
            
            const isNew = infectionMonth === monthIndex;
            const color = isNew ? '#FF8C00' : '#DC143C'; // Orange vs Crimson
            
            m.marker.setStyle({ color: color, fillColor: color, radius: 8, fillOpacity: 0.9 });
            if(el) el.classList.add('pulse-icon');
        } else {
            m.marker.setStyle({ color: '#bdc3c7', fillColor: '#bdc3c7', radius: 4, fillOpacity: 0.5 });
            if(el) el.classList.remove('pulse-icon');
        }
    });
}

function togglePlay() {
    if (animInterval) {
        clearInterval(animInterval);
        animInterval = null;
        document.getElementById('btn-play').innerText = "▶ Play";
    } else {
        document.getElementById('btn-play').innerText = "⏸ Pause";
        animInterval = setInterval(() => {
            let val = parseInt(document.getElementById('anim-slider').value);
            if (val >= totalMonths) val = -1; // Loop
            val++;
            document.getElementById('anim-slider').value = val;
            showFrame(val);
        }, 500); // 500ms per month
    }
}

function resetAnim() {
    if (animInterval) togglePlay(); // Stop
    document.getElementById('anim-slider').value = 0;
    showFrame(0);
}
