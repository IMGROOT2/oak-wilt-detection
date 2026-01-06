// Global State
let map;
let layers = new L.LayerGroup();
let markers = []; // {lat, lon, type, id, markerObj}
let mode = 'network';
let activeTool = 'healthy';
let scenarioData = null; // For historical

// Animation State
let timelineEvents = []; // [{month: 1, new_cases: [id, id]}]
let currentAnimFrame = 0;
let animInterval = null;
let totalMonths = 12;

document.addEventListener('DOMContentLoaded', () => {
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
    document.querySelectorAll('.content').forEach(el => el.classList.add('hidden'));
    document.getElementById(`step-${step}`).classList.remove('hidden');
    
    document.querySelectorAll('.step').forEach((el, i) => {
        if (i + 1 === step) el.classList.add('active');
        else el.classList.remove('active');
    });

    if (step === 2) setupConfig();
}

function selectMode(m) {
    mode = m;
    document.querySelectorAll('.card').forEach(el => el.classList.remove('selected'));
    document.getElementById(`mode-${m}`).classList.add('selected');
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
    document.querySelectorAll('.tool').forEach(el => el.classList.remove('active'));
    document.getElementById(`tool-${t}`).classList.add('active');
}

function addManualTree(latlng) {
    const color = activeTool === 'healthy' ? '#27ae60' : '#e74c3c';
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
        const res = await fetch('/api/historical_scenario');
        if (!res.ok) throw new Error("API Error");
        const data = await res.json();
        
        scenarioData = data;
        
        // Render Map
        // 1. Past Infections (Known)
        data.past_infection.forEach(p => {
            L.circleMarker([p.lat, p.lon], {
                color: '#e74c3c', fillColor: '#e74c3c', fillOpacity:0.6, radius:5
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
        btn.innerText = "🎲 Load New Scenario";
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
            url = '/api/network_simulation';
            // Use 24 months (2 years) for historical validation to capture more ground truth data
            const months = mode === 'network' ? parseInt(document.getElementById('net-months').value) : 24;
            
            // For historical, we use the scenario date, else today
            const startC = mode === 'historical' ? scenarioData.cutoff_date : new Date().toISOString().split('T')[0];
            
            payload = {
                trees: markers.map(m => ({lat: m.lat, lon: m.lon, type: m.type})),
                start_date: startC,
                months: months
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
    // We use the 90th percentile distance to exclude outliers (satellite infections)
    // and provide a more representative "Frontline" expansion rate.
    let yearlyRate = 0;
    const originInfected = markers.filter(m => m.type === 'infected');
    
    if (originInfected.length > 0) {
        // Centroid of original infection
        const cLat = originInfected.reduce((sum, m) => sum + m.lat, 0) / originInfected.length;
        const cLon = originInfected.reduce((sum, m) => sum + m.lon, 0) / originInfected.length;
        const centroid = L.latLng(cLat, cLon);
        
        // Helper: Get Robust "Frontline" Radius (75th Percentile)
        // We use the 75th percentile (Upper Quartile) to track the main infection front
        // while ignoring stochastic "spark" outliers that inflate the spread rate logic.
        function getEffectiveRadius(idList) {
            if (idList.length === 0) return 0;
            const dists = idList.map(id => {
                const m = markers.find(mk => mk.id === id);
                return m ? centroid.distanceTo(L.latLng(m.lat, m.lon)) : 0;
            }).sort((a, b) => a - b);
            
            // 75th Percentile index (Upper Quartile)
            const k = Math.floor(dists.length * 0.75);
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
            <span>📉 Est. Spread Rate:</span>
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
                <div class="metric-row"><span>✅ Captured Infections:</span> <strong>${truePositives}</strong></div>
                <div class="metric-row" title="Safe areas flagged as risky (Safety Buffer)"><span>🛡️ High Pressure Zones:</span> <strong>${falsePositives}</strong></div>
                <div class="metric-row" title="Actual infected trees we missed (within ${scopeYears} yr window)"><span>⚠️ Unpredicted Infections:</span> <strong>${missed}</strong></div>
                <div class="metric-row" style="color:#7f8c8d; font-size:0.9em"><span>📅 Out of Scope (>${scopeYears} yrs):</span> <strong>${outOfScope}</strong></div>
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
                
                // Yellow for Near Future (In Scope), Grey for Far Future (Out of Scope)
                const color = isRelevant ? '#f1c40f' : '#95a5a6';
                const style = isRelevant ? 'solid' : 'dotted';
                
                L.circleMarker([m.lat, m.lon], {
                   radius: 8, color: color, fill: false, weight: 2, dashArray: isRelevant ? null : '4,4'
                }).addTo(layers).bindPopup(`Infected: ${m.infection_date} (${isRelevant ? 'In Scope' : 'Future'})`);
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
        let isInfected = false;
        for (let t of timelineEvents) {
            if (t.month <= monthIndex && t.new_cases.includes(m.id)) {
                isInfected = true;
                break;
            }
        }
        
        const el = m.marker.getElement();
        
        if (isInfected) {
            m.marker.setStyle({ color: '#e74c3c', fillColor: '#e74c3c', radius: 8, fillOpacity: 0.9 });
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
