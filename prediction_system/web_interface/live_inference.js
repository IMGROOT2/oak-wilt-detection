// Global state
let map;
let layers = new L.LayerGroup();
let markers = [];
let mode = 'network';
let activeTool = 'healthy';
let scenarioData = null;

// Tile layers
let baseTile = null;
let satelliteTile = null;
let isSatellite = false;

// 200ft radius guide circle for infected trees
let radiusGuide = null;
const RADIUS_200FT = 60.96; // 200 feet in meters

const API_BASE_URL = window.location.protocol === 'file:' ? 'http://localhost:8000' : '';

let timelineEvents = [];

// Server health check
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
setInterval(checkServerStatus, 5000);

let currentAnimFrame = 0;
let animInterval = null;
let totalMonths = 12;

document.addEventListener('DOMContentLoaded', () => {
    checkServerStatus(); // Initial check
    initMap();
    selectMode('network');
});

// Map initialization
function initMap() {
    map = L.map('map', { maxZoom: 19 }).setView([30.2672, -97.7431], 13);
    baseTile = L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
        attribution: '© OpenStreetMap, © CARTO',
        maxZoom: 20
    }).addTo(map);
    satelliteTile = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: '© Esri, Maxar, Earthstar Geographics',
        maxZoom: 20
    });
    layers.addTo(map);
    
    map.on('click', (e) => {
        if (document.getElementById('step-2').classList.contains('hidden')) return;
        if (mode === 'network') addManualTree(e.latlng);
    });
}

// Satellite toggle
function toggleSatellite(enabled) {
    isSatellite = enabled;
    if (enabled) {
        map.removeLayer(baseTile);
        satelliteTile.addTo(map);
    } else {
        map.removeLayer(satelliteTile);
        baseTile.addTo(map);
    }
}

function clearMap() {
    layers.clearLayers();
    markers = [];
    radiusGuide = null;
    updateCounts();
}

// Navigation
function nextStep(step) {
    document.querySelectorAll('.content').forEach(el => el.classList.add('hidden'));
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

// Active tool selection
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
    const color = activeTool === 'healthy' ? '#27ae60' : '#4B0082';
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
    if (activeTool === 'infected') updateRadiusGuide();
}

// Update 200ft radius guide circle around infected tree centroid
function updateRadiusGuide() {
    if (radiusGuide) { layers.removeLayer(radiusGuide); radiusGuide = null; }
    const infected = markers.filter(m => m.type === 'infected');
    if (infected.length === 0) return;
    const cLat = infected.reduce((s, m) => s + m.lat, 0) / infected.length;
    const cLon = infected.reduce((s, m) => s + m.lon, 0) / infected.length;
    radiusGuide = L.circle([cLat, cLon], {
        radius: RADIUS_200FT,
        color: '#1f2937',
        weight: 3,
        dashArray: '12, 8',
        fill: true,
        fillColor: '#4b5563',
        fillOpacity: 0.25,
        interactive: false
    }).addTo(layers);
}

function updateCounts() {
    const h = markers.filter(t => t.type === 'healthy').length;
    const i = markers.filter(t => t.type === 'infected').length;
    document.getElementById('count-h').innerText = h;
    document.getElementById('count-i').innerText = i;
}


// Historical scenario loading
async function loadScenario() {
    const btn = document.getElementById('btn-load-scenario');
    btn.innerText = "Loading...";
    btn.disabled = true;
    
    try {
        clearMap();
        let scenario = null;
        try {
            const csvUrl = API_BASE_URL ? `${API_BASE_URL}/data/simulated_spread_rates.csv` : 'data/simulated_spread_rates.csv';
            const csvRes = await fetch(csvUrl);
            if (csvRes.ok) {
                const text = await csvRes.text();
                const lines = text.split(/\r?\n/).filter(l => l.trim() !== '');
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

                    if (eligibleIds.length > 0) {
                        const pick = eligibleIds[Math.floor(Math.random() * eligibleIds.length)];
                        const res = await fetch(`${API_BASE_URL}/api/historical_scenario?cluster_id=${encodeURIComponent(pick)}`);
                        if (res.ok) scenario = await res.json();
                    } else {
                        try {
                            const featuresUrl = API_BASE_URL ? `${API_BASE_URL}/data/oak_wilt_cluster_features.csv` : 'data/oak_wilt_cluster_features.csv';
                            const fRes = await fetch(featuresUrl);
                            if (fRes.ok) {
                                const fText = await fRes.text();
                                const fLines = fText.split(/\r?\n/).filter(l => l.trim() !== '');
                                const fHeader = fLines.shift().split(',').map(h => h.trim());
                                const fidIdx = fHeader.indexOf('cluster_id');
                                const frateIdx = fHeader.indexOf('spread_rate_km_per_year');
                                try {
                                    if (API_BASE_URL) {
                                        const eligRes = await fetch(`${API_BASE_URL}/api/eligible_clusters`);
                                        if (eligRes.ok) {
                                            const elig = await eligRes.json();
                                            if (elig.eligible && elig.eligible.length > 0) {
                                                const pick2 = elig.eligible[Math.floor(Math.random() * elig.eligible.length)];
                                                const res2 = await fetch(`${API_BASE_URL}/api/historical_scenario?cluster_id=${encodeURIComponent(pick2)}`);
                                                if (res2.ok) scenario = await res2.json();
                                            }
                                        }
                                    } else {
                                        // CSV-based fallback
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
                                        if (fallbackIds.length > 0) {
                                            const pick2 = fallbackIds[Math.floor(Math.random() * fallbackIds.length)];
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

        // fall back to a random scenario from the API if CSV selection failed
        if (!scenario) {
            const res = await fetch(`${API_BASE_URL}/api/historical_scenario`);
            if (!res.ok) throw new Error("API Error");
            scenario = await res.json();
        }

        const data = scenario;
        scenarioData = data;
        
        // render past infections on the map
        data.past_infection.forEach(p => {
            L.circleMarker([p.lat, p.lon], {
                color: '#4B0082', fillColor: '#4B0082', fillOpacity:0.6, radius:5
            }).addTo(layers);
            markers.push({id: markers.length, lat: p.lat, lon: p.lon, type: 'infected', marker: null});
        });

        // candidate trees
        data.candidates.forEach(c => {
             const m = L.circleMarker([c.lat, c.lon], {
                color: '#bdc3c7', fillColor: '#bdc3c7', fillOpacity:0.5, radius:4 
            }).addTo(layers);
            
            markers.push({
                id: markers.length,
                lat: c.lat,
                lon: c.lon,
                type: 'healthy',
                real_future: c.is_future_infection,
                infection_date: c.infection_date,
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


// Run analysis
async function runAnalysis() {
    const btn = document.getElementById('btn-run');
    btn.innerText = "Simulating...";
    btn.disabled = true;

    try {
        let payload = {};
        let url = '';

        if (mode === 'network' || mode === 'historical') {
            url = `${API_BASE_URL}/api/network_simulation`;
            // for historical mode, use 24 months and the scenario date
            const months = mode === 'network' ? parseInt(document.getElementById('net-months').value) : 24;
            const startC = mode === 'historical' ? scenarioData.cutoff_date : new Date().toISOString().split('T')[0];
            
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
        
        // backfill weather fields if they were left blank
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
    
    // spread rate calculation via 90th-percentile radius expansion
    let yearlyRate = 0;
    const originInfected = markers.filter(m => m.type === 'infected');
    
    if (originInfected.length > 0) {
        const cLat = originInfected.reduce((sum, m) => sum + m.lat, 0) / originInfected.length;
        const cLon = originInfected.reduce((sum, m) => sum + m.lon, 0) / originInfected.length;
        const centroid = L.latLng(cLat, cLon);
        
        // 90th percentile radius
        function getEffectiveRadius(idList) {
            if (idList.length === 0) return 0;
            const dists = idList.map(id => {
                const m = markers.find(mk => mk.id === id);
                return m ? centroid.distanceTo(L.latLng(m.lat, m.lon)) : 0;
            }).sort((a, b) => a - b);
            
            const k = Math.floor(dists.length * 0.9);
            return dists[k];
        }

        // Initial Set IDs
        const initialIds = originInfected.map(m => m.id);
        const r0 = getEffectiveRadius(initialIds);
        
        // Final Set IDs (initial + all newly predicted)
        const finalIds = [...initialIds];
        timelineEvents.forEach(ev => ev.new_cases.forEach(id => finalIds.push(id)));
        const r1 = getEffectiveRadius(finalIds);
        
        // annualized rate
        const deltaMonths = Math.max(1, totalMonths);
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
        let truePositives = 0;
        let falsePositives = 0;
        let missed = 0;
        let outOfScope = 0;

        // predicted infection IDs
        const allPredictedIds = new Set();
        timelineEvents.forEach(e => e.new_cases.forEach(id => allPredictedIds.add(id)));
        
        // only count misses for infections within the simulated time window
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

        // Dynamic label for scope
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
        
        // overlay ground truth markers for in-scope future infections
        markers.forEach(m => {
            if (m.real_future) {
                const infDate = new Date(m.infection_date);
                const isRelevant = infDate <= simEndDate;
                if (isRelevant) {
                    const color = '#f1c40f';
                    
                    L.circleMarker([m.lat, m.lon], {
                       radius: 8, color: color, fill: false, weight: 2
                    }).addTo(layers).bindPopup(`Infected: ${m.infection_date} (In Scope)`);
                }
            }
        });
    }

    document.getElementById('results-container').innerHTML = html;
    
    // Setup animation controls
    document.getElementById('media-controls').classList.remove('hidden');
    const slider = document.getElementById('anim-slider');
    slider.max = totalMonths;
    slider.value = 0;
    slider.oninput = (e) => showFrame(parseInt(e.target.value));
    if (!animInterval) togglePlay();
}

// Animation playback
function showFrame(monthIndex) {
    document.getElementById('anim-label').innerText = monthIndex === 0 ? "Start" : `Month ${monthIndex}`;
    
    // Reset all markers to base state, then color infected ones
    markers.forEach(m => {
        if (m.type === 'infected' && m.marker) return;
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
            const isNew = infectionMonth === monthIndex;
            const color = isNew ? '#FF8C00' : '#DC143C';
            
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
            if (val >= totalMonths) val = -1;
            val++;
            document.getElementById('anim-slider').value = val;
            showFrame(val);
        }, 500);
    }
}

function resetAnim() {
    if (animInterval) togglePlay(); // Stop
    document.getElementById('anim-slider').value = 0;
    showFrame(0);
}
