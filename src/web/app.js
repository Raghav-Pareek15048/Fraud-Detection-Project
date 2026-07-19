// SentryFraud Client Dashboard Logic

const API_BASE_URL = "http://localhost:8000";

// Preset transactions data
const PRESET_LEGIT = {
    Time: 1044.0,
    Amount: 18.99,
    V1: 1.191857, V2: 0.266151, V3: 0.166480, V4: 0.448154, V5: 0.060018,
    V6: -0.082361, V7: -0.078803, V8: 0.085102, V9: -0.255425, V10: -0.166974,
    V11: 1.612, V12: 1.065, V13: -0.342, V14: 0.432, V15: 0.954,
    V16: 0.456, V17: -0.123, V18: -0.054, V19: 0.123, V20: -0.069,
    V21: -0.225775, V22: -0.638672, V23: 0.101288, V24: -0.339846, V25: 0.167170,
    V26: 0.125895, V27: -0.008983, V28: 0.014724
};

const PRESET_FRAUD = {
    Time: 406.0,
    Amount: 0.00,
    V1: -2.312227, V2: 1.951992, V3: -1.609851, V4: 3.997908, V5: -0.522188,
    V6: -1.426545, V7: -2.537387, V8: 1.391657, V9: -2.770089, V10: -2.772272,
    V11: 3.202033, V12: -2.899907, V13: -0.595222, V14: -4.289254, V15: 0.389724,
    V16: -1.140747, V17: -2.830022, V18: -0.016822, V19: 0.416956, V20: 0.126911,
    V21: 0.517232, V22: -0.035049, V23: -0.465211, V24: 0.320198, V25: 0.045133,
    V26: 0.177611, V27: 0.261145, V28: -0.143276
};

// Main DOM Elements
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const accordionHeader = document.getElementById("accordionHeader");
const accordionBody = document.getElementById("accordionBody");
const accordion = accordionHeader.parentElement;
const vFeaturesGrid = document.querySelector(".v-features-grid");
const vFeatureTemplate = document.getElementById("vFeatureTemplate");

const presetLegitBtn = document.getElementById("presetLegitBtn");
const presetFraudBtn = document.getElementById("presetFraudBtn");
const predictionForm = document.getElementById("predictionForm");
const inputTime = document.getElementById("inputTime");
const inputAmount = document.getElementById("inputAmount");

const predictionResultCard = document.getElementById("predictionResultCard");
const resultIconContainer = document.getElementById("resultIconContainer");
const resultLabel = document.getElementById("resultLabel");
const resultProbText = document.getElementById("resultProbText");
const resultConfidenceBar = document.getElementById("resultConfidenceBar");
const resultConfidencePercent = document.getElementById("resultConfidencePercent");

const csvDropzone = document.getElementById("csvDropzone");
const csvFileInput = document.getElementById("csvFileInput");
const batchLoader = document.getElementById("batchLoader");
const batchResultsWrapper = document.getElementById("batchResultsWrapper");
const resultsTableBody = document.getElementById("resultsTableBody");
const clearBatchBtn = document.getElementById("clearBatchBtn");

let metricsChart = null;

// Initialize Web App
document.addEventListener("DOMContentLoaded", () => {
    generateVInputFields();
    checkApiStatus();
    initMetricsChart();
    
    // Periodically poll API status
    setInterval(checkApiStatus, 15000);
});

// Generate 28 Input Fields for PCA features
function generateVInputFields() {
    vFeaturesGrid.innerHTML = ""; // Clear template holder
    for (let i = 1; i <= 28; i++) {
        const clone = vFeatureTemplate.content.cloneNode(true);
        const inputGroup = clone.querySelector(".v-input-group");
        const label = clone.querySelector(".v-label");
        const input = clone.querySelector(".v-input");
        
        label.textContent = `V${i}`;
        label.setAttribute("for", `inputV${i}`);
        input.setAttribute("id", `inputV${i}`);
        input.setAttribute("name", `V${i}`);
        
        vFeaturesGrid.appendChild(clone);
    }
}

// Check Backend API Connection Status
async function checkApiStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (response.ok && data.status === "healthy") {
            statusDot.className = "status-dot connected";
            statusText.textContent = "Connected to Backend API";
        } else {
            statusDot.className = "status-dot disconnected";
            statusText.textContent = "Model Loading...";
        }
    } catch (error) {
        statusDot.className = "status-dot disconnected";
        statusText.textContent = "Backend Offline";
    }
}

// Draw Model Performance Comparison Chart
function initMetricsChart(metricsData = null) {
    const ctx = document.getElementById("metricsChart").getContext("2d");
    
    // Default static performance summary from notebook evaluation
    const defaultData = {
        labels: ["XGBoost (Best)", "Random Forest", "Logistic Regression"],
        recall: [84.04, 85.11, 92.55],
        precision: [78.22, 45.98, 5.57],
        f1Score: [81.03, 59.70, 10.51],
        prAuc: [85.73, 83.65, 75.56]
    };
    
    const chartData = metricsData || defaultData;
    
    if (metricsChart) {
        metricsChart.destroy();
    }
    
    metricsChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: chartData.labels,
            datasets: [
                {
                    label: "Precision (%)",
                    data: chartData.precision,
                    backgroundColor: "rgba(99, 102, 241, 0.75)",
                    borderColor: "rgba(99, 102, 241, 1)",
                    borderWidth: 1,
                    borderRadius: 4
                },
                {
                    label: "Recall (%)",
                    data: chartData.recall,
                    backgroundColor: "rgba(16, 185, 129, 0.75)",
                    borderColor: "rgba(16, 185, 129, 1)",
                    borderWidth: 1,
                    borderRadius: 4
                },
                {
                    label: "F1 Score (%)",
                    data: chartData.f1Score,
                    backgroundColor: "rgba(245, 158, 11, 0.75)",
                    borderColor: "rgba(245, 158, 11, 1)",
                    borderWidth: 1,
                    borderRadius: 4
                },
                {
                    label: "PR-AUC (%)",
                    data: chartData.prAuc,
                    backgroundColor: "rgba(139, 92, 246, 0.75)",
                    borderColor: "rgba(139, 92, 246, 1)",
                    borderWidth: 1,
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "top",
                    labels: {
                        color: "#9CA3AF",
                        font: { family: "Outfit", size: 11 }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.raw}%`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: "#9CA3AF", font: { family: "Outfit", size: 12 } }
                },
                y: {
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#9CA3AF", font: { family: "Outfit", size: 11 } },
                    max: 100
                }
            }
        }
    });
}

// Toggle PCA Accordion Section
accordionHeader.addEventListener("click", () => {
    accordion.classList.toggle("active");
    if (accordion.classList.contains("active")) {
        accordionBody.style.maxHeight = accordionBody.scrollHeight + "px";
    } else {
        accordionBody.style.maxHeight = 0;
    }
});

// Load Presets into Form
presetLegitBtn.addEventListener("click", () => populateForm(PRESET_LEGIT));
presetFraudBtn.addEventListener("click", () => populateForm(PRESET_FRAUD));

function populateForm(data) {
    inputTime.value = data.Time;
    inputAmount.value = data.Amount;
    for (let i = 1; i <= 28; i++) {
        const el = document.getElementById(`inputV${i}`);
        if (el) el.value = data[`V${i}`];
    }
    
    // Automatically open accordion to show loaded features
    if (!accordion.classList.contains("active")) {
        accordion.classList.add("active");
        accordionBody.style.maxHeight = accordionBody.scrollHeight + "px";
    }
}

// Predict Form Submit
predictionForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    
    // Build payload
    const payload = {
        Time: parseFloat(inputTime.value),
        Amount: parseFloat(inputAmount.value)
    };
    for (let i = 1; i <= 28; i++) {
        payload[`V${i}`] = parseFloat(document.getElementById(`inputV${i}`).value || 0.0);
    }
    
    // Disable Submit Button and show loading
    const submitBtn = document.getElementById("submitBtn");
    const origHtml = submitBtn.innerHTML;
    submitBtn.disabled = true;
    submitBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Running Analysis...`;
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            throw new Error(`Server returned status ${response.status}`);
        }
        
        const data = await response.json();
        renderPredictionResult(data);
    } catch (error) {
        alert(`Prediction request failed: ${error.message}. Is the backend API running?`);
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = origHtml;
    }
});

// Render Real-Time Prediction Card Visuals
function renderPredictionResult(result) {
    predictionResultCard.style.display = "flex";
    
    // Remove previous classes
    predictionResultCard.className = "prediction-result-card";
    
    if (result.is_fraud) {
        predictionResultCard.classList.add("fraudulent");
        resultIconContainer.innerHTML = `<i class="fa-solid fa-shield-virus"></i>`;
        resultLabel.textContent = "FRAUDULENT TRANSACTION DETECTED";
        resultLabel.style.color = "var(--danger)";
        resultProbText.textContent = `${(result.probability * 100).toFixed(2)}%`;
    } else {
        predictionResultCard.classList.add("legitimate");
        resultIconContainer.innerHTML = `<i class="fa-solid fa-shield-heart"></i>`;
        resultLabel.textContent = "TRANSACTION CONFIRMED LEGITIMATE";
        resultLabel.style.color = "var(--success)";
        resultProbText.textContent = `${(result.probability * 100).toFixed(2)}%`;
    }
    
    // Animate confidence bar
    const confidenceVal = result.confidence * 100;
    resultConfidencePercent.textContent = `Evaluation Confidence: ${confidenceVal.toFixed(2)}%`;
    
    // Need a tiny delay for CSS transitions to trigger
    setTimeout(() => {
        resultConfidenceBar.style.width = `${confidenceVal}%`;
    }, 50);
}

// --- Drag & Drop Batch Upload Logic ---
csvDropzone.addEventListener("click", () => csvFileInput.click());

csvDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    csvDropzone.classList.add("dragover");
});

csvDropzone.addEventListener("dragleave", () => {
    csvDropzone.classList.remove("dragover");
});

csvDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    csvDropzone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) handleBatchFile(file);
});

csvFileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) handleBatchFile(file);
});

// Send File to Backend CSV Batch Endpoint
async function handleBatchFile(file) {
    if (!file.name.endsWith(".csv")) {
        alert("Please upload a valid CSV file.");
        return;
    }
    
    // Show Loading, Hide Table
    csvDropzone.style.display = "none";
    batchLoader.style.display = "flex";
    batchResultsWrapper.style.display = "none";
    
    const formData = new FormData();
    formData.append("file", file);
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict/batch/csv`, {
            method: "POST",
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Server returned status ${response.status}`);
        }
        
        const data = await response.json();
        renderBatchTable(data);
    } catch (error) {
        alert(`Failed to process batch file: ${error.message}`);
        resetBatchUI();
    } finally {
        batchLoader.style.display = "none";
    }
}

// Render dynamic table rows
function renderBatchTable(records) {
    resultsTableBody.innerHTML = "";
    
    records.slice(0, 100).forEach((rec, idx) => {
        const tr = document.createElement("tr");
        
        const badgeClass = rec.is_fraud ? "badge fraud" : "badge legit";
        const badgeText = rec.is_fraud ? "Fraud" : "Legit";
        
        tr.innerHTML = `
            <td>${idx + 1}</td>
            <td>${parseFloat(rec.Time || 0).toFixed(1)}</td>
            <td>$${parseFloat(rec.Amount || 0).toFixed(2)}</td>
            <td><span class="${badgeClass}">${badgeText}</span></td>
            <td>${(rec.confidence * 100).toFixed(1)}%</td>
        `;
        resultsTableBody.appendChild(tr);
    });
    
    // Show results
    batchResultsWrapper.style.display = "block";
}

// Reset dropzone UI
function resetBatchUI() {
    csvFileInput.value = "";
    csvDropzone.style.display = "block";
    batchResultsWrapper.style.display = "none";
}

clearBatchBtn.addEventListener("click", resetBatchUI);
