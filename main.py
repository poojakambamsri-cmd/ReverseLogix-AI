import json
import sqlite3
import numpy as np
import cv2
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI(title="ReverseLogix AI - Return Optimization Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def init_db():
    conn = sqlite3.connect("returns_analytics.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inspection_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            sku TEXT,
            recommendation TEXT,
            lifecycle_action TEXT,
            net_recovery REAL,
            loss_amount REAL,
            shipping_charged REAL DEFAULT 0.0,
            restock_fee_recovered REAL DEFAULT 0.0,
            labor_cost REAL DEFAULT 0.0,
            color_fade_pct REAL,
            crack_severity_pct REAL,
            distortion_pct REAL,
            is_repeat_return INTEGER DEFAULT 0,
            ewaste_saved_kg REAL DEFAULT 0.0
        )
    """)
    conn.commit()

    cursor.execute("PRAGMA table_info(inspection_logs)")
    existing_cols = {col[1] for col in cursor.fetchall()}

    required_cols = {
        "shipping_charged": "REAL DEFAULT 0.0",
        "restock_fee_recovered": "REAL DEFAULT 0.0",
        "labor_cost": "REAL DEFAULT 0.0",
        "is_repeat_return": "INTEGER DEFAULT 0",
        "ewaste_saved_kg": "REAL DEFAULT 0.0",
        "lifecycle_action": "TEXT DEFAULT ''",
        "color_fade_pct": "REAL DEFAULT 0.0",
        "crack_severity_pct": "REAL DEFAULT 0.0",
        "distortion_pct": "REAL DEFAULT 0.0"
    }

    for col_name, col_type in required_cols.items():
        if col_name not in existing_cols:
            cursor.execute(f"ALTER TABLE inspection_logs ADD COLUMN {col_name} {col_type}")
            conn.commit()

    conn.close()

init_db()

def check_repeat_return(sku: str) -> bool:
    conn = sqlite3.connect("returns_analytics.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM inspection_logs WHERE sku = ?", (sku,))
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0

def log_inspection(sku: str, rec: str, lifecycle: str, recovery: float, loss: float, shipping: float, restock_fee: float, labor: float, fade: float, cracks: float, distortion: float, is_repeat: bool):
    conn = sqlite3.connect("returns_analytics.db")
    cursor = conn.cursor()
    ewaste = 1.2 if lifecycle not in ["Scrap / Recycle"] else 0.0
    cursor.execute("""
        INSERT INTO inspection_logs 
        (timestamp, sku, recommendation, lifecycle_action, net_recovery, loss_amount, shipping_charged, restock_fee_recovered, labor_cost, color_fade_pct, crack_severity_pct, distortion_pct, is_repeat_return, ewaste_saved_kg)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        sku, rec, lifecycle, recovery, loss, shipping, restock_fee, labor, fade, cracks, distortion, 1 if is_repeat else 0, ewaste
    ))
    conn.commit()
    conn.close()

def align_images_robust(img_ref: np.ndarray, img_tgt: np.ndarray):
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_tgt = cv2.cvtColor(img_tgt, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(1000)
    kp1, des1 = sift.detectAndCompute(gray_ref, None)
    kp2, des2 = sift.detectAndCompute(gray_tgt, None)

    if des1 is not None and des2 is not None and len(des1) >= 4 and len(des2) >= 4:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        if len(good_matches) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            if H is not None:
                return cv2.warpPerspective(img_tgt, H, (img_ref.shape[1], img_ref.shape[0]))

    return cv2.resize(img_tgt, (img_ref.shape[1], img_ref.shape[0]))

def evaluate_defects(ref_bytes: bytes, tgt_bytes: bytes):
    ref_arr = np.frombuffer(ref_bytes, np.uint8)
    tgt_arr = np.frombuffer(tgt_bytes, np.uint8)

    img_ref = cv2.imdecode(ref_arr, cv2.IMREAD_COLOR)
    img_tgt = cv2.imdecode(tgt_arr, cv2.IMREAD_COLOR)

    if img_ref is None or img_tgt is None:
        raise ValueError("Invalid image input")

    img_tgt_aligned = align_images_robust(img_ref, img_tgt)

    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_tgt = cv2.cvtColor(img_tgt_aligned, cv2.COLOR_BGR2GRAY)

    # Delta-E Color Fading
    lab_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2LAB).astype("float32")
    lab_tgt = cv2.cvtColor(img_tgt_aligned, cv2.COLOR_BGR2LAB).astype("float32")
    delta_e = np.sqrt(np.sum((lab_ref - lab_tgt) ** 2, axis=2))
    color_fade_pct = round(min(100.0, (float(np.mean(delta_e)) / 50.0) * 100.0), 2)

    # Edge & Crack Analysis
    edges_ref = cv2.Canny(gray_ref, 50, 150)
    edges_tgt = cv2.Canny(gray_tgt, 50, 150)
    edge_diff = cv2.absdiff(edges_ref, edges_tgt)
    crack_severity_pct = round(min(100.0, (np.count_nonzero(edge_diff) / float(edge_diff.size)) * 350.0), 2)

    # Contour Distortion
    diff = cv2.absdiff(gray_ref, gray_tgt)
    _, thresh = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = gray_ref.shape[0] * gray_ref.shape[1]
    defect_area = sum(cv2.contourArea(c) for c in contours)
    distortion_pct = round(min(100.0, (defect_area / float(total_area)) * 100.0), 2)

    has_detached_component = any(cv2.contourArea(c) > (total_area * 0.035) for c in contours)

    # SSIM Score
    score_ssim, _ = ssim(gray_ref, gray_tgt, full=True)
    ssim_pct = round(max(0.0, float(score_ssim) * 100.0), 2)

    # Lifecycle Routing
    if distortion_pct > 35.0 or ssim_pct < 40.0:
        lifecycle_action = "Scrap / Recycle"
        action = "Total Destroyed / Unsalvageable"
        reason = "Severe structural destruction. Unrepairable; dispose or send to scrap."
    elif distortion_pct > 20.0 or (crack_severity_pct > 25.0 and ssim_pct < 65.0):
        lifecycle_action = "Harvest for Parts"
        action = "Disassemble & Salvage"
        reason = "Chassis ruined, but internal modules are reusable. Harvest components."
    elif has_detached_component or crack_severity_pct > 15.0:
        lifecycle_action = "Hardware Repair"
        action = "Fix & Replace Components"
        reason = "Physical fractures or detached components. Bench technician repair required."
    elif distortion_pct > 6.0:
        lifecycle_action = "Surface Repair"
        action = "Align & Reshape"
        reason = "Slight shape distortion. Can be reshaped and restored."
    elif color_fade_pct > 10.0:
        lifecycle_action = "Refurbish & Touchup"
        action = "Cosmetic Refurbish"
        reason = "Color fading or surface scuffing found. Apply minor touchup."
    else:
        lifecycle_action = "Direct Reuse"
        action = "Direct Restock"
        reason = "Item is in pristine condition. Ready for immediate direct reselling."

    return {
        "action": action,
        "lifecycle_action": lifecycle_action,
        "reason": reason,
        "color_fade_pct": color_fade_pct,
        "crack_severity_pct": crack_severity_pct,
        "distortion_pct": distortion_pct,
        "ssim_pct": ssim_pct,
        "has_detached_component": has_detached_component
    }

@app.post("/api/evaluate")
async def evaluate_return(
    ref_image: UploadFile = File(...),
    tgt_image: UploadFile = File(...),
    economics_json: str = Form(...)
):
    try:
        econ = json.loads(economics_json)
        sku = econ.get("sku", "SKU-PROD-01")
        ref_bytes = await ref_image.read()
        tgt_bytes = await tgt_image.read()

        analysis = evaluate_defects(ref_bytes, tgt_bytes)

        # Dynamic Charges & Variables
        msrp = float(econ.get("msrp", 500.0))
        base_freight = float(econ.get("freight", 20.0))
        refurb_materials = float(econ.get("refurb_cost", 25.0))
        restock_fee_pct = float(econ.get("restock_fee_pct", 10.0))
        labor_rate_hr = float(econ.get("labor_rate", 45.0))
        harvest_value = float(econ.get("harvest_value", 75.0))
        ewaste_fee = float(econ.get("ewaste_fee", 12.0))
        secondary_discount_pct = float(econ.get("secondary_discount", 20.0))

        is_repeat = check_repeat_return(sku)
        shipping_charged = (base_freight * 2.0) if is_repeat else base_freight

        lifecycle = analysis["lifecycle_action"]
        restock_fee_recovered = 0.0
        labor_cost = 0.0

        # Detailed Financial Calculation Engine
        if lifecycle == "Direct Reuse":
            restock_fee_recovered = msrp * (restock_fee_pct / 100.0)
            net_recovery = msrp + restock_fee_recovered - shipping_charged
        
        elif lifecycle == "Refurbish & Touchup":
            labor_cost = labor_rate_hr * 0.5  # 30 mins bench work
            resale_value = msrp * (1.0 - (secondary_discount_pct / 100.0))
            net_recovery = resale_value - refurb_materials - labor_cost - shipping_charged
        
        elif lifecycle == "Surface Repair":
            labor_cost = labor_rate_hr * 1.0  # 1 hour work
            resale_value = msrp * (1.0 - ((secondary_discount_pct + 10.0) / 100.0))
            net_recovery = resale_value - (refurb_materials * 1.5) - labor_cost - shipping_charged
        
        elif lifecycle == "Hardware Repair":
            labor_cost = labor_rate_hr * 1.75  # 1.75 hours work
            resale_value = msrp * (1.0 - ((secondary_discount_pct + 15.0) / 100.0))
            net_recovery = resale_value - 50.0 - labor_cost - shipping_charged
        
        elif lifecycle == "Harvest for Parts":
            labor_cost = labor_rate_hr * 0.75  # 45 mins teardown
            net_recovery = harvest_value - labor_cost - shipping_charged
        
        else:  # Scrap / Recycle
            net_recovery = - ewaste_fee - shipping_charged

        loss_amount = max(0.0, msrp - net_recovery)

        log_inspection(
            sku=sku,
            rec=analysis["action"],
            lifecycle=lifecycle,
            recovery=net_recovery,
            loss=loss_amount,
            shipping=shipping_charged,
            restock_fee=restock_fee_recovered,
            labor=labor_cost,
            fade=analysis["color_fade_pct"],
            cracks=analysis["crack_severity_pct"],
            distortion=analysis["distortion_pct"],
            is_repeat=is_repeat
        )

        return {
            "analysis": analysis,
            "sku": sku,
            "is_repeat_return": is_repeat,
            "shipping_charged": round(shipping_charged, 2),
            "restock_fee_recovered": round(restock_fee_recovered, 2),
            "labor_cost": round(labor_cost, 2),
            "recommendation": analysis["action"],
            "lifecycle_action": lifecycle,
            "net_recovery": round(net_recovery, 2),
            "loss_amount": round(loss_amount, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics")
def get_analytics():
    conn = sqlite3.connect("returns_analytics.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, sku, recommendation, net_recovery, loss_amount, shipping_charged, is_repeat_return, ewaste_saved_kg, lifecycle_action FROM inspection_logs ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()

    total_inspections = len(rows)
    total_net_recovery = sum(r[4] for r in rows) if rows else 0.0
    total_shipping = sum(r[6] for r in rows) if rows else 0.0
    total_ewaste = sum(r[8] for r in rows) if rows else 0.0

    time_series = []
    running_pl = 0.0
    for r in rows:
        running_pl += r[4]
        time_series.append({
            "id": r[0],
            "timestamp": r[1],
            "sku": r[2],
            "recommendation": r[3],
            "lifecycle": r[9],
            "recovery": r[4],
            "shipping": r[6],
            "is_repeat": bool(r[7]),
            "running_total": round(running_pl, 2)
        })

    return {
        "summary": {
            "total_inspections": total_inspections,
            "total_net_recovery": round(total_net_recovery, 2),
            "total_shipping_charged": round(total_shipping, 2),
            "ewaste_saved_kg": round(total_ewaste, 2)
        },
        "time_series": time_series
    }

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4000)