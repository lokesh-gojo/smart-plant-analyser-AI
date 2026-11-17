-- SmartVine AI: Production Database Schema & Spatial Vineyard Mapping
-- PostgreSQL / Supabase Schema for Precision Viticulture Analytics
-- Aligned with Seng et al. (IEEE Access, 2018) & GrapeCS-ML Database

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Vineyard Blocks & Cultivar Metadata
CREATE TABLE IF NOT EXISTS vineyard_blocks (
    block_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vineyard_name TEXT NOT NULL,
    cultivar_name TEXT NOT NULL, -- e.g. 'Shiraz', 'Sauvignon Blanc', 'Merlot', 'Cabernet Sauvignon'
    cultivar_color TEXT CHECK (cultivar_color IN ('Red', 'White', 'Black')) NOT NULL,
    row_count INT NOT NULL DEFAULT 10,
    vines_per_row INT NOT NULL DEFAULT 50,
    row_spacing_m REAL DEFAULT 2.5,
    vine_spacing_m REAL DEFAULT 1.5,
    planted_year INT DEFAULT 2018,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. In-Field Vision Captures & Disease Screening Table
CREATE TABLE IF NOT EXISTS vine_inspections (
    inspection_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    block_id UUID REFERENCES vineyard_blocks(block_id) ON DELETE CASCADE,
    row_number INT NOT NULL,
    vine_number INT NOT NULL,
    gps_lat DOUBLE PRECISION,
    gps_lon DOUBLE PRECISION,
    image_url TEXT NOT NULL,
    calibration_px_per_mm REAL DEFAULT 2.5,
    diagnosed_disease TEXT DEFAULT 'Healthy_Vine',
    disease_confidence_pct REAL,
    vinesens_spray_alert BOOLEAN DEFAULT FALSE,
    captured_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Segmented Grape Clusters & Ripeness Metrics Table (GrapeCS-ML Schema)
CREATE TABLE IF NOT EXISTS cluster_metrics (
    cluster_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    inspection_id UUID REFERENCES vine_inspections(inspection_id) ON DELETE CASCADE,
    cluster_index INT NOT NULL DEFAULT 1,
    visible_berry_count INT NOT NULL,
    estimated_total_berries INT NOT NULL,
    compactness_solidity REAL NOT NULL,
    botrytis_rot_risk TEXT CHECK (botrytis_rot_risk IN ('Low', 'Moderate', 'High')),
    cielab_l REAL,
    cielab_a REAL,
    cielab_b REAL,
    dominant_hue_deg REAL,
    predicted_brix_tss REAL NOT NULL,
    predicted_ph REAL NOT NULL,
    estimated_bunch_weight_g REAL,
    phenological_stage TEXT,
    harvest_ready BOOLEAN DEFAULT FALSE,
    peduncle_cutting_x INT,
    peduncle_cutting_y INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Longitudinal Phenology & Yield Aggregation View
CREATE OR REPLACE VIEW vineyard_yield_summary AS
SELECT 
    b.block_id,
    b.vineyard_name,
    b.cultivar_name,
    b.cultivar_color,
    COUNT(DISTINCT i.inspection_id) AS total_inspections,
    COUNT(c.cluster_id) AS total_clusters_detected,
    ROUND(AVG(c.predicted_brix_tss)::numeric, 2) AS avg_brix_tss,
    ROUND(AVG(c.predicted_ph)::numeric, 2) AS avg_ph,
    ROUND(AVG(c.compactness_solidity)::numeric, 3) AS avg_compactness,
    SUM(CASE WHEN c.botrytis_rot_risk = 'High' THEN 1 ELSE 0 END) AS high_rot_risk_clusters,
    SUM(CASE WHEN c.harvest_ready THEN 1 ELSE 0 END) AS harvest_ready_clusters,
    ROUND(
        (COUNT(c.cluster_id)::numeric / NULLIF(COUNT(DISTINCT i.inspection_id), 0) * 
         COALESCE(AVG(c.estimated_bunch_weight_g), 150.0) * b.vines_per_row * b.row_count / 1000000.0 * 
         (10000.0 / (b.row_spacing_m * b.vine_spacing_m * b.vines_per_row * b.row_count)))::numeric,
        2
    ) AS estimated_tonnes_per_hectare
FROM vineyard_blocks b
LEFT JOIN vine_inspections i ON b.block_id = i.block_id
LEFT JOIN cluster_metrics c ON i.inspection_id = c.inspection_id
GROUP BY b.block_id, b.vineyard_name, b.cultivar_name, b.cultivar_color, b.vines_per_row, b.row_count, b.row_spacing_m, b.vine_spacing_m;

-- Indexes for fast spatial and temporal queries
CREATE INDEX IF NOT EXISTS idx_vine_inspections_block ON vine_inspections(block_id);
CREATE INDEX IF NOT EXISTS idx_vine_inspections_coords ON vine_inspections(gps_lat, gps_lon);
CREATE INDEX IF NOT EXISTS idx_cluster_metrics_inspection ON cluster_metrics(inspection_id);
CREATE INDEX IF NOT EXISTS idx_cluster_metrics_harvest ON cluster_metrics(harvest_ready);
