// =====================================================================
// SCRIPT 5 — "Convergence of evidence": multi-source agreement map of
// irrigated / Rabi-cropped land, Gaya district (Bihar). No ground data needed.
//
// Three independent spatial sources:
//   1. Our Sentinel-2 Random Forest map 2023-24 (class 0 = Rabi-cropped)
//   2. ESA WorldCereal 2021 — actively irrigated in the winter-cereal season
//   3. MODIS 2001–2025 — persistent Rabi greenness (Jan 1–Mar 15 max NDVI
//      above GREEN_T in at least 80% of years)
// Agreement score per pixel = number of sources saying "irrigated" (0–3).
//
// Exports (Drive folder Gaya_Irrigation):
//   gaya_agreement_area.csv     area (ha) per agreement score
//   gaya_agreement_samples.csv  5000 random cropland pixels with all 3 sources
//   gaya_agreement_map.tif      score map, 30 m
// =====================================================================

var S2_ASSET = 'projects/gee-class-474916/assets/gaya_irrigation_RF_2023';
var GREEN_T = 0.5;
var EXPORT_FOLDER = 'Gaya_Irrigation';

var gaya = ee.FeatureCollection('FAO/GAUL/2015/level2')
  .filter(ee.Filter.eq('ADM1_NAME', 'Bihar'))
  .filter(ee.Filter.eq('ADM2_NAME', 'Gaya'));
var aoi = gaya.geometry();
Map.centerObject(gaya, 9);
Map.setOptions('SATELLITE');

// ---------- Source 1: our Sentinel-2 map ----------
var src1 = ee.Image(S2_ASSET).eq(0).unmask(0).rename('s2_rabi');

// ---------- Source 2: ESA WorldCereal 2021 ----------
var wc = ee.ImageCollection('ESA/WorldCereal/2021/MODELS/v100').filterBounds(aoi);
function wcLayer(product, season) {
  return wc.filter(ee.Filter.eq('product', product)).filter(ee.Filter.eq('season', season))
    .mosaic().select('classification').eq(100).unmask(0);
}
var src2 = wcLayer('irrigation', 'tc-wintercereals').rename('wc_irr');
var wcCrop = wcLayer('temporarycrops', 'tc-annual');

// ---------- Source 3: MODIS persistent Rabi greenness ----------
var ndviCol = ee.ImageCollection('MODIS/061/MOD13Q1').filterBounds(aoi).map(function(img) {
  return ee.Image(img.select('NDVI').multiply(0.0001)
    .updateMask(img.select('SummaryQA').lte(1)).copyProperties(img, ['system:time_start']));
});
var rabiGreen = ee.ImageCollection(ee.List.sequence(2001, 2025).map(function(y) {
  y = ee.Number(y);
  return ndviCol.filterDate(ee.Date.fromYMD(y, 1, 1), ee.Date.fromYMD(y, 3, 16))
    .max().gt(GREEN_T).rename('g');
}));
var src3 = rabiGreen.mean().gte(0.8).unmask(0).rename('modis_rabi');

// ---------- Agreement on cropland ----------
var worldCover = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').eq(40);
var base = worldCover.or(wcCrop);
var stack = src1.addBands(src2).addBands(src3).updateMask(base).clip(aoi);
var score = src1.add(src2).add(src3).updateMask(base).clip(aoi).rename('score').toByte();

Map.addLayer(score, {min: 0, max: 3, palette: ['f7f7f7', 'fdd49e', 'fc8d59', '0570b0']},
  'Agreement score (0 = none, 3 = all sources irrigated)');
Map.addLayer(src1.selfMask(), {palette: ['1f78b4']}, 'Source 1: Sentinel-2 Rabi-cropped', false);
Map.addLayer(src2.selfMask(), {palette: ['33a02c']}, 'Source 2: WorldCereal irrigated', false);
Map.addLayer(src3.selfMask(), {palette: ['e31a1c']}, 'Source 3: MODIS persistent Rabi green', false);

// ---------- Area per score ----------
var areaStats = ee.Image.pixelArea().divide(1e4).addBands(score).reduceRegion({
  reducer: ee.Reducer.sum().group({groupField: 1, groupName: 'score'}),
  geometry: aoi, scale: 30, crs: 'EPSG:32645', maxPixels: 1e13, tileScale: 8
});
var areaTable = ee.FeatureCollection(ee.List(areaStats.get('groups')).map(function(g) {
  g = ee.Dictionary(g);
  return ee.Feature(null, {score: g.get('score'), area_ha: g.get('sum')});
}));
print('Area (ha) per agreement score (may time out here; the export still works)', areaTable);
Export.table.toDrive({collection: areaTable, description: 'gaya_agreement_area',
  folder: EXPORT_FOLDER, fileFormat: 'CSV', selectors: ['score', 'area_ha']});

// ---------- Pixel samples for pairwise agreement (kappa) ----------
var samples = stack.sample({region: aoi, scale: 30, numPixels: 5000, seed: 11,
                            geometries: true, tileScale: 8});
Export.table.toDrive({collection: samples, description: 'gaya_agreement_samples',
  folder: EXPORT_FOLDER, fileFormat: 'CSV', selectors: ['s2_rabi', 'wc_irr', 'modis_rabi', '.geo']});

// ---------- Map for figures ----------
Export.image.toDrive({image: score, description: 'gaya_agreement_map', folder: EXPORT_FOLDER,
  region: aoi, scale: 30, crs: 'EPSG:32645', maxPixels: 1e13});
