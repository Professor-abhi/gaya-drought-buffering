// =====================================================================
// SCRIPT 2 — Drought indices for irrigated vs rainfed cropland, Gaya
// MODIS NDVI + LST (2000–2025), CHIRPS rainfall, ERA5-Land temperature
// Produces 3 CSV tables + 2 GeoTIFFs for the Python analysis (Script 3).
// =====================================================================

// ---------------- SETTINGS ----------------
// Leave '' to classify irrigated/rainfed from MODIS itself (works right away).
// After Script 1 exports the RF map as an Asset, paste its ID here, e.g.
// 'projects/your-project/assets/gaya_irrigation_RF_2023'
var S2_ASSET = 'projects/gee-class-474916/assets/gaya_irrigation_RF_2023';
// true = keep only pixels where BOTH sources agree (recommended):
//   irrigated core = our Rabi-cropped class AND ESA WorldCereal 2021 irrigated
//   rainfed core   = our rainfed class      AND NOT WorldCereal irrigated
var USE_WORLDCEREAL = true;
var KH_Y0 = 2001, KH_Y1 = 2025;   // Kharif seasons (Aug 15 – Oct 31)
var RB_Y0 = 2000, RB_Y1 = 2024;   // Rabi of agri year y (Jan 1 – Mar 15 of y+1)
var GREEN_T = 0.5;                // MODIS NDVI "crop present" threshold
var EXPORT_FOLDER = 'Gaya_Irrigation';

// ---------------- STUDY AREA ----------------
var gaya = ee.FeatureCollection('FAO/GAUL/2015/level2')
  .filter(ee.Filter.eq('ADM1_NAME', 'Bihar'))
  .filter(ee.Filter.eq('ADM2_NAME', 'Gaya'));
var aoi = gaya.geometry();
Map.centerObject(gaya, 9);

var modisProj = ee.ImageCollection('MODIS/061/MOD13Q1').first().projection();

// ---------------- MODIS NDVI & LST ----------------
var ndviCol = ee.ImageCollection('MODIS/061/MOD13Q1').filterBounds(aoi).map(function(img) {
  return ee.Image(img.select('NDVI').multiply(0.0001)
    .updateMask(img.select('SummaryQA').lte(1))             // good + marginal quality
    .copyProperties(img, ['system:time_start']));
});
var lstCol = ee.ImageCollection('MODIS/061/MOD11A2').filterBounds(aoi).map(function(img) {
  return ee.Image(img.select('LST_Day_1km').multiply(0.02).subtract(273.15)
    .copyProperties(img, ['system:time_start']));
});

var SEASONS = {
  kharif: {m0: 8, d0: 15, m1: 11, d1: 1, offset: 0},
  rabi:   {m0: 1, d0: 1,  m1: 3,  d1: 16, offset: 1}
};

function yearly(key, y0, y1) {
  var s = SEASONS[key], imgs = [];
  for (var y = y0; y <= y1; y++) {
    var d0 = ee.Date.fromYMD(y + s.offset, s.m0, s.d0);
    var d1 = ee.Date.fromYMD(y + s.offset, s.m1, s.d1);
    var n = ndviCol.filterDate(d0, d1), l = lstCol.filterDate(d0, d1);
    imgs.push(n.mean().rename('NDVI')
      .addBands(n.max().rename('NDVImax'))
      .addBands(l.mean().rename('LST'))
      .set({year: y, season: key}));
  }
  return ee.ImageCollection(imgs);
}

// VCI, TCI, VHI (Kogan) — per pixel, relative to the pixel's own record
function addIndices(col) {
  var nmin = col.select('NDVI').min(), nmax = col.select('NDVI').max();
  var lmin = col.select('LST').min(), lmax = col.select('LST').max();
  return col.map(function(i) {
    var vci = i.select('NDVI').subtract(nmin).divide(nmax.subtract(nmin)).multiply(100).rename('VCI');
    var tci = lmax.subtract(i.select('LST')).divide(lmax.subtract(lmin)).multiply(100).rename('TCI');
    var vhi = vci.multiply(0.5).add(tci.multiply(0.5)).rename('VHI');
    return i.addBands([vci, tci, vhi]);
  });
}

var kh = addIndices(yearly('kharif', KH_Y0, KH_Y1));
var rb = addIndices(yearly('rabi', RB_Y0, RB_Y1));

// ---------------- IRRIGATED / RAINFED PIXEL GROUPS ----------------
// No explicit reproject: the 10 m -> 250 m averaging is done tile by tile at the
// scale each reduce/sample/export asks for (avoids "Reprojection output too large").
var cropFrac = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').eq(40)
  .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024});

var rabiGreenFreq = rb.map(function(i) { return i.select('NDVImax').gt(GREEN_T); }).mean();
var khGreenFreq = kh.map(function(i) { return i.select('NDVImax').gt(GREEN_T); }).mean();

var groups, irrIndex;
if (S2_ASSET === '') {
  // MODIS method: persistent Rabi greenness = irrigated
  groups = ee.Image(0)
    .where(rabiGreenFreq.gte(0.8), 1)                                   // 1 = irrigated
    .where(khGreenFreq.gte(0.8).and(rabiGreenFreq.lte(0.2)), 2);        // 2 = rainfed
  irrIndex = rabiGreenFreq;
} else {
  // Sentinel-2 RF map aggregated to MODIS 250 m
  var s2map = ee.Image(S2_ASSET);
  var irrMask = s2map.eq(0), rainMask = s2map.eq(1);
  if (USE_WORLDCEREAL) {
    var wcIrr = ee.ImageCollection('ESA/WorldCereal/2021/MODELS/v100')
      .filterBounds(aoi)
      .filter(ee.Filter.eq('product', 'irrigation'))
      .filter(ee.Filter.eq('season', 'tc-wintercereals'))
      .mosaic().select('classification').eq(100).unmask(0);
    irrMask = irrMask.and(wcIrr);
    rainMask = rainMask.and(wcIrr.not());
  }
  var frac = function(m) {
    return m.setDefaultProjection(s2map.projection())      // fixed 10 m input grid
      .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024});
  };
  var irrF = frac(irrMask), rainF = frac(rainMask);
  groups = ee.Image(0).where(irrF.gte(0.7), 1).where(rainF.gte(0.6), 2);
  irrIndex = irrF;
}
groups = groups.updateMask(cropFrac.gte(0.5)).selfMask().rename('grp');
Map.addLayer(groups, {min: 1, max: 2, palette: ['1f78b4', 'e6ab02']}, 'Groups: 1 irrigated, 2 rainfed');
Map.addLayer(irrIndex.updateMask(cropFrac.gte(0.5)), {min: 0, max: 1, palette: ['fff7bc', '2171b5']}, 'Irrigation index', false);

// ---------------- TABLE 1: indices by group, year, season ----------------
var BANDS = ['NDVI', 'LST', 'VCI', 'TCI', 'VHI'];
function groupStats(col) {
  return ee.FeatureCollection(col.map(function(i) {
    var res = i.select(BANDS).addBands(groups).reduceRegion({
      reducer: ee.Reducer.mean().repeat(BANDS.length).group({groupField: BANDS.length, groupName: 'grp'}),
      geometry: aoi, scale: 250, maxPixels: 1e13, tileScale: 4
    });
    return ee.FeatureCollection(ee.List(res.get('groups')).map(function(g) {
      g = ee.Dictionary(g);
      var m = ee.List(g.get('mean'));
      return ee.Feature(null, {
        year: i.get('year'), season: i.get('season'), grp: g.get('grp'),
        NDVI: m.get(0), LST: m.get(1), VCI: m.get(2), TCI: m.get(3), VHI: m.get(4)
      });
    }));
  })).flatten();
}
var table1 = groupStats(kh).merge(groupStats(rb));
print('Preview: indices by group', table1.limit(10));
Export.table.toDrive({
  collection: table1, description: 'gaya_vhi_by_group', folder: EXPORT_FOLDER,
  fileFormat: 'CSV', selectors: ['year', 'season', 'grp', 'NDVI', 'LST', 'VCI', 'TCI', 'VHI']
});

// ---------------- TABLE 2: monthly climate 1981–2025 ----------------
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY');
var era = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR');
var t0 = ee.Date.fromYMD(1981, 1, 1);
var nMonths = (2025 - 1981 + 1) * 12;
var table2 = ee.FeatureCollection(ee.List.sequence(0, nMonths - 1).map(function(k) {
  var d0 = t0.advance(k, 'month'), d1 = d0.advance(1, 'month');
  var p = chirps.filterDate(d0, d1).sum().reduceRegion({
    reducer: ee.Reducer.mean(), geometry: aoi, scale: 5566, maxPixels: 1e9}).get('precipitation');
  var t = era.filterDate(d0, d1).select(['temperature_2m_max', 'temperature_2m_min']).mean()
    .reduceRegion({reducer: ee.Reducer.mean(), geometry: aoi, scale: 11132, maxPixels: 1e9});
  return ee.Feature(null, {
    date: d0.format('YYYY-MM'), pr_mm: p,
    tmax_C: ee.Number(t.get('temperature_2m_max')).subtract(273.15),
    tmin_C: ee.Number(t.get('temperature_2m_min')).subtract(273.15)
  });
}));
Export.table.toDrive({
  collection: table2, description: 'gaya_climate_monthly', folder: EXPORT_FOLDER,
  fileFormat: 'CSV', selectors: ['date', 'pr_mm', 'tmax_C', 'tmin_C']
});

// ---------------- TABLE 3: pixel drought sensitivity ----------------
// Per pixel: slope of Kharif VHI against standardized JJAS rainfall anomaly.
// Low slope = crop health does not fall when rain fails = buffered.
var jjas = [];
for (var y = KH_Y0; y <= KH_Y1; y++) {
  jjas.push(chirps.filterDate(ee.Date.fromYMD(y, 6, 1), ee.Date.fromYMD(y, 10, 1))
    .sum().rename('P').set('year', y));
}
jjas = ee.ImageCollection(jjas);
var pMean = jjas.mean(), pSd = jjas.reduce(ee.Reducer.stdDev());
var pairs = ee.ImageCollection(jjas.map(function(p) {
  var vhi = ee.Image(kh.filter(ee.Filter.eq('year', p.get('year'))).first()).select('VHI');
  return p.subtract(pMean).divide(pSd).rename('SPA').addBands(vhi);
}));
var fit = pairs.select(['SPA', 'VHI']).reduce(ee.Reducer.linearFit());          // scale, offset
var corr = pairs.select(['SPA', 'VHI']).reduce(ee.Reducer.pearsonsCorrelation()); // correlation, p-value
var sens = fit.select('scale').rename('slope')
  .addBands(corr.select('correlation').rename('r'))
  .addBands(irrIndex.rename('irr_index'))
  .addBands(groups.unmask(0).rename('grp'))
  .updateMask(cropFrac.gte(0.5)).clip(aoi);
Map.addLayer(sens.select('slope'), {min: 0, max: 15, palette: ['2c7bb6', 'ffffbf', 'd7191c']}, 'VHI sensitivity to rainfall');

// ---------------- DRIVERS of drought sensitivity (explainable ML in Script 7) ----------------
var dem = ee.Image('USGS/SRTMGL1_003');
var rivers = ee.FeatureCollection('WWF/HydroSHEDS/v1/FreeFlowingRivers').filterBounds(aoi.buffer(30000));
var drivers = dem.rename('elev_m')
  .addBands(ee.Terrain.slope(dem).rename('slope_deg'))
  .addBands(rivers.distance({searchRadius: 30000, maxError: 50}).unmask(30000).rename('dist_river_m'))
  .addBands(ee.Image('OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02').select('b30').rename('clay_pct'))
  .addBands(pSd.divide(pMean).rename('rain_cv'))
  .addBands(era.filter(ee.Filter.calendarRange(3, 3, 'month')).filterDate('2001-01-01', '2026-01-01')
    .select('temperature_2m_max').mean().subtract(273.15).rename('mar_tmax_C'));
var DRIVER_NAMES = ['elev_m', 'slope_deg', 'dist_river_m', 'clay_pct', 'rain_cv', 'mar_tmax_C'];
if (S2_ASSET !== '' && USE_WORLDCEREAL) {
  drivers = drivers.addBands(frac(wcIrr).rename('wc_irr_frac'));
  DRIVER_NAMES.push('wc_irr_frac');
}
sens = sens.addBands(drivers.updateMask(cropFrac.gte(0.5)).clip(aoi));

var table3 = sens.sample({region: aoi, scale: 250, numPixels: 6000, seed: 7, geometries: true,
                          tileScale: 4});
Export.table.toDrive({
  collection: table3, description: 'gaya_pixel_sensitivity', folder: EXPORT_FOLDER,
  fileFormat: 'CSV', selectors: ['slope', 'r', 'irr_index', 'grp'].concat(DRIVER_NAMES).concat(['.geo'])
});

// ---------------- MAPS FOR FIGURES ----------------
Export.image.toDrive({image: sens.select(['slope', 'r', 'irr_index']).toFloat(),
  description: 'gaya_sensitivity_map', folder: EXPORT_FOLDER, region: aoi, scale: 250, maxPixels: 1e13});
Export.image.toDrive({image: groups.toByte(), description: 'gaya_groups_map',
  folder: EXPORT_FOLDER, region: aoi, scale: 250, maxPixels: 1e13});
