// =====================================================================
// SCRIPT 2c — Revision groups for the reviewer's concerns 2 and 5 (Gaya)
// Same VHI method as Script 2 (MODIS NDVI + LST, 2000–2025), new pixel groups:
//   1 = PERSISTENTLY IRRIGATED: >=70% Rabi-cropped (Sentinel-2 map) AND WorldCereal-irrigated,
//       and Rabi-green (MODIS Jan–mid-Mar max NDVI > 0.5) in >= 80% of years 2000–2024
//   2 = PERSISTENTLY RAINFED (Kharif-only): >=60% rainfed and not WorldCereal-irrigated,
//       and Rabi-green in <= 20% of years
//   3 = RAINFED RABI CROPS (residual moisture): >=60% Rabi-cropped in the Sentinel-2 map,
//       <=10% WorldCereal-irrigated, Rabi-green in >= 50% of years
// Use: Kharif drought test 1 vs 2 (no misclassified years); Rabi heat test 1 vs 3 (crop vs crop).
// Export: gaya_vhi_revision_groups.csv  (folder Gaya_Irrigation)
// =====================================================================
var S2_ASSET = 'projects/gee-class-474916/assets/gaya_irrigation_RF_2023';
var gaya = ee.FeatureCollection('FAO/GAUL/2015/level2').filter(ee.Filter.eq('ADM1_NAME', 'Bihar')).filter(ee.Filter.eq('ADM2_NAME', 'Gaya'));
var aoi = gaya.geometry();
Map.centerObject(gaya, 9);
var ndviCol = ee.ImageCollection('MODIS/061/MOD13Q1').filterBounds(aoi).map(function(img) { return ee.Image(img.select('NDVI').multiply(0.0001).updateMask(img.select('SummaryQA').lte(1)).copyProperties(img, ['system:time_start'])); });
var lstCol = ee.ImageCollection('MODIS/061/MOD11A2').filterBounds(aoi).map(function(img) { return ee.Image(img.select('LST_Day_1km').multiply(0.02).subtract(273.15).copyProperties(img, ['system:time_start'])); });
var SEAS = {kharif: {m0: 8, d0: 15, m1: 11, d1: 1, off: 0}, rabi: {m0: 1, d0: 1, m1: 3, d1: 16, off: 1}};
function yearly(key, y0, y1) { var s = SEAS[key], imgs = []; for (var y = y0; y <= y1; y++) { var a = ee.Date.fromYMD(y + s.off, s.m0, s.d0), b = ee.Date.fromYMD(y + s.off, s.m1, s.d1); var n = ndviCol.filterDate(a, b), l = lstCol.filterDate(a, b); imgs.push(n.mean().rename('NDVI').addBands(n.max().rename('NDVImax')).addBands(l.mean().rename('LST')).set({year: y, season: key})); } return ee.ImageCollection(imgs); }
function addIdx(col) { var nmin = col.select('NDVI').min(), nmax = col.select('NDVI').max(), lmin = col.select('LST').min(), lmax = col.select('LST').max(); return col.map(function(i) { var vci = i.select('NDVI').subtract(nmin).divide(nmax.subtract(nmin)).multiply(100).rename('VCI'); var tci = lmax.subtract(i.select('LST')).divide(lmax.subtract(lmin)).multiply(100).rename('TCI'); return i.addBands([vci, tci, vci.multiply(0.5).add(tci.multiply(0.5)).rename('VHI')]); }); }
var kh = addIdx(yearly('kharif', 2001, 2025)), rb = addIdx(yearly('rabi', 2000, 2024));
var cropFrac = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').eq(40).reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024});
var rabiGreenFreq = rb.map(function(i) { return i.select('NDVImax').gt(0.5); }).mean();
var s2map = ee.Image(S2_ASSET);
var wcIrr = ee.ImageCollection('ESA/WorldCereal/2021/MODELS/v100').filterBounds(aoi).filter(ee.Filter.eq('product', 'irrigation')).filter(ee.Filter.eq('season', 'tc-wintercereals')).mosaic().select('classification').eq(100).unmask(0);
function frac(m) { return m.setDefaultProjection(s2map.projection()).reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024}); }
var irrF = frac(s2map.eq(0).and(wcIrr));
var rainF = frac(s2map.eq(1).and(wcIrr.not()));
var rabiF = frac(s2map.eq(0));
var wcF = frac(wcIrr);
var groups = ee.Image(0)
  .where(irrF.gte(0.7).and(rabiGreenFreq.gte(0.8)), 1)
  .where(rainF.gte(0.6).and(rabiGreenFreq.lte(0.2)), 2)
  .where(rabiF.gte(0.6).and(wcF.lte(0.1)).and(rabiGreenFreq.gte(0.5)), 3)
  .updateMask(cropFrac.gte(0.5)).selfMask().rename('grp');
Map.addLayer(groups, {min: 1, max: 3, palette: ['1f78b4', 'e6ab02', '33a02c']}, 'Revision groups: 1 persistent irrigated, 2 persistent rainfed, 3 rainfed Rabi crop');
var B = ['NDVI', 'LST', 'VCI', 'TCI', 'VHI'];
function stats(col) { return ee.FeatureCollection(col.map(function(i) { var res = i.select(B).addBands(groups).reduceRegion({reducer: ee.Reducer.mean().repeat(5).group({groupField: 5, groupName: 'grp'}), geometry: aoi, scale: 250, maxPixels: 1e13, tileScale: 4}); return ee.FeatureCollection(ee.List(res.get('groups')).map(function(g) { g = ee.Dictionary(g); var m = ee.List(g.get('mean')); return ee.Feature(null, {year: i.get('year'), season: i.get('season'), grp: g.get('grp'), NDVI: m.get(0), LST: m.get(1), VCI: m.get(2), TCI: m.get(3), VHI: m.get(4)}); })); })).flatten(); }
var t = stats(kh).merge(stats(rb));
Export.table.toDrive({collection: t, description: 'gaya_vhi_revision_groups', folder: 'Gaya_Irrigation', fileFormat: 'CSV', selectors: ['year', 'season', 'grp', 'NDVI', 'LST', 'VCI', 'TCI', 'VHI']});
// pixel counts per group (for the paper)
var counts = ee.Image.pixelArea().divide(1e4).addBands(groups).reduceRegion({reducer: ee.Reducer.sum().group({groupField: 1, groupName: 'grp'}), geometry: aoi, scale: 250, maxPixels: 1e13, tileScale: 4});
Export.table.toDrive({collection: ee.FeatureCollection(ee.List(counts.get('groups')).map(function(g) { g = ee.Dictionary(g); return ee.Feature(null, {grp: g.get('grp'), area_ha: g.get('sum')}); })), description: 'gaya_revision_group_areas', folder: 'Gaya_Irrigation', fileFormat: 'CSV', selectors: ['grp', 'area_ha']});
