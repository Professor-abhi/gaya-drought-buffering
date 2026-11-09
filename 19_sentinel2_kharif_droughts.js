// =====================================================================
// SCRIPT 19 — Field-scale check of Kharif drought buffering with Sentinel-2 (reviewer concerns M1, M4)
// Groups are defined WITHOUT any Kharif signal (so the test is not circular):
//   1 = winter-irrigated: WorldCereal 2021 winter-cereal irrigation (10 m) AND MODIS Rabi-green in >= 80% of
//       years 2000-2024 (Jan-mid-Mar max NDVI > 0.5)
//   2 = rainfed: WorldCereal not irrigated AND MODIS Rabi-green in <= 20% of years
//   both within ESA WorldCover cropland
// Kharif greenness = 90th percentile of Sentinel-2 NDVI (Cloud Score+ >= 0.6), 15 Aug - 31 Oct, each year
// 2018-2025. Drought years (SPEI-4 Sep <= -1): 2022, 2023. Normal: 2018-2021, 2024, 2025.
// Output (printed and exported): group mean NDVI per year; per-pixel drought anomaly (drought year minus the
// pixel's normal-year mean); share of pixels whose NDVI fell by more than 0.1.
// Export: gaya_s2_kharif_droughts.csv (folder Gaya_Irrigation)
// =====================================================================
var gaya = ee.FeatureCollection('FAO/GAUL/2015/level2').filter(ee.Filter.eq('ADM1_NAME', 'Bihar'))
  .filter(ee.Filter.eq('ADM2_NAME', 'Gaya'));
var aoi = gaya.geometry();
var SCALE = 20;
var YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025];
var DROUGHT = [2022, 2023];

// groups (no Kharif information used)
var crop = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').eq(40);
var wcIrr = ee.ImageCollection('ESA/WorldCereal/2021/MODELS/v100').filterBounds(aoi)
  .filter(ee.Filter.eq('product', 'irrigation')).filter(ee.Filter.eq('season', 'tc-wintercereals'))
  .mosaic().select('classification').eq(100).unmask(0);
var modis = ee.ImageCollection('MODIS/061/MOD13Q1').filterBounds(aoi);
var rabiGreen = ee.ImageCollection(ee.List.sequence(2000, 2024).map(function(y) {
  y = ee.Number(y);
  return modis.filterDate(ee.Date.fromYMD(y.add(1), 1, 1), ee.Date.fromYMD(y.add(1), 3, 16))
    .select('NDVI').max().multiply(0.0001).gt(0.5);
})).mean();
var grp = ee.Image(0).where(wcIrr.eq(1).and(rabiGreen.gte(0.8)), 1)
  .where(wcIrr.eq(0).and(rabiGreen.lte(0.2)), 2)
  .updateMask(crop).selfMask().rename('grp');

// Kharif NDVI (p90) per year
var cs = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
function kharif(y) {
  var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(aoi)
    .filterDate(ee.Date.fromYMD(y, 8, 15), ee.Date.fromYMD(y, 11, 1))
    .linkCollection(cs, ['cs'])
    .map(function(i) { return i.updateMask(i.select('cs').gte(0.6)).normalizedDifference(['B8', 'B4']); });
  return col.reduce(ee.Reducer.percentile([90])).rename('ndvi_' + y);
}
var stack = ee.Image.cat(YEARS.map(kharif));
var normalNames = YEARS.filter(function(y) { return DROUGHT.indexOf(y) < 0; }).map(function(y) { return 'ndvi_' + y; });
var normal = stack.select(normalNames).reduce(ee.Reducer.mean()).rename('ndvi_normal');
var anoms = ee.Image.cat(DROUGHT.map(function(y) {
  var a = stack.select('ndvi_' + y).subtract(normal);
  return a.rename('anom_' + y).addBands(a.lt(-0.1).rename('drop01_' + y));
}));
var img = stack.addBands(normal).addBands(anoms);
var bands = img.bandNames();

var res = img.addBands(grp).reduceRegion({
  reducer: ee.Reducer.mean().repeat(bands.length()).group({groupField: bands.length(), groupName: 'grp'})
    .combine(ee.Reducer.count().unweighted(), null, true),
  geometry: aoi, scale: SCALE, maxPixels: 1e13, tileScale: 16});
var fc = ee.FeatureCollection(ee.List(ee.Dictionary(res).get('groups')).map(function(g) {
  g = ee.Dictionary(g);
  return ee.Feature(null, ee.Dictionary.fromLists(bands, ee.List(g.get('mean'))).set('grp', g.get('grp')));
}));
print('Kharif Sentinel-2 NDVI by group', fc);
var area = ee.Image.pixelArea().divide(1e4).addBands(grp).reduceRegion({
  reducer: ee.Reducer.sum().group({groupField: 1, groupName: 'grp'}), geometry: aoi, scale: SCALE,
  maxPixels: 1e13, tileScale: 16});
print('Group areas (ha)', area);
Map.centerObject(gaya, 9);
Map.addLayer(grp, {min: 1, max: 2, palette: ['1f78b4', 'e6ab02']}, 'Groups (no Kharif info)');
Map.addLayer(anoms.select('anom_2023'), {min: -0.3, max: 0.1, palette: ['b2182b', 'f7f7f7', '2166ac']},
  'Kharif NDVI anomaly 2023');
Export.table.toDrive({collection: fc, description: 'gaya_s2_kharif_droughts', folder: 'Gaya_Irrigation',
  fileFormat: 'CSV'});
