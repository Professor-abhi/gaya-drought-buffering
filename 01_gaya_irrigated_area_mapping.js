// =====================================================================
// SCRIPT 1 — Irrigated cropland mapping, Gaya district (Bihar)
// Sentinel-2 (optical) + Sentinel-1 (SAR), agricultural years 2019–2025
// Paste into https://code.earthengine.google.com and press Run.
//
// Classes (cropland only, non-crop masked with ESA WorldCover):
//   0 = Irrigated   (Rabi crop present in the dry season, Dec–Mar)
//   1 = Rainfed     (Kharif crop only)
//   2 = Fallow      (no crop in either season)
//
// PART A (rule-based) runs immediately, no training data needed.
// PART B (Random Forest) needs training points — see instructions below,
// then set USE_RF = true.
// =====================================================================

// ---------------- SETTINGS ----------------
var YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]; // agri year = Jun YEAR – May YEAR+1
var RF_YEAR = 2023;      // year for which you collect training points
var NDVI_T = 0.45;       // Kharif crop/no-crop NDVI threshold
var AUTO_THRESHOLD = true;   // Otsu threshold for Jan–Feb NDVI (irrigated Rabi crop)
var RABI_T_MANUAL = 0.5;     // used only if AUTO_THRESHOLD = false
var USE_RF = false;      // set true after adding training points
var EXPORT_FOLDER = 'Gaya_Irrigation';

// ---------------- STUDY AREA ----------------
var gaya = ee.FeatureCollection('FAO/GAUL/2015/level2')
  .filter(ee.Filter.eq('ADM1_NAME', 'Bihar'))
  .filter(ee.Filter.eq('ADM2_NAME', 'Gaya'));
var aoi = gaya.geometry();
Map.centerObject(gaya, 9);
Map.addLayer(gaya.style({color: 'red', fillColor: '00000000'}), {}, 'Gaya boundary');

// Cropland mask (ESA WorldCover 2021, class 40 = cropland)
var cropMask = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').eq(40);

// ---------------- FEATURE STACK FOR ONE AGRICULTURAL YEAR ----------------
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');

function buildFeatures(year) {
  var start = ee.Date.fromYMD(year, 6, 1);
  var end = ee.Date.fromYMD(year + 1, 6, 1);

  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi).filterDate(start, end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
    .linkCollection(csPlus, ['cs'])
    .map(function(img) {
      var m = img.updateMask(img.select('cs').gte(0.6));     // Cloud Score+ mask
      var ndvi = m.normalizedDifference(['B8', 'B4']).rename('NDVI');
      var lswi = m.normalizedDifference(['B8', 'B11']).rename('LSWI');
      return ee.Image(ndvi.addBands(lswi).copyProperties(img, ['system:time_start']));
    });

  var kharif = s2.filterDate(start.advance(1, 'month'), start.advance(5, 'month'));   // Jul–Oct
  var rabi   = s2.filterDate(start.advance(6, 'month'), start.advance(10, 'month'));  // Dec–Mar
  var summer = s2.filterDate(start.advance(10, 'month'), end);                         // Apr–May

  var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoi).filterDate(start, end)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .select(['VV', 'VH']);
  function s1Month(k) {   // k = months after June
    var d0 = start.advance(k, 'month');
    return s1.filterDate(d0, d0.advance(1, 'month')).median()
      .rename(['VV_m' + k, 'VH_m' + k]);
  }

  return ee.Image.cat([
    kharif.select('NDVI').max().rename('NDVI_kh_max'),
    kharif.select('NDVI').median().rename('NDVI_kh_med'),
    rabi.select('NDVI').max().rename('NDVI_rb_max'),
    rabi.select('NDVI').median().rename('NDVI_rb_med'),
    rabi.select('LSWI').median().rename('LSWI_rb_med'),
    s2.filterDate(start.advance(7, 'month'), start.advance(9, 'month'))  // Jan–Feb wheat peak
      .select('NDVI').median().rename('NDVI_rb_peak'),
    summer.select('NDVI').max().rename('NDVI_su_max'),
    s1Month(1), s1Month(2), s1Month(3), s1Month(4),   // Jul–Oct (sees through monsoon cloud)
    s1Month(7)                                         // Jan
  ]).unmask(-99).clip(aoi).updateMask(cropMask).toFloat();
}

// ---------------- PART A: RULE-BASED MAP ----------------
// Irrigated = sustained green crop in the Jan–Feb wheat peak (median NDVI),
// not a single green moment. The threshold is found automatically (Otsu)
// from the Jan–Feb NDVI histogram of RF_YEAR and used for every year.
function otsu(histogram) {
  histogram = ee.Dictionary(histogram);
  var counts = ee.Array(histogram.get('histogram'));
  var means = ee.Array(histogram.get('bucketMeans'));
  var size = means.length().get([0]);
  var total = counts.reduce(ee.Reducer.sum(), [0]).get([0]);
  var sum = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0]);
  var mean = sum.divide(total);
  var bss = ee.List.sequence(1, size.subtract(1)).map(function(i) {
    var aCounts = counts.slice(0, 0, i);
    var aCount = aCounts.reduce(ee.Reducer.sum(), [0]).get([0]);
    var aMean = means.slice(0, 0, i).multiply(aCounts).reduce(ee.Reducer.sum(), [0]).get([0]).divide(aCount);
    var bCount = total.subtract(aCount);
    var bMean = sum.subtract(aCount.multiply(aMean)).divide(bCount);
    return aCount.multiply(aMean.subtract(mean).pow(2)).add(bCount.multiply(bMean.subtract(mean).pow(2)));
  });
  return ee.Number(means.slice(0, 0, size.subtract(1)).sort(ee.Array(bss)).get([-1]));
}
var rbPeakRef = buildFeatures(RF_YEAR).select('NDVI_rb_peak');
rbPeakRef = rbPeakRef.updateMask(rbPeakRef.gt(-1));            // drop no-data (-99)
var rbHist = rbPeakRef.reduceRegion({
  reducer: ee.Reducer.histogram(100, 0.01), geometry: aoi, scale: 60, maxPixels: 1e13, tileScale: 8
}).get('NDVI_rb_peak');
var RABI_T = AUTO_THRESHOLD ? otsu(rbHist) : ee.Number(RABI_T_MANUAL);
print('Jan–Feb NDVI threshold used for "irrigated"', RABI_T);
print(ui.Chart.image.histogram({image: rbPeakRef, region: aoi, scale: 60, maxPixels: 1e13})
  .setOptions({title: 'Jan–Feb median NDVI on cropland (two humps = irrigated vs not)',
               hAxis: {title: 'NDVI'}, vAxis: {title: 'pixels'}}));

function ruleMap(feat) {
  return ee.Image(2)                                           // fallow
    .where(feat.select('NDVI_kh_max').gte(NDVI_T), 1)          // rainfed (Kharif only)
    .where(feat.select('NDVI_rb_peak').gte(RABI_T), 0)         // irrigated (sustained Rabi crop)
    .updateMask(cropMask).clip(aoi).rename('class').toByte();
}

var PALETTE = ['1f78b4', 'b2df8a', 'd9d9d9'];   // irrigated, rainfed, fallow
var VIS = {min: 0, max: 2, palette: PALETTE};

function areaFeature(img, year, method) {
  var stats = ee.Image.pixelArea().divide(1e4).addBands(img).reduceRegion({
    reducer: ee.Reducer.sum().group({groupField: 1, groupName: 'class'}),
    geometry: aoi, scale: 20, maxPixels: 1e13, tileScale: 8
  });
  var groups = ee.List(stats.get('groups'));
  var d = ee.Dictionary.fromLists(
    groups.map(function(g) { return ee.String('ha_class').cat(ee.Number(ee.Dictionary(g).get('class')).format('%d')); }),
    groups.map(function(g) { return ee.Dictionary(g).get('sum'); }));
  return ee.Feature(null, d).set({year: year, method: method});
}

var areaRows = [];
YEARS.forEach(function(y) {
  var feat = buildFeatures(y);
  var rm = ruleMap(feat);
  areaRows.push(areaFeature(rm, y, 'rule'));
  if (y === RF_YEAR) Map.addLayer(rm, VIS, 'Rule-based ' + y + '-' + (y + 1));
});
var areaTable = ee.FeatureCollection(areaRows);
print('Area (ha) by class per year [0=Irrigated, 1=Rainfed, 2=Fallow]', areaTable);
Export.table.toDrive({
  collection: areaTable, description: 'gaya_irrigated_area_by_year',
  folder: EXPORT_FOLDER, fileFormat: 'CSV',
  selectors: ['year', 'method', 'ha_class0', 'ha_class1', 'ha_class2']
});

// NDVI time-series helper for labelling training points:
// select the hand tool, click any field, and its NDVI profile for RF_YEAR appears.
var featRF = buildFeatures(RF_YEAR);
var tsStart = ee.Date.fromYMD(RF_YEAR, 6, 1);
var tsCol = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi).filterDate(tsStart, tsStart.advance(1, 'year'))
  .linkCollection(csPlus, ['cs'])
  .map(function(img) {
    return ee.Image(img.updateMask(img.select('cs').gte(0.6))
      .normalizedDifference(['B8', 'B4']).rename('NDVI')
      .copyProperties(img, ['system:time_start']));
  });
var tsPanel = ui.Panel({style: {position: 'bottom-right', width: '430px'}});
tsPanel.add(ui.Label('Hand tool: click a field to see its NDVI profile'));
Map.add(tsPanel);
Map.onClick(function(c) {
  var pt = ee.Geometry.Point([c.lon, c.lat]);
  tsPanel.clear();
  tsPanel.add(ui.Label('Lon ' + c.lon.toFixed(5) + ', Lat ' + c.lat.toFixed(5)));
  tsPanel.add(ui.Chart.image.series(tsCol, pt, ee.Reducer.mean(), 10).setOptions({
    title: 'NDVI, Jun ' + RF_YEAR + ' – May ' + (RF_YEAR + 1),
    vAxis: {viewWindow: {min: 0, max: 0.9}}, pointSize: 3, lineWidth: 1, legend: {position: 'none'}
  }));
  tsPanel.add(ui.Label('Irrigated wheat: NDVI > 0.6 through Jan–Feb, peak in Feb. ' +
    'Rainfed gram/lentil/mustard: lower, earlier peak (Dec–Jan). ' +
    'Rainfed Kharif-only: green Aug–Oct, bare in winter. Fallow: low all year.',
    {fontSize: '11px'}));
});
Map.addLayer(featRF.select(['NDVI_kh_max', 'NDVI_rb_peak', 'NDVI_su_max']),
  {min: 0, max: 0.8}, 'NDVI composite (R=Kharif, G=Jan–Feb, B=Summer)', false);

// ---------------- PART B: RANDOM FOREST ----------------
// HOW TO ADD TRAINING POINTS (for RF_YEAR):
//  1. In the map, use the "Add a marker" tool to create 3 point layers:
//       irrigated, rainfed, fallow
//  2. For each layer click the gear icon -> Import as: FeatureCollection,
//     add property  class  = 0 / 1 / 2 respectively.
//  3. Put 150–300 POINTS per class (points, not polygons), spread across
//     all 24 blocks. Check each point in Google Earth Pro high-res imagery
//     and with the NDVI composite layer above (Rabi green = irrigated).
//  4. Set USE_RF = true and run again.
if (USE_RF) {
  var training = irrigated.merge(rainfed).merge(fallow);
  var samples = featRF.sampleRegions({
    collection: training, properties: ['class'], scale: 10, tileScale: 8
  }).randomColumn('rand', 42);
  var trainSet = samples.filter(ee.Filter.lt('rand', 0.7));
  var testSet = samples.filter(ee.Filter.gte('rand', 0.7));

  var rf = ee.Classifier.smileRandomForest(300)
    .train(trainSet, 'class', featRF.bandNames());
  var rfMap = featRF.classify(rf).updateMask(cropMask).clip(aoi).rename('class').toByte();
  Map.addLayer(rfMap, VIS, 'Random Forest ' + RF_YEAR);

  var cm = testSet.classify(rf).errorMatrix('class', 'classification');
  print('Confusion matrix (rows = reference)', cm);
  print('Overall accuracy', cm.accuracy());
  print('Kappa', cm.kappa());
  print("Producer's accuracy", cm.producersAccuracy());
  print("User's accuracy", cm.consumersAccuracy());
  print('Variable importance', ee.Dictionary(rf.explain().get('importance')));

  print('RF area (ha)', areaFeature(rfMap, RF_YEAR, 'rf'));

  // Map for figures (GeoTIFF) and as an Asset for Script 2
  Export.image.toDrive({
    image: rfMap, description: 'gaya_irrigation_RF_' + RF_YEAR, folder: EXPORT_FOLDER,
    region: aoi, scale: 10, maxPixels: 1e13
  });
  Export.image.toAsset({
    image: rfMap, description: 'asset_gaya_irrigation_RF_' + RF_YEAR,
    assetId: 'gaya_irrigation_RF_' + RF_YEAR, region: aoi, scale: 10, maxPixels: 1e13
  });
}

// Rule-based map for RF_YEAR as GeoTIFF too
Export.image.toDrive({
  image: ruleMap(featRF), description: 'gaya_irrigation_rule_' + RF_YEAR,
  folder: EXPORT_FOLDER, region: aoi, scale: 10, maxPixels: 1e13
});
