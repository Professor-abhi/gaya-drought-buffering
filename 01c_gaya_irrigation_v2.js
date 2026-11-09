// =====================================================================
// SCRIPT 1c — VERSION 2 of the irrigated cropland map (stricter)
// + a blind validation tool. Gaya district (Bihar), Sentinel-2 + Sentinel-1
//
// Changes from version 1 (Script 1b), after comparison with official
// statistics showed v1 irrigated area ~2.5x the official net irrigated area:
//  a) Non-crop vegetation removed: a real crop field has a bare phase in the
//     year (10th-percentile NDVI < PERENNIAL_T). Trees, groves, scrub and
//     homestead vegetation stay green all year and are excluded.
//  b) Irrigated = crop green in February AND still green in early March
//     (late-sown wheat). Rainfed lentil/gram senesce by late February.
//  c) New, independent validation sample (different random seed).
//  d) Assets saved with a _v2 suffix; version 1 maps are not touched.
//
// How it works
//  1. Strict crop-calendar rules find fields that are clearly irrigated,
//     clearly rainfed or clearly fallow ("high-confidence pixels").
//  2. Random Forest is trained on a stratified sample of those pixels,
//     separately for every year, then classifies ALL cropland pixels.
//  3. A validation panel shows random points one by one (map class hidden).
//     You label each with a button -> independent accuracy for the paper.
//
// Classes: 0 = Irrigated, 1 = Rainfed, 2 = Fallow
// =====================================================================

// TWO MODES (the full-district Random Forest is too heavy to show live):
//   MODE = 'export'   -> Run, then in the Tasks tab click RUN on every task.
//                        Each year's map is classified in the background and
//                        saved to your Assets (about 20–60 min per year).
//   MODE = 'validate' -> after the exports finish: loads the saved maps
//                        (fast), shows them, prints areas, opens the
//                        validation tool and exports the area table.
// ---------------- SETTINGS ----------------
var MODE = 'preview';        // 'preview' (quick area check) -> 'export' -> 'validate'
var ASSET_ROOT = 'projects/gee-class-474916/assets/';  // copy from the Assets tab (see note below)
var VERSION = 'v2';
var YEARS = [2023];          // export 2023 first; add other years after it is validated
var RF_YEAR = 2023;          // year that is validated
var PERENNIAL_T = 0.35;      // 10th-percentile NDVI above this = green all year = not a crop field
var IRR_FEB_T = 0.5;         // irrigated: February NDVI at least this
var IRR_MAR_T = 0.4;         // irrigated: March NDVI at least this (crop not yet harvested)
var VAL_SEED = 2025;         // new seed -> new, independent validation points
var N_TRAIN = 500;           // automatic training points per class per year
var N_TREES = 100;           // Random Forest trees
var EXPORT_SCALE = 10;       // use 20 if an export fails with "memory capacity exceeded"
var VAL_PER_CLASS = 60;      // validation points per mapped class (blind); label at least 50 each
var EXPORT_FOLDER = 'Gaya_Irrigation';
var RESUME = [];             // paste saved progress here (see "Save progress" button)
// ASSET_ROOT: open the Assets tab (left panel). The top line shows your
// project, e.g. projects/gaya-research/assets — use that, ending with '/'.

// ---------------- STUDY AREA ----------------
var gaya = ee.FeatureCollection('FAO/GAUL/2015/level2')
  .filter(ee.Filter.eq('ADM1_NAME', 'Bihar'))
  .filter(ee.Filter.eq('ADM2_NAME', 'Gaya'));
var aoi = gaya.geometry();
Map.centerObject(gaya, 9);
Map.setOptions('SATELLITE');
Map.addLayer(gaya.style({color: 'red', fillColor: '00000000'}), {}, 'Gaya boundary');

var cropMask = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').eq(40);
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');

function s2Ndvi(start, end) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi).filterDate(start, end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
    .linkCollection(csPlus, ['cs'])
    .map(function(img) {
      var m = img.updateMask(img.select('cs').gte(0.6));
      return ee.Image(m.normalizedDifference(['B8', 'B4']).rename('NDVI')
        .addBands(m.normalizedDifference(['B8', 'B11']).rename('LSWI'))
        .copyProperties(img, ['system:time_start']));
    });
}

// ---------------- FEATURES FOR ONE AGRICULTURAL YEAR ----------------
function buildFeatures(year) {
  var start = ee.Date.fromYMD(year, 6, 1);
  var end = start.advance(1, 'year');
  var s2 = s2Ndvi(start, end);
  function win(k0, k1) { return s2.filterDate(start.advance(k0, 'month'), start.advance(k1, 'month')); }
  function ndviMonth(k) { return win(k, k + 1).select('NDVI').median().rename('NDVI_m' + k); }

  var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoi).filterDate(start, end)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .select(['VV', 'VH']);
  function s1Month(k) {
    var d0 = start.advance(k, 'month');
    return s1.filterDate(d0, d0.advance(1, 'month')).median().rename(['VV_m' + k, 'VH_m' + k]);
  }

  // Bare phase test: crop fields drop to bare soil after harvest; trees do not
  var p10 = s2.select('NDVI').reduce(ee.Reducer.percentile([10])).rename('NDVI_p10');
  var fieldMask = p10.lt(PERENNIAL_T);

  return ee.Image.cat([
    p10,
    win(1, 5).select('NDVI').max().rename('NDVI_kh_max'),     // Jul–Oct
    win(1, 5).select('NDVI').median().rename('NDVI_kh_med'),
    win(6, 10).select('NDVI').max().rename('NDVI_rb_max'),    // Dec–Mar
    win(6, 10).select('LSWI').median().rename('LSWI_rb_med'),
    win(10, 12).select('NDVI').max().rename('NDVI_su_max'),   // Apr–May
    ndviMonth(5), ndviMonth(6), ndviMonth(7), ndviMonth(8), ndviMonth(9),  // Nov, Dec, Jan, Feb, Mar
    s1Month(1), s1Month(2), s1Month(3), s1Month(4), s1Month(7)
  ]).unmask(-99).clip(aoi).updateMask(cropMask).updateMask(fieldMask).toFloat();
}

// ---------------- STEP 1: HIGH-CONFIDENCE LABELS FROM CROP CALENDAR ----------------
// Only pixels that match a rule very clearly get a label; everything in
// between stays unlabelled and is decided by the Random Forest.
function autoLabels(f) {
  var kh = f.select('NDVI_kh_max'), rb = f.select('NDVI_rb_max');
  var dec = f.select('NDVI_m6'), jan = f.select('NDVI_m7'), feb = f.select('NDVI_m8');
  var mar = f.select('NDVI_m9');
  var valid = kh.gt(-1).and(rb.gt(-1)).and(dec.gt(-1)).and(jan.gt(-1)).and(feb.gt(-1)).and(mar.gt(-1));
  var decJan = dec.max(jan);

  // Training labels are a little stricter than the validation rule (only clear cases)
  var irrigated = feb.gte(IRR_FEB_T + 0.1).and(mar.gte(IRR_MAR_T + 0.1));   // wheat green Feb–Mar
  var kharifOnly = kh.gte(0.5).and(rb.lte(0.3));                         // green in monsoon only
  var rainfedPulse = decJan.gte(0.35).and(feb.lt(IRR_FEB_T)).and(mar.lt(0.3)); // early Rabi, gone by Mar
  var fallow = kh.lt(0.3).and(rb.lt(0.25));                              // no crop all year

  var lab = ee.Image(-1)
    .where(kharifOnly.or(rainfedPulse), 1)
    .where(irrigated, 0)
    .where(fallow, 2)
    .updateMask(valid).updateMask(cropMask).rename('class');
  return lab.updateMask(lab.gte(0)).toByte();
}

// ---------------- STEP 2: RANDOM FOREST PER YEAR ----------------
function classifyYear(year) {
  var f = buildFeatures(year);
  var lab = autoLabels(f);
  var samples = f.addBands(lab).stratifiedSample({
    numPoints: N_TRAIN, classBand: 'class', region: aoi, scale: 30,
    seed: year, geometries: true, tileScale: 16
  }).randomColumn('rand', year);
  var train = samples.filter(ee.Filter.lt('rand', 0.8));
  var test = samples.filter(ee.Filter.gte('rand', 0.8));
  var rf = ee.Classifier.smileRandomForest({numberOfTrees: N_TREES, minLeafPopulation: 2})
    .train(train, 'class', f.bandNames());
  var map = f.classify(rf).updateMask(cropMask).clip(aoi).rename('class').toByte();
  return {features: f, labels: lab, rf: rf, test: test, map: map};
}

function areaFeature(img, year) {
  var stats = ee.Image.pixelArea().divide(1e4).addBands(img).reduceRegion({
    reducer: ee.Reducer.sum().group({groupField: 1, groupName: 'class'}),
    geometry: aoi, scale: 20, maxPixels: 1e13, tileScale: 8
  });
  var groups = ee.List(stats.get('groups'));
  var d = ee.Dictionary.fromLists(
    groups.map(function(g) { return ee.String('ha_class').cat(ee.Number(ee.Dictionary(g).get('class')).format('%d')); }),
    groups.map(function(g) { return ee.Dictionary(g).get('sum'); }));
  return ee.Feature(null, d).set({year: year, method: 'rf_auto_' + VERSION});
}

var PALETTE = ['1f78b4', 'b2df8a', 'd9d9d9'];
var VIS = {min: 0, max: 2, palette: PALETTE};

function assetId(y) { return ASSET_ROOT + 'gaya_irrigation_RF_' + VERSION + '_' + y; }

if (MODE === 'preview') {
  // ---- quick check (no Random Forest, so it is light): v2 RULES applied to every field
  //      pixel, area at 100 m. Tells us roughly what the v2 map will give before export. ----
  var pf = buildFeatures(RF_YEAR);
  var pvMap = ee.Image(1)                                                      // rainfed
    .where(pf.select('NDVI_kh_max').lt(0.3).and(pf.select('NDVI_rb_max').lt(0.25)), 2)   // fallow
    .where(pf.select('NDVI_m8').gte(IRR_FEB_T).and(pf.select('NDVI_m9').gte(IRR_MAR_T)), 0) // irrigated
    .updateMask(pf.select('NDVI_p10').mask()).rename('class').toByte();
  Map.addLayer(pvMap, VIS, 'v2 rule preview ' + RF_YEAR, false);
  var pvArea = ee.Image.pixelArea().divide(1e4).addBands(pvMap).reduceRegion({
    reducer: ee.Reducer.sum().group({groupField: 1, groupName: 'class'}),
    geometry: aoi, scale: 100, maxPixels: 1e13, tileScale: 16
  });
  print('v2 RULE PREVIEW area (ha) at 100 m, ' + RF_YEAR + ' [0=Irrigated, 1=Rainfed, 2=Fallow]', pvArea);
  print('Compare: v1 irrigated 263,132 ha; official net irrigated (2008-09) 100,300 ha; ' +
        'official Rabi crops ~135,000 ha; net sown area 235,800 ha.');
}

if (MODE === 'export') {
  // ---- background exports: one map per year ----
  YEARS.forEach(function(y) {
    var r = classifyYear(y);
    Export.image.toAsset({image: r.map, description: 'asset_gaya_irrigation_RF_' + VERSION + '_' + y,
      assetId: assetId(y), region: aoi, scale: EXPORT_SCALE, crs: 'EPSG:32645', maxPixels: 1e13,
      pyramidingPolicy: {'class': 'mode'}});
    if (y === RF_YEAR) {
      Export.image.toDrive({image: r.map, description: 'gaya_irrigation_RF_' + VERSION + '_' + y,
        folder: EXPORT_FOLDER, region: aoi, scale: EXPORT_SCALE, crs: 'EPSG:32645', maxPixels: 1e13});
      // Internal check + variable importance (small table, safe to export)
      var cmAuto = r.test.classify(r.rf).errorMatrix('class', 'classification');
      var imp = ee.Dictionary(r.rf.explain().get('importance'));
      Export.table.toDrive({
        collection: ee.FeatureCollection([ee.Feature(null, imp.set('auto_label_accuracy', cmAuto.accuracy()))]),
        description: 'gaya_rf_importance_' + VERSION + '_' + y, folder: EXPORT_FOLDER, fileFormat: 'CSV'});
      Map.addLayer(r.labels, VIS, 'Auto training labels ' + y + ' (zoom in to view)', false);
    }
  });
  print('MODE = export: open the Tasks tab and click RUN on each task (' + YEARS.length +
        ' maps + 2 tables). When all are finished, set MODE = \'validate\' and Run again.');
}

// ---- MODE = 'validate': everything below uses the saved maps (fast) ----
if (MODE === 'validate') {
var areaRows = [];
YEARS.forEach(function(y) {
  try {
    ee.data.getAsset(assetId(y));
    areaRows.push(areaFeature(ee.Image(assetId(y)).rename('class'), y));
  } catch (e) {
    print('Map for ' + y + ' not found yet (' + assetId(y) + ') — skipped.');
  }
});
var refMap = ee.Image(assetId(RF_YEAR)).rename('class');
// Hidden by default: during validation you must NOT see the map's answer
Map.addLayer(refMap, VIS, 'RF map ' + RF_YEAR + '-' + (RF_YEAR + 1), false);
var areaTable = ee.FeatureCollection(areaRows);
print('RF area (ha) by year [0=Irrigated, 1=Rainfed, 2=Fallow]', areaTable);

// Same file name as Script 1 so Script 3 reads it automatically
Export.table.toDrive({
  collection: areaTable, description: 'gaya_irrigated_area_by_year_' + VERSION,
  folder: EXPORT_FOLDER, fileFormat: 'CSV',
  selectors: ['year', 'method', 'ha_class0', 'ha_class1', 'ha_class2']
});

// ---------------- STEP 3: BLIND VALIDATION TOOL ----------------
var tsStart = ee.Date.fromYMD(RF_YEAR, 6, 1);
var tsCol = s2Ndvi(tsStart, tsStart.advance(1, 'year')).select('NDVI');

var valPts = refMap.stratifiedSample({
  numPoints: VAL_PER_CLASS, classBand: 'class', region: aoi, scale: 30,
  seed: VAL_SEED, geometries: true, tileScale: 16
}).randomColumn('order', 7).sort('order');          // shuffled: classes mixed, map class hidden

// RESUME may be [[lon,lat,map,ref], ...] or a flat list lon,lat,map,ref,lon,lat,...
// (inner brackets are often lost when copying from the Console)
function normalizeResume(r) {
  if (!r.length || typeof r[0] !== 'number') return r.slice();
  var out = [];
  for (var i = 0; i + 3 < r.length; i += 4) out.push([r[i], r[i + 1], r[i + 2], r[i + 3]]);
  if (r.length % 4 !== 0) print('Warning: RESUME has ' + r.length % 4 + ' extra number(s) at the end — ignored.');
  return out;
}
var pts = [], idx = 0, labels = normalizeResume(RESUME);
var info = ui.Label('Loading validation points… (can take 1–2 minutes)');
var chartBox = ui.Panel();
var febLabel = ui.Label('', {fontWeight: 'bold'});
var febImg = tsCol.filterDate(tsStart.advance(8, 'month'), tsStart.advance(9, 'month')).median()
  .addBands(tsCol.filterDate(tsStart.advance(9, 'month'), tsStart.advance(10, 'month')).median()
  .rename('NDVI_mar'));
var help = ui.Label(
  'v2 RULE. IRRIGATED = February NDVI 0.5 or more AND March NDVI 0.4 or more ' +
  '(late-sown wheat, still green until harvest in late March/April). ' +
  'RAINFED = green only Aug–Oct, OR a winter crop that is fading by February/March ' +
  '(gram, lentil on leftover moisture). FALLOW = low all year. ' +
  'NOT SURE = not a field (trees, groves, scrub, houses, water, field edge) or you cannot decide.',
  {fontSize: '11px'});

function record(code) {
  if (idx >= pts.length) return;
  var p = pts[idx];
  labels.push([+p.geometry.coordinates[0].toFixed(6), +p.geometry.coordinates[1].toFixed(6),
               p.properties['class'], code]);
  idx++;
  updateSaveBox();
  if (labels.length % 20 === 0) {
    print('Auto-save reminder at ' + labels.length + ' points: copy the COPY BOX line ' +
          '(click in it, Ctrl+A, Ctrl+C) into line 33 and Notepad, then Save.');
  }
  show();
}
function back() {
  if (idx > 0) { labels.pop(); idx--; show(); }   // only undoes points labelled this session
}
// Copy box: always holds the COMPLETE progress line (the Console shortens long lines)
var saveBox = ui.Textbox({value: '', style: {width: '390px', fontSize: '10px'}});
function updateSaveBox() {
  saveBox.setValue('var RESUME = ' + JSON.stringify(labels) + ';');
}
function saveProgress() {
  updateSaveBox();
  print('Progress: ' + labels.length + ' points. Copy the full line from the COPY BOX in the panel ' +
        '(click in it, Ctrl+A, Ctrl+C) and paste it over line 33, then Save.');
}
function finish() {
  var done = labels.filter(function(l) { return l[3] >= 0; });
  if (done.length === 0) { print('No points labelled yet.'); return; }
  var fc = ee.FeatureCollection(done.map(function(l) {
    return ee.Feature(ee.Geometry.Point([l[0], l[1]]), {map: l[2], ref: l[3]});
  }));
  var cm = fc.errorMatrix('ref', 'map', [0, 1, 2]);
  print('INDEPENDENT VALIDATION — ' + done.length + ' points (' +
        (labels.length - done.length) + ' "not sure" excluded)');
  print('Confusion matrix (rows = your label, columns = map; order Irrigated, Rainfed, Fallow)', cm);
  print('Overall accuracy', cm.accuracy());
  print('Kappa', cm.kappa());
  print("Producer's accuracy (per class)", cm.producersAccuracy());
  print("User's accuracy (per class)", cm.consumersAccuracy());
  Export.table.toDrive({collection: fc, description: 'gaya_validation_points_' + VERSION + '_' + RF_YEAR,
    folder: EXPORT_FOLDER, fileFormat: 'CSV', selectors: ['map', 'ref', '.geo']});
  print('Validation points added to the Tasks tab — click RUN to save them.');
}

var ptLayer = ui.Map.Layer(ee.Geometry.Point([85.0, 24.75]), {color: 'red'}, 'Validation point', false);
Map.layers().add(ptLayer);

function show() {
  if (idx >= pts.length) {
    info.setValue('All ' + pts.length + ' points labelled. Click "Finish & compute accuracy".');
    chartBox.clear();
    return;
  }
  var c = pts[idx].geometry.coordinates;
  var g = ee.Geometry.Point(c);
  ptLayer.setEeObject(g);
  ptLayer.setShown(true);
  Map.setCenter(c[0], c[1], 17);
  info.setValue('Point ' + (idx + 1) + ' of ' + pts.length + '   (' + c[0].toFixed(5) + ', ' + c[1].toFixed(5) + ')');
  chartBox.clear();
  chartBox.add(ui.Chart.image.series(tsCol, g, ee.Reducer.mean(), 10).setOptions({
    title: 'NDVI, Jun ' + RF_YEAR + ' – May ' + (RF_YEAR + 1), legend: {position: 'none'},
    vAxis: {viewWindow: {min: 0, max: 0.9}}, pointSize: 3, lineWidth: 1, height: 180
  }));
  var token = idx;
  febLabel.setValue('February NDVI here: loading…');
  febImg.reduceRegion({reducer: ee.Reducer.mean(), geometry: g, scale: 10}).evaluate(function(v) {
    if (token !== idx) return;                       // user already moved on
    var fmt = function(x) { return (x === null || x === undefined) ? 'no clear image' : x.toFixed(2); };
    febLabel.setValue('February NDVI: ' + fmt(v && v.NDVI) + '    March NDVI: ' + fmt(v && v.NDVI_mar));
  });
}

function button(label, code, color) {
  return ui.Button({label: label, style: {color: color}, onClick: function() { record(code); }});
}
var vPanel = ui.Panel({style: {position: 'bottom-left', width: '420px'}});
vPanel.add(ui.Label('VALIDATION — label the red point', {fontWeight: 'bold'}));
vPanel.add(info);
vPanel.add(chartBox);
vPanel.add(febLabel);
vPanel.add(ui.Panel([button('Irrigated', 0, '1f78b4'), button('Rainfed', 1, '33a02c'),
                     button('Fallow', 2, '636363'), button('Not sure', -1, 'black')],
                    ui.Panel.Layout.flow('horizontal')));
vPanel.add(ui.Panel([ui.Button({label: '◀ Back', onClick: back}),
                     ui.Button({label: 'Save progress', onClick: saveProgress}),
                     ui.Button({label: 'Finish & compute accuracy', onClick: finish})],
                    ui.Panel.Layout.flow('horizontal')));
vPanel.add(help);
vPanel.add(ui.Label('COPY BOX — your full progress (click in box, Ctrl+A, Ctrl+C):',
                    {fontSize: '11px', fontWeight: 'bold'}));
vPanel.add(saveBox);
Map.add(vPanel);

valPts.evaluate(function(fc, err) {
  if (err) { info.setValue('Error loading points: ' + err); return; }
  // Skip every point already in RESUME (matched by coordinates, any order)
  var key = function(lon, lat) { return lon.toFixed(5) + ',' + lat.toFixed(5); };
  var done = {};
  labels.forEach(function(l) { done[key(l[0], l[1])] = true; });
  pts = fc.features.filter(function(p) {
    return !done[key(p.geometry.coordinates[0], p.geometry.coordinates[1])];
  });
  idx = 0;
  updateSaveBox();
  if (labels.length) {
    print('Resumed: ' + labels.length + ' points already labelled, ' + pts.length + ' remaining.');
  }
  show();
});
}  // end MODE = 'validate'
