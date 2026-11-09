// =====================================================================
// SCRIPT 1e — SECOND INTERPRETER tool (blind), Gaya Rabi-cropped validation
// For a colleague who did NOT make the map or the first labels.
// 50 random points; the map class and the first labels are NOT shown.
// Paste into the Earth Engine Code Editor (any account), press Run, label each point.
// When done, click "Finish" and send the COPY BOX text to the author.
// =====================================================================
var POINTS = [[84.731622,24.507512],[84.871068,24.767367],[84.709647,24.609273],[85.214691,24.861782],[85.360433,24.886206],[84.925092,24.764876],[85.037645,24.887998],[85.085477,24.905412],[84.483589,24.314844],[84.566301,24.439675],[85.118882,24.895816],[84.731131,24.584143],[85.241335,24.571737],[85.190236,24.664849],[84.604851,24.481452],[85.168589,24.948166],[84.782641,24.518019],[85.092616,24.7885],[84.728062,24.932349],[85.279522,24.832207],[85.265169,24.754576],[85.022959,24.989637],[84.992952,24.509318],[84.882583,24.805445],[85.004093,24.888095],[84.704892,24.480027],[84.83759,24.472506],[84.838184,24.730877],[85.109996,24.679028],[84.519022,24.437282],[84.816115,24.670166],[84.948485,24.586179],[85.338179,24.995659],[85.156456,24.825052],[84.936547,24.446807],[85.059849,24.797016],[85.253271,24.650698],[85.248337,24.791487],[84.666494,24.572323],[85.31816,24.894133],[84.785243,24.672963],[85.15674,24.584269],[84.770102,25.054844],[85.291692,25.004612],[84.695496,24.621785],[84.673932,24.873293],[85.090755,24.813123],[84.443971,24.374836],[84.922235,24.829563],[84.997361,24.900463]];
var RESUME = [];   // paste a saved COPY BOX line here to continue later

Map.setOptions('SATELLITE');
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
var start = ee.Date('2023-06-01');
var ts = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate(start, start.advance(1, 'year'))
  .filterBounds(ee.Geometry.Rectangle([84.3, 24.25, 85.45, 25.1]))
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
  .linkCollection(csPlus, ['cs'])
  .map(function(img) {
    return ee.Image(img.updateMask(img.select('cs').gte(0.6)).normalizedDifference(['B8', 'B4'])
      .rename('NDVI').copyProperties(img, ['system:time_start']));
  });
var feb = ts.filterDate('2024-02-01', '2024-03-01').median().rename('feb')
  .addBands(ts.filterDate('2024-03-01', '2024-04-01').median().rename('mar'));

var labels = RESUME.slice(), idx = labels.length;
var info = ui.Label(''), chartBox = ui.Panel(), val = ui.Label('', {fontWeight: 'bold'});
var copy = ui.Textbox({value: '', style: {width: '390px', fontSize: '10px'}});
var layer = ui.Map.Layer(ee.Geometry.Point(POINTS[0]), {color: 'red'}, 'Point');
Map.layers().add(layer);

function show() {
  copy.setValue('var RESUME = ' + JSON.stringify(labels) + ';');
  if (idx >= POINTS.length) { info.setValue('All 50 done. Click Finish.'); chartBox.clear(); return; }
  var c = POINTS[idx], g = ee.Geometry.Point(c);
  layer.setEeObject(g);
  Map.setCenter(c[0], c[1], 17);
  info.setValue('Point ' + (idx + 1) + ' of ' + POINTS.length);
  chartBox.clear();
  chartBox.add(ui.Chart.image.series(ts, g, ee.Reducer.mean(), 10).setOptions({
    title: 'NDVI, Jun 2023 – May 2024', legend: {position: 'none'},
    vAxis: {viewWindow: {min: 0, max: 0.9}}, pointSize: 3, lineWidth: 1, height: 180}));
  var me = idx;
  val.setValue('February / March NDVI: loading…');
  feb.reduceRegion({reducer: ee.Reducer.mean(), geometry: g, scale: 10}).evaluate(function(v) {
    if (me !== idx) return;
    var f = function(x) { return (x === null || x === undefined) ? 'no clear image' : x.toFixed(2); };
    val.setValue('February NDVI: ' + f(v && v.feb) + '    March NDVI: ' + f(v && v.mar));
  });
}
function record(code) {
  if (idx >= POINTS.length) return;
  labels.push([POINTS[idx][0], POINTS[idx][1], code]);
  idx++;
  show();
}
function btn(t, c) { return ui.Button({label: t, onClick: function() { record(c); }}); }
var panel = ui.Panel({style: {position: 'bottom-left', width: '420px'}});
panel.add(ui.Label('SECOND INTERPRETER — label the red point', {fontWeight: 'bold'}));
panel.add(info); panel.add(chartBox); panel.add(val);
panel.add(ui.Panel([btn('Rabi-cropped', 0), btn('Rainfed', 1), btn('Fallow', 2), btn('Not sure', -1)],
                   ui.Panel.Layout.flow('horizontal')));
panel.add(ui.Panel([ui.Button({label: '◀ Back', onClick: function() { if (labels.length) { labels.pop(); idx--; show(); } }}),
                    ui.Button({label: 'Finish', onClick: function() {
                      print('Finished: ' + labels.length + ' points. Send this line to the author:', copy.getValue()); }})],
                   ui.Panel.Layout.flow('horizontal')));
panel.add(ui.Label('RABI-CROPPED = a winter crop still green in February (NDVI about 0.5 or more). ' +
  'RAINFED = green only Aug–Oct, or a winter crop that has faded by February. FALLOW = low all year. ' +
  'NOT SURE = not a crop field (trees, houses, water, field edge) or cannot decide.', {fontSize: '11px'}));
panel.add(ui.Label('COPY BOX — your progress (click, Ctrl+A, Ctrl+C):', {fontSize: '11px', fontWeight: 'bold'}));
panel.add(copy);
Map.add(panel);
show();
