// =====================================================================
// SCRIPT 14 — Exports for Figure 1 (study area map)
// Exports to Drive folder Gaya_Irrigation:
//   fig1_rabi_map_30m.tif        2023–24 RF map (1 Rabi-cropped, 2 rainfed, 3 fallow, 0 no data), 30 m
//   fig1_gaya_boundary           Gaya district (SHP)
//   fig1_region_districts        Gaya + the six neighbouring districts (SHP)
//   fig1_bihar_districts         all Bihar districts (SHP)
//   fig1_india_states            Indian states (SHP)
// Then run: python 14b_figure1_study_area.py
// =====================================================================
var S2_ASSET = 'projects/gee-class-474916/assets/gaya_irrigation_RF_2023';
var gaul2 = ee.FeatureCollection('FAO/GAUL/2015/level2');
var gaul1 = ee.FeatureCollection('FAO/GAUL/2015/level1');
var bihar = gaul2.filter(ee.Filter.eq('ADM1_NAME', 'Bihar'));
var gaya = bihar.filter(ee.Filter.eq('ADM2_NAME', 'Gaya'));
var region = bihar.filter(ee.Filter.inList('ADM2_NAME',
  ['Gaya', 'Nawada', 'Aurangabad', 'Jehanabad', 'Nalanda', 'Rohtas', 'Bhojpur']));
var india = gaul1.filter(ee.Filter.eq('ADM0_NAME', 'India'));
var keep = ['ADM0_NAME', 'ADM1_NAME', 'ADM2_NAME'];

// classes shifted by +1 so that 0 = no data (GeoTIFF export fills masked pixels with 0)
var map = ee.Image(S2_ASSET).add(1).unmask(0).toByte().clip(gaya.geometry());
Map.centerObject(gaya, 9);
Map.addLayer(map.selfMask(), {min: 1, max: 3, palette: ['1f78b4', 'e6ab02', 'bdbdbd']}, 'RF map 2023–24');

Export.image.toDrive({image: map, description: 'fig1_rabi_map_30m', folder: 'Gaya_Irrigation',
  region: gaya.geometry().bounds(), scale: 30, crs: 'EPSG:4326', maxPixels: 1e10});
function shp(fc, name, props) {
  Export.table.toDrive({collection: fc.map(function(f) {
    return ee.Feature(ee.Feature(f).select(props)).simplify(100); }),
    description: name, folder: 'Gaya_Irrigation', fileFormat: 'SHP'});
}
shp(gaya, 'fig1_gaya_boundary', keep);
shp(region, 'fig1_region_districts', keep);
shp(bihar, 'fig1_bihar_districts', keep);
// India as GeoJSON: simplified state outlines mix geometry types, which SHP cannot hold
Export.table.toDrive({collection: india.map(function(f) {
    return ee.Feature(ee.Feature(f).select(['ADM0_NAME', 'ADM1_NAME'])).simplify(2000); }),
  description: 'fig1_india_states', folder: 'Gaya_Irrigation', fileFormat: 'GeoJSON'});
