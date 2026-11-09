// =====================================================================
// SCRIPT 2b — South Bihar regional replication of the drought-buffering test
// Same VHI method as Script 2; irrigated/rainfed groups from ESA WorldCereal + MODIS only,
// so the definition is identical in every district (Gaya included).
//   irrigated = >= 60% WorldCereal winter-season irrigated and >= 50% temporary crops (250 m pixel)
//   rainfed   = >= 60% temporary crops, <= 10% irrigated, Kharif-green in >= 80% of years
// Districts: Gaya, Nawada, Aurangabad, Jehanabad (incl. Arwal in GAUL 2015), Nalanda,
//            Rohtas and Bhojpur (Sone canal command, for contrast)
// Exports: region_vhi_by_group.csv, region_climate_monthly.csv  (folder Gaya_Irrigation)
// =====================================================================
var NAMES = ['Gaya','Nawada','Aurangabad','Jehanabad','Nalanda','Rohtas','Bhojpur'];
var dists = ee.FeatureCollection('FAO/GAUL/2015/level2').filter(ee.Filter.eq('ADM1_NAME','Bihar')).filter(ee.Filter.inList('ADM2_NAME', NAMES)).map(function(f){ return ee.Feature(f.geometry(), {district: f.get('ADM2_NAME')}); });
var region = dists.geometry().bounds();
Map.centerObject(dists, 8); Map.addLayer(dists, {color: 'red'}, 'districts');
var ndviCol = ee.ImageCollection('MODIS/061/MOD13Q1').filterBounds(region).map(function(img){ return ee.Image(img.select('NDVI').multiply(0.0001).updateMask(img.select('SummaryQA').lte(1)).copyProperties(img, ['system:time_start'])); });
var lstCol = ee.ImageCollection('MODIS/061/MOD11A2').filterBounds(region).map(function(img){ return ee.Image(img.select('LST_Day_1km').multiply(0.02).subtract(273.15).copyProperties(img, ['system:time_start'])); });
var SEAS = {kharif: {m0: 8, d0: 15, m1: 11, d1: 1, off: 0}, rabi: {m0: 1, d0: 1, m1: 3, d1: 16, off: 1}};
function yearly(key, y0, y1) { var s = SEAS[key], imgs = []; for (var y = y0; y <= y1; y++) { var a = ee.Date.fromYMD(y + s.off, s.m0, s.d0), b = ee.Date.fromYMD(y + s.off, s.m1, s.d1); var n = ndviCol.filterDate(a, b), l = lstCol.filterDate(a, b); imgs.push(n.mean().rename('NDVI').addBands(n.max().rename('NDVImax')).addBands(l.mean().rename('LST')).set({year: y, season: key})); } return ee.ImageCollection(imgs); }
function addIdx(col) { var nmin = col.select('NDVI').min(), nmax = col.select('NDVI').max(), lmin = col.select('LST').min(), lmax = col.select('LST').max(); return col.map(function(i){ var vci = i.select('NDVI').subtract(nmin).divide(nmax.subtract(nmin)).multiply(100).rename('VCI'); var tci = lmax.subtract(i.select('LST')).divide(lmax.subtract(lmin)).multiply(100).rename('TCI'); return i.addBands([vci, tci, vci.multiply(0.5).add(tci.multiply(0.5)).rename('VHI')]); }); }
var kh = addIdx(yearly('kharif', 2001, 2025)), rb = addIdx(yearly('rabi', 2000, 2024));
var wc = ee.ImageCollection('ESA/WorldCereal/2021/MODELS/v100').filterBounds(region);
function wcl(p, s) { return wc.filter(ee.Filter.eq('product', p)).filter(ee.Filter.eq('season', s)).mosaic().select('classification').eq(100).unmask(0).setDefaultProjection({crs: 'EPSG:4326', scale: 10}); }
function frac(m) { return m.reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024}); }
var irrF = frac(wcl('irrigation', 'tc-wintercereals')), cropF = frac(wcl('temporarycrops', 'tc-annual'));
var khGreen = kh.map(function(i){ return i.select('NDVImax').gt(0.5); }).mean();
var groups = ee.Image(0).where(irrF.gte(0.6).and(cropF.gte(0.5)), 1).where(cropF.gte(0.6).and(irrF.lte(0.1)).and(khGreen.gte(0.8)), 2).selfMask().rename('grp');
Map.addLayer(groups, {min: 1, max: 2, palette: ['1f78b4', 'e6ab02']}, 'Groups: 1 irrigated, 2 rainfed');
var B = ['NDVI', 'LST', 'VCI', 'TCI', 'VHI'];
function stats(col) { return ee.FeatureCollection(col.map(function(i){ var img = i.select(B).addBands(groups); return img.reduceRegions({collection: dists, reducer: ee.Reducer.mean().repeat(5).group({groupField: 5, groupName: 'grp'}), scale: 250, tileScale: 4}).map(function(f){ return ee.FeatureCollection(ee.List(ee.Algorithms.If(f.get('groups'), f.get('groups'), [])).map(function(g){ g = ee.Dictionary(g); var m = ee.List(g.get('mean')); return ee.Feature(null, {district: f.get('district'), year: i.get('year'), season: i.get('season'), grp: g.get('grp'), NDVI: m.get(0), LST: m.get(1), VCI: m.get(2), TCI: m.get(3), VHI: m.get(4)}); })); }).flatten(); })).flatten(); }
var t1 = stats(kh).merge(stats(rb));
Export.table.toDrive({collection: t1, description: 'region_vhi_by_group', folder: 'Gaya_Irrigation', fileFormat: 'CSV', selectors: ['district', 'year', 'season', 'grp', 'NDVI', 'LST', 'VCI', 'TCI', 'VHI']});
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY'), era = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR');
var t0 = ee.Date.fromYMD(1981, 1, 1);
var t2 = ee.FeatureCollection(ee.List.sequence(0, 45 * 12 - 1).map(function(k){ var a = t0.advance(k, 'month'), b = a.advance(1, 'month'); var img = chirps.filterDate(a, b).sum().rename('pr_mm').addBands(era.filterDate(a, b).select(['temperature_2m_max', 'temperature_2m_min']).mean().subtract(273.15).rename(['tmax_C', 'tmin_C'])); return img.reduceRegions({collection: dists, reducer: ee.Reducer.mean(), scale: 5566}).map(function(f){ return f.set('date', a.format('YYYY-MM')).setGeometry(null); }); })).flatten();
Export.table.toDrive({collection: t2, description: 'region_climate_monthly', folder: 'Gaya_Irrigation', fileFormat: 'CSV', selectors: ['district', 'date', 'pr_mm', 'tmax_C', 'tmin_C']});
