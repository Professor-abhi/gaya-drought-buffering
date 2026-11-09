// =====================================================================
// SCRIPT 6 — Groundwater storage from GRACE/GRACE-FO, Gaya region (Bihar)
// "Buffering at what cost?" — monthly groundwater storage 2002–2025.
//
// Two estimates, both exported in one table:
//   A) GRACE mascon terrestrial water storage (TWS) minus GLDAS-2.1 Noah soil
//      moisture + canopy water + snow  ->  groundwater storage anomaly (GWSA)
//      (done in Script 7; the needed parts are exported here)
//   B) GLDAS-2.2 CLSM with GRACE data assimilation: groundwater storage directly
//      (used as a check, and as a fallback where GRACE has gaps)
//
// Note: GRACE's true resolution is ~300 km. Values describe the wider South
// Bihar region around Gaya, not individual blocks. State this in the paper.
//
// Export: gaya_grace_monthly.csv
// =====================================================================

var EXPORT_FOLDER = 'Gaya_Irrigation';
var gaya = ee.FeatureCollection('FAO/GAUL/2015/level2')
  .filter(ee.Filter.eq('ADM1_NAME', 'Bihar'))
  .filter(ee.Filter.eq('ADM2_NAME', 'Gaya'));
var region = gaya.geometry();
Map.centerObject(gaya, 8);

var grace = ee.ImageCollection('NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI').select('lwe_thickness'); // cm
var noah = ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H')
  .select(['SoilMoi0_10cm_inst', 'SoilMoi10_40cm_inst', 'SoilMoi40_100cm_inst',
           'SoilMoi100_200cm_inst', 'CanopInt_inst', 'SWE_inst']);              // kg/m2 = mm
var clsm = ee.ImageCollection('NASA/GLDAS/V022/CLSM/G025/DA1D').select('GWS_tavg'); // mm

// Check coverage before exporting (look at these dates in the Console)
print('GRACE mascon first / last month',
  ee.Date(grace.aggregate_min('system:time_start')), ee.Date(grace.aggregate_max('system:time_start')));
print('GLDAS-2.2 CLSM (GWS) first / last day',
  ee.Date(clsm.aggregate_min('system:time_start')), ee.Date(clsm.aggregate_max('system:time_start')));

var t0 = ee.Date.fromYMD(2002, 4, 1);
var nMonths = 9 + (2025 - 2002) * 12;   // Apr 2002 – Dec 2025

function regionMean(img, band, scale) {
  return img.reduceRegion({reducer: ee.Reducer.mean(), geometry: region, scale: scale,
                           maxPixels: 1e9}).get(band);
}

var rows = ee.List.sequence(0, nMonths - 1).map(function(k) {
  var d0 = t0.advance(k, 'month'), d1 = d0.advance(1, 'month');
  var g = grace.filterDate(d0, d1);
  var tws = ee.Algorithms.If(g.size().gt(0), regionMean(g.mean(), 'lwe_thickness', 55660), null);
  var n = noah.filterDate(d0, d1);
  var nm = n.mean();
  var sm = nm.select(['SoilMoi0_10cm_inst', 'SoilMoi10_40cm_inst', 'SoilMoi40_100cm_inst',
                      'SoilMoi100_200cm_inst']).reduce(ee.Reducer.sum()).rename('sm');
  var parts = sm.addBands(nm.select('CanopInt_inst').rename('cnp'))
                .addBands(nm.select('SWE_inst').rename('swe'));
  var p = ee.Algorithms.If(n.size().gt(0),
    parts.reduceRegion({reducer: ee.Reducer.mean(), geometry: region, scale: 27830, maxPixels: 1e9}),
    ee.Dictionary({}));
  p = ee.Dictionary(p);
  var c = clsm.filterDate(d0, d1);
  var gws = ee.Algorithms.If(c.size().gt(0), regionMean(c.mean(), 'GWS_tavg', 27830), null);
  return ee.Feature(null, {
    date: d0.format('YYYY-MM'), tws_cm: tws,
    sm_mm: p.get('sm', null), canopy_mm: p.get('cnp', null), swe_mm: p.get('swe', null),
    gws_clsm_mm: gws
  });
});

Export.table.toDrive({
  collection: ee.FeatureCollection(rows), description: 'gaya_grace_monthly',
  folder: EXPORT_FOLDER, fileFormat: 'CSV',
  selectors: ['date', 'tws_cm', 'sm_mm', 'canopy_mm', 'swe_mm', 'gws_clsm_mm']
});
print('Export task created: open the Tasks tab and click RUN on gaya_grace_monthly.');
