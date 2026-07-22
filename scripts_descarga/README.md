# Scripts de descarga y conversion a NetCDF CDT

Esta carpeta agrupa scripts Python para descargar fuentes remotas y convertirlas al formato NetCDF compatible con CDT.

## Script principal IMERG V07

Archivo: `gpm_imerg_v07_to_cdt.py`

Implementa la logica equivalente a CDT para:

- `GPM_L3_IMERG_V07_EARLY_daily`
- `GPM_L3_IMERG_V07_FINAL_daily`
- `GPM_L3_IMERG_V07_FINAL_monthly`

## Script principal CHIRPSv2

Archivo: `chirps_v2_to_cdt.py`

Implementa la logica equivalente a CDT para:

- `CHIRPSv2_daily`
- `CHIRPSv2_monthly`

## Script principal GSMaP CLM

Archivo: `gsmap_clm_to_cdt.py`

Implementa la logica equivalente a CDT para:

- `GSMaP_CLM_daily`
- `GSMaP_CLM_monthly`

Soporta dos modos de entrada:

- `--source ftp` para descargar desde JAXA FTP
- `--source local` para usar archivos ya descargados localmente

## Script principal MSWEP v2

Archivo: `mswep_v2_to_cdt.py`

Implementa la logica equivalente a CDT para:

- `MSWEP_Past_v2.8_daily`
- `MSWEP_Past_v2.8_monthly`

Descarga desde Google Drive usando `rclone` y transforma cada NetCDF al formato CDT.

Folder IDs por defecto:

- monthly: `16BS6ezP7AEJPgZ8dA1FH6-IIIZUp8ASE`
- daily: `1gWoZ2bK2u5osJ8Iw-dvguZ56Kmz2QWrL`

## Script principal PERSIANN-CDR/CCS

Archivo: `persiann_cdr_ccs_to_cdt.py`

Implementa la logica equivalente a CDT para:

- `PERSIANN-CDR_daily`
- `PERSIANN-CDR_monthly`
- `PERSIANN-CCS_daily`
- `PERSIANN-CCS_monthly`

## Script principal ERA5-Land 1Hr

Archivo: `era5_land_1hr_to_cdt.py`

Implementa la logica equivalente a CDT para ERA5-Land horario con seleccion de variables
(`--variables`), por ejemplo:

- `evp` (evaporation)
- `pet` (potential evapotranspiration; variable ERA5 `pev`, salida `pet_YYYYMMDDHH.nc`)
- `tair`, `wind`, `prcp`, etc.

## Script de blanqueo NetCDF por shapefile

Archivo: `blank_netcdf_by_shapefile.py`

Aplica blanqueo como CDT a todos los NetCDF de una carpeta:

- crea mascara con un shapefile de poligonos de referencia
- pone missing fuera del poligono (equivalente a NA fuera)
- escribe nuevos NetCDF en carpeta de salida

Opciones de buffer (como CDT):

- `--buffer-option default`: buffer = 4 x resolucion espacial de la grilla
- `--buffer-option user --buffer-width X`: buffer manual en grados

## Script CLIMATOLOGY/ANOMALIES/SPI (equivalente CDT)

Archivo: `cdt_clim_anom_spi_from_netcdf.py`

Genera carpetas y NetCDF compatibles con flujo CDT para series mensuales:

- `CLIMATOLOGY_data` (mean/std por mes)
- `ANOMALIES_data` (Difference, Percentage, Standardized)
- `SPI_data` (escalas configurables, por ejemplo `1,3,6,12,24`)

Notas:

- entrada esperada: carpeta con NetCDF mensuales (fecha `YYYYMM` en nombre)
- orientado a precipitacion mensual en grilla regular
- para anomalias puede reutilizar una climatologia existente con `--climatology-dir`

### Dependencias Python

```bash
python3 -m pip install netCDF4 numpy rasterio requests geopandas shapely
```

Tambien requiere `curl` en el sistema.

Para GSMaP tambien se usa `gzip` del sistema para descomprimir `.dat.gz`.

Para MSWEP se requiere `rclone` configurado con un remoto Google Drive (por defecto `gdrive`).

Para ERA5-Land se usa la API de CDS (Copernicus) mediante peticiones HTTP.

### Credenciales con .env

Puedes guardar credenciales locales en un archivo `.env` en la raiz del repo.

Variables soportadas:

- `EARTHDATA_USERNAME`
- `EARTHDATA_PASSWORD`
- `GSMAP_FTP_USER`
- `GSMAP_FTP_PASSWORD`
- `CDS_TOKEN`

Hay una plantilla en `.env.example`.

El `.env` no se sube al repo; `.env.example` si.

### Area geografica por defecto para descargas

Si no se especifica `--minlon`, `--maxlon`, `--minlat` y `--maxlat`, los scripts principales de descarga usan por defecto el bbox de Panama:

- lon: `-84.2312` a `-75.9853`
- lat: `6.721` a `10.114`

## Namelist para ejecucion por lotes

Hay una plantilla en `download_namelist.ini` y un ejecutor en `scripts_descarga/run_namelist.py`.

Idea general:

- una seccion `[global]` para valores comunes como `outdir`, bbox y `python`
- una seccion `[job:nombre]` por tarea
- `enabled = true` activa una tarea
- `args = ...` permite pasar parametros extra especificos del script

Ejemplo de uso:

```bash
python3 scripts_descarga/run_namelist.py --namelist download_namelist.ini --dry-run
```

Ejecutar solo algunas tareas:

```bash
python3 scripts_descarga/run_namelist.py \
  --namelist download_namelist.ini \
  --only gpm_early_daily,era5_land_pet
```

Por defecto las tareas habilitadas se ejecutan secuencialmente, en el orden del
namelist. Cada tarea procesa su rango `start`/`end` completo antes de comenzar la
siguiente. Para ejecutar hasta cuatro productos completos en paralelo:

```bash
python3 scripts_descarga/run_namelist.py \
  --namelist download_namelist.ini \
  --jobs 4 \
  --continue-on-error
```

Cada producto conserva su propia ventana temporal. `--jobs` controla productos
simultaneos, no descargas individuales dentro de un producto.

Notas:

- por defecto agrega automaticamente `--start`, `--end`, `--minlon`, `--maxlon`, `--minlat`, `--maxlat`, `--outdir` y `--verbose` cuando existen en el namelist
- para scripts que no usan fechas o bbox, desactiva esos grupos con `use_dates = false` o `use_bbox = false`
- para pasar opciones especiales como `--product`, `--variables`, `--source` o rutas de entrada, usa `args = ...`

### Credenciales NASA Earthdata

Se leen desde argumentos o variables de entorno:

- `EARTHDATA_USERNAME`
- `EARTHDATA_PASSWORD`

## Uso rapido

### 1) EARLY daily

```bash
python3 scripts_descarga/gpm_imerg_v07_to_cdt.py \
  --product early_daily \
  --start 20260601 \
  --end 20260605 \
  --minlon -75 --maxlon -70 --minlat -15 --maxlat -10 \
  --outdir . \
  --verbose
```

### 2) FINAL monthly

```bash
python3 scripts_descarga/gpm_imerg_v07_to_cdt.py \
  --product final_monthly \
  --start 202501 \
  --end 202503 \
  --minlon -75 --maxlon -70 --minlat -15 --maxlat -10 \
  --outdir . \
  --verbose
```

### 3) CHIRPSv2 daily

```bash
python3 scripts_descarga/chirps_v2_to_cdt.py \
  --product daily \
  --start 20260101 \
  --end 20260105 \
  --minlon -75 --maxlon -70 --minlat -15 --maxlat -10 \
  --outdir . \
  --verbose
```

### 4) CHIRPSv2 monthly

```bash
python3 scripts_descarga/chirps_v2_to_cdt.py \
  --product monthly \
  --start 202501 \
  --end 202503 \
  --minlon -75 --maxlon -70 --minlat -15 --maxlat -10 \
  --outdir . \
  --verbose
```

### 5) GSMaP CLM monthly (desde local)

```bash
python3 scripts_descarga/gsmap_clm_to_cdt.py \
  --product monthly \
  --source local \
  --local-root /home/adrian/wildfire-cathalac-platform/local/CDT/GSMaP \
  --start 202601 \
  --end 202606 \
  --minlon -75 --maxlon -70 --minlat -15 --maxlat -10 \
  --outdir . \
  --verbose
```

### 6) GSMaP CLM daily (desde FTP)

```bash
python3 scripts_descarga/gsmap_clm_to_cdt.py \
  --product daily \
  --source ftp \
  --start 20260101 \
  --end 20260103 \
  --minlon -75 --maxlon -70 --minlat -15 --maxlat -10 \
  --outdir . \
  --verbose
```

### 7) MSWEP v2 monthly

```bash
python3 scripts_descarga/mswep_v2_to_cdt.py \
  --product monthly \
  --start 202601 \
  --end 202606 \
  --minlon -75 --maxlon -70 --minlat -15 --maxlat -10 \
  --outdir . \
  --verbose
```

### 8) MSWEP v2 daily

```bash
python3 scripts_descarga/mswep_v2_to_cdt.py \
  --product daily \
  --start 20260101 \
  --end 20260105 \
  --minlon -75 --maxlon -70 --minlat -15 --maxlat -10 \
  --outdir . \
  --verbose
```

### 9) PERSIANN-CDR daily

```bash
python3 scripts_descarga/persiann_cdr_ccs_to_cdt.py \
  --source cdr \
  --tstep daily \
  --start 20200101 \
  --end 20200103 \
  --minlon -75 --maxlon -70 --minlat -15 --maxlat -10 \
  --outdir . \
  --verbose
```

### 10) PERSIANN-CCS monthly

```bash
python3 scripts_descarga/persiann_cdr_ccs_to_cdt.py \
  --source ccs \
  --tstep monthly \
  --start 202001 \
  --end 202003 \
  --minlon -75 --maxlon -70 --minlat -15 --maxlat -10 \
  --outdir . \
  --verbose
```

### 11) ERA5-Land 1Hr (evapotranspiracion potencial)

```bash
python3 scripts_descarga/era5_land_1hr_to_cdt.py \
  --start 2020010100 \
  --end 2020010123 \
  --variables pet \
  --minlon -75 --maxlon -70 --minlat -15 --maxlat -10 \
  --outdir . \
  --verbose
```

### 12) Blanqueo NetCDF por shapefile

```bash
python3 scripts_descarga/blank_netcdf_by_shapefile.py \
  --input-dir /ruta/netcdf_entrada \
  --output-dir /ruta/netcdf_salida \
  --shapefile /ruta/shape/area.shp \
  --buffer-option user \
  --buffer-width 0 \
  --recursive \
  --verbose
```

### 13) Climatologia + Anomalias + SPI (todo en una corrida)

```bash
python3 scripts_descarga/cdt_clim_anom_spi_from_netcdf.py \
  --input-dir /ruta/netcdf_mensual \
  --output-dir /ruta/salidas_cdt \
  --operation all \
  --all-years \
  --min-years 15 \
  --anomaly-type Difference \
  --spi-scales 1,3,6,12,24 \
  --spi-distribution gamma \
  --verbose
```

### 14) Solo ANOMALIES usando climatologia ya calculada

```bash
python3 scripts_descarga/cdt_clim_anom_spi_from_netcdf.py \
  --input-dir /ruta/netcdf_mensual \
  --output-dir /ruta/salidas_cdt \
  --operation anomaly \
  --anomaly-type Standardized \
  --climatology-dir /ruta/salidas_cdt/CLIMATOLOGY_data \
  --date-start 200001 \
  --date-end 202412 \
  --verbose
```

### 15) SPEI (precipitacion - PET) en escalas mensuales

```bash
python3 scripts_descarga/cdt_spei_from_netcdf.py \
  --precip-dir /ruta/precip_mensual_netcdf \
  --pet-dir /ruta/pet_mensual_netcdf \
  --output-dir /ruta/salidas_cdt \
  --scales 1,3,6,12,24 \
  --distribution gamma \
  --date-start 198101 \
  --date-end 202412 \
  --verbose
```

### 16) Deciles mensuales (base period configurable)

```bash
python3 scripts_descarga/cdt_deciles_from_netcdf.py \
  --input-dir /ruta/precip_mensual_netcdf \
  --output-dir /ruta/salidas_cdt \
  --scale 3 \
  --base-start-year 1991 \
  --base-end-year 2020 \
  --min-year 20 \
  --date-start 198101 \
  --date-end 202412 \
  --verbose
```

### 17) Balance hidrico diario (P y PET diarios)

```bash
python3 scripts_descarga/cdt_water_balance_from_netcdf.py \
  --precip-dir /ruta/precip_diario_netcdf \
  --pet-dir /ruta/pet_diario_netcdf \
  --output-dir /ruta/salidas_cdt \
  --capacity-max 100 \
  --initial-wb 0 \
  --date-start 20000101 \
  --date-end 20241231 \
  --verbose
```

Opcionalmente, para emular capacidad/condicion inicial espacialmente variable:

- `--capacity-grid /ruta/capacidad_max.nc`
- `--initial-grid /ruta/wb_inicial.nc`

### 18) CLIMDEX precipitacion (indices diarios anuales)

```bash
python3 scripts_descarga/cdt_climdex_rr_from_netcdf.py \
  --input-dir /ruta/precip_diario_netcdf \
  --output-dir /ruta/salidas_cdt \
  --indices Rx1day,Rx5day,R10mm,R20mm,Rnnmm,CDD,CWD,PRCPTOT \
  --rnn-threshold 25 \
  --min-frac 0.95 \
  --year-start 1981 \
  --year-end 2024 \
  --verbose
```

La salida incluye:

- `CLIMDEX_PRECIP_data/DATA_NetCDF/<INDICE>/Yearly/<indice>_YYYY.nc`
- `CLIMDEX_PRECIP_data/DATA_NetCDF/<INDICE>/Trend/<INDICE>.nc`

El archivo de `Trend` contiene 10 variables como CDT:

- `slope`, `std.slope`, `t.value.slope`, `p.value.slope`
- `intercept`, `std.intercept`, `t.value.intercept`, `p.value.intercept`
- `R2`, `sigma`

### 20) CLIMDEX temperatura (TX/TN y derivados)

```bash
python3 scripts_descarga/cdt_climdex_tt_from_netcdf.py \
  --tx-dir /ruta/tmax_diario_netcdf \
  --tn-dir /ruta/tmin_diario_netcdf \
  --output-dir /ruta/salidas_cdt \
  --indices TXn,TXx,TNn,TNx,SU,ID,FD,TR,TX10p,TX90p,TN10p,TN90p,WSDI,CSDI,DTR,GSL \
  --base-start-year 1991 \
  --base-end-year 2020 \
  --base-min-year 20 \
  --base-window 5 \
  --upTX 25 --loTX 0 --upTN 20 --loTN 0 \
  --thresGSL 5 --dayGSL 6 \
  --trend-min-years 20 \
  --year-start 1981 --year-end 2024 \
  --verbose
```

Salida:

- `CLIMDEX_TEMP_data/DATA_NetCDF/<INDICE>/Yearly/<INDICE>_YYYY.nc`
- `CLIMDEX_TEMP_data/DATA_NetCDF/<INDICE>/Trend/<INDICE>.nc`

### 19) Convertir carpeta NetCDF a formato CDT Dataset

```bash
Rscript scripts_descarga/cdt_netcdf_to_cdtdataset.R \
  --input-dir /ruta/netcdf \
  --output-dir /ruta/salidas_cdt \
  --dataset-name PRECIP \
  --time-step daily \
  --varid precip \
  --lon-dim lon \
  --lat-dim lat \
  --chunk-size 100 \
  --chunk-fac 5 \
  --minhour 0 \
  --overwrite \
  --verbose
```

Salida esperada:

- `/ruta/salidas_cdt/PRECIP/PRECIP.rds`
- `/ruta/salidas_cdt/PRECIP/DATA/*.rds`

Notas:

- usa una fecha por archivo detectada desde el nombre (`YYYYMMDD`, `YYYYMM` o `YYYYMMD`)
- soporta recorte espacial opcional con `--bbox minlon,maxlon,minlat,maxlat`

## Wrappers por producto

Para simplificar, hay wrappers:

- `gpm_imerg_v07_early_daily_to_cdt.py`
- `gpm_imerg_v07_final_monthly_to_cdt.py`
- `chirps_v2_daily_to_cdt.py`
- `chirps_v2_monthly_to_cdt.py`
- `gsmap_clm_daily_to_cdt.py`
- `gsmap_clm_monthly_to_cdt.py`
- `mswep_v2_daily_to_cdt.py`
- `mswep_v2_monthly_to_cdt.py`
- `persiann_cdr_daily_to_cdt.py`
- `persiann_cdr_monthly_to_cdt.py`
- `persiann_ccs_daily_to_cdt.py`
- `persiann_ccs_monthly_to_cdt.py`

Ejemplo:

```bash
python3 scripts_descarga/gpm_imerg_v07_early_daily_to_cdt.py \
  --start 20260601 --end 20260605 \
  --minlon -75 --maxlon -70 --minlat -15 --maxlat -10 \
  --outdir . --verbose
```

## Estructura de salida

Se crea una carpeta por producto bajo `--outdir`:

- `GPM_L3_IMERG_V07_EARLY_daily/`
- `GPM_L3_IMERG_V07_FINAL_monthly/`
- `CHIRPSv2_daily/`
- `CHIRPSv2_monthly/`
- `GSMaP_CLM_daily/`
- `GSMaP_CLM_monthly/`
- `MSWEP_Past_v2.8_daily/`
- `MSWEP_Past_v2.8_monthly/`
- `PERSIANN-CDR_daily/`
- `PERSIANN-CDR_monthly/`
- `PERSIANN-CCS_daily/`
- `PERSIANN-CCS_monthly/`
- `ERA5_1Hr_Land/`

Con archivos en formato CDT:

- daily EARLY: `imerg_early_YYYYMMDD.nc`
- monthly FINAL: `imerg_final_YYYYMM.nc`
- daily CHIRPSv2: `chirps_YYYYMMDD.nc`
- monthly CHIRPSv2: `chirps_YYYYMM.nc`
- daily GSMaP CLM: `gsmap_clm_00z_YYYYMMDD.nc`
- monthly GSMaP CLM: `gsmap_clm_YYYYMM.nc`
- daily MSWEP v2: `mswep_v2.8_YYYYMMDD.nc`
- monthly MSWEP v2: `mswep_v2.8_YYYYMM.nc`
- daily PERSIANN-CDR: `persiann-cdr_YYYYMMDD.nc`
- monthly PERSIANN-CDR: `persiann-cdr_YYYYMM.nc`
- daily PERSIANN-CCS: `persiann-css_YYYYMMDD.nc`
- monthly PERSIANN-CCS: `persiann-css_YYYYMM.nc`
- ERA5-Land 1Hr: `<var>_YYYYMMDDHH.nc`

Cada NetCDF generado contiene:

- dimensiones `lon`, `lat`
- variable `precip` en `mm`
- nodata `-99`
- compresion interna netCDF4
