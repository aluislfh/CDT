#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  if (!requireNamespace("ncdf4", quietly = TRUE)) {
    stop("Package 'ncdf4' is required. Install with: install.packages('ncdf4')")
  }
})

print_help <- function() {
  cat(
"CDT NetCDF -> CDT Dataset converter

Usage:
  Rscript scripts_descarga/cdt_netcdf_to_cdtdataset.R \
    --input-dir /path/netcdf \
    --output-dir /path/out \
    --dataset-name PRECIP \
    --time-step daily \
    [--varid precip] \
    [--lon-dim lon] [--lat-dim lat] \
    [--chunk-size 100] [--chunk-fac 5] \
    [--minhour 0] \
    [--bbox minlon,maxlon,minlat,maxlat] \
    [--overwrite] [--verbose]

Notes:
  - Reads one NetCDF file per time step from --input-dir.
  - Auto-detects dates in file names (YYYYMMDD, YYYYMM, or YYYYMMD for dekad).
  - Output structure:
      <output-dir>/<dataset-name>/
        - <dataset-name>.rds
        - DATA/<chunk>.rds
",
  sep = ""
  )
}

parse_args <- function(args) {
  opts <- list(
    input_dir = NULL,
    output_dir = NULL,
    dataset_name = "CDTDATASET",
    time_step = "daily",
    varid = NULL,
    lon_dim = "lon",
    lat_dim = "lat",
    chunk_size = 100,
    chunk_fac = 5,
    minhour = "0",
    bbox = NULL,
    overwrite = FALSE,
    verbose = FALSE
  )

  i <- 1
  while (i <= length(args)) {
    a <- args[[i]]
    if (a == "--help" || a == "-h") {
      print_help()
      quit(save = "no", status = 0)
    }

    if (a == "--overwrite") {
      opts$overwrite <- TRUE
      i <- i + 1
      next
    }
    if (a == "--verbose") {
      opts$verbose <- TRUE
      i <- i + 1
      next
    }

    if (!startsWith(a, "--")) {
      stop(paste("Unknown positional arg:", a))
    }

    if (i == length(args)) {
      stop(paste("Missing value for", a))
    }
    val <- args[[i + 1]]

    key <- substring(a, 3)
    if (key == "input-dir") opts$input_dir <- val
    else if (key == "output-dir") opts$output_dir <- val
    else if (key == "dataset-name") opts$dataset_name <- val
    else if (key == "time-step") opts$time_step <- val
    else if (key == "varid") opts$varid <- val
    else if (key == "lon-dim") opts$lon_dim <- val
    else if (key == "lat-dim") opts$lat_dim <- val
    else if (key == "chunk-size") opts$chunk_size <- as.integer(val)
    else if (key == "chunk-fac") opts$chunk_fac <- as.integer(val)
    else if (key == "minhour") opts$minhour <- val
    else if (key == "bbox") opts$bbox <- val
    else stop(paste("Unknown option:", a))

    i <- i + 2
  }

  if (is.null(opts$input_dir) || is.null(opts$output_dir)) {
    stop("--input-dir and --output-dir are required. Use --help for usage.")
  }

  if (is.na(opts$chunk_size) || opts$chunk_size < 1) {
    stop("--chunk-size must be >= 1")
  }
  if (is.na(opts$chunk_fac) || opts$chunk_fac < 1) {
    stop("--chunk-fac must be >= 1")
  }

  opts
}

extract_date <- function(fname) {
  b <- basename(fname)

  m8 <- regmatches(b, gregexpr("(19|20)[0-9]{2}(0[1-9]|1[0-2])([0-2][0-9]|3[0-1])", b, perl = TRUE))[[1]]
  if (length(m8) > 0) return(m8[[length(m8)]])

  m7 <- regmatches(b, gregexpr("(19|20)[0-9]{2}(0[1-9]|1[0-2])[1-3]", b, perl = TRUE))[[1]]
  if (length(m7) > 0) return(m7[[length(m7)]])

  m6 <- regmatches(b, gregexpr("(19|20)[0-9]{2}(0[1-9]|1[0-2])", b, perl = TRUE))[[1]]
  if (length(m6) > 0) return(m6[[length(m6)]])

  NA_character_
}

detect_varid <- function(nc, lon_dim, lat_dim, varid = NULL) {
  if (!is.null(varid)) {
    if (is.null(nc$var[[varid]])) {
      stop(paste("Variable not found:", varid))
    }
    return(varid)
  }

  for (nm in names(nc$var)) {
    dnames <- sapply(nc$var[[nm]]$dim, `[[`, "name")
    if (lon_dim %in% dnames && lat_dim %in% dnames) {
      return(nm)
    }
  }

  stop("Could not auto-detect data variable containing lon/lat dims")
}

get_varinfo <- function(varobj, varid) {
  units <- varobj$units
  longname <- varobj$longname
  if (is.null(units) || is.na(units)) units <- ""
  if (is.null(longname) || is.na(longname) || nchar(longname) == 0) longname <- varid

  list(
    name = varid,
    units = units,
    longname = longname,
    prec = varobj$prec,
    missval = varobj$missval
  )
}

read_grid <- function(path, varid, lon_dim, lat_dim, lon_ref = NULL, lat_ref = NULL, bbox = NULL) {
  nc <- ncdf4::nc_open(path)
  on.exit(ncdf4::nc_close(nc))

  var <- nc$var[[varid]]
  dnames <- sapply(var$dim, `[[`, "name")
  dsizes <- sapply(var$dim, `[[`, "len")

  ilo <- match(lon_dim, dnames)
  ila <- match(lat_dim, dnames)
  if (is.na(ilo) || is.na(ila)) {
    stop(paste("lon/lat dims not found in", basename(path)))
  }

  lon <- var$dim[[ilo]]$vals
  lat <- var$dim[[ila]]$vals

  ord_lon <- order(lon)
  ord_lat <- order(lat)
  lon <- lon[ord_lon]
  lat <- lat[ord_lat]

  if (!is.null(bbox)) {
    ix <- lon >= bbox$minlon & lon <= bbox$maxlon
    iy <- lat >= bbox$minlat & lat <= bbox$maxlat
    if (!any(ix) || !any(iy)) {
      stop(paste("bbox outside grid for", basename(path)))
    }
    lon <- lon[ix]
    lat <- lat[iy]
    ord_lon <- ord_lon[ix]
    ord_lat <- ord_lat[iy]
  }

  raw <- ncdf4::ncvar_get(nc, varid)

  idx <- vector("list", length(dnames))
  for (i in seq_along(dnames)) {
    if (i == ilo) idx[[i]] <- ord_lon
    else if (i == ila) idx[[i]] <- ord_lat
    else idx[[i]] <- 1
  }

  sub <- do.call("[", c(list(raw), idx, list(drop = FALSE)))
  perm <- c(ilo, ila, setdiff(seq_along(dnames), c(ilo, ila)))
  arr <- aperm(sub, perm)

  if (length(dim(arr)) > 2) {
    extra_size <- prod(dim(arr)[-(1:2)])
    if (extra_size != 1) {
      stop(paste("Variable has extra dimensions > 1 in", basename(path), "(use files with one step/slice)."))
    }
    arr <- arr[, , 1, drop = FALSE]
  }

  mat <- matrix(as.numeric(arr), nrow = length(lon), ncol = length(lat))

  missval <- nc$var[[varid]]$missval
  if (!is.null(missval) && !all(is.na(missval))) {
    mat[mat %in% missval] <- NA_real_
  }

  if (!is.null(lon_ref) && !is.null(lat_ref)) {
    if (length(lon) != length(lon_ref) || length(lat) != length(lat_ref) ||
        any(abs(lon - lon_ref) > 1e-8) || any(abs(lat - lat_ref) > 1e-8)) {
      stop(paste("Grid mismatch in", basename(path)))
    }
  }

  list(lon = lon, lat = lat, mat = mat)
}

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))

  input_dir <- normalizePath(opts$input_dir, mustWork = TRUE)
  output_dir <- normalizePath(opts$output_dir, mustWork = FALSE)

  files <- list.files(input_dir, pattern = "\\.nc$", full.names = TRUE)
  if (length(files) == 0) stop("No .nc files found in input-dir")

  dates <- vapply(files, extract_date, character(1))
  keep <- !is.na(dates)
  files <- files[keep]
  dates <- dates[keep]
  if (length(files) == 0) stop("No date-like files found (YYYYMMDD, YYYYMMD, YYYYMM)")

  ord <- order(dates, files)
  files <- files[ord]
  dates <- dates[ord]

  if (any(duplicated(dates))) {
    stop("Duplicate dates detected in filenames")
  }

  if (!is.null(opts$bbox)) {
    parts <- strsplit(opts$bbox, ",", fixed = TRUE)[[1]]
    if (length(parts) != 4) stop("--bbox must be minlon,maxlon,minlat,maxlat")
    bbox <- list(
      minlon = as.numeric(parts[[1]]),
      maxlon = as.numeric(parts[[2]]),
      minlat = as.numeric(parts[[3]]),
      maxlat = as.numeric(parts[[4]])
    )
    if (any(is.na(unlist(bbox)))) stop("Invalid numeric values in --bbox")
  } else {
    bbox <- NULL
  }

  sample_nc <- ncdf4::nc_open(files[[1]])
  on.exit(ncdf4::nc_close(sample_nc), add = TRUE)
  varid <- detect_varid(sample_nc, opts$lon_dim, opts$lat_dim, opts$varid)
  varinfo <- get_varinfo(sample_nc$var[[varid]], varid)

  first <- read_grid(files[[1]], varid, opts$lon_dim, opts$lat_dim, bbox = bbox)
  lon <- first$lon
  lat <- first$lat

  nxy_chunksize <- round(sqrt(opts$chunk_size))
  if (nxy_chunksize < 1) nxy_chunksize <- 1

  seqlon <- seq_along(lon)
  seqlat <- seq_along(lat)
  seqcol <- cbind(id = seq(length(lon) * length(lat)), expand.grid(x = seqlon, y = seqlat))

  split_lon <- split(seqlon, ceiling(seqlon / nxy_chunksize))
  split_lat <- split(seqlat, ceiling(seqlat / nxy_chunksize))
  xgrid <- expand.grid(x = seq_along(split_lon), y = seq_along(split_lat))

  xarrg <- lapply(seq_len(nrow(xgrid)), function(j) {
    xs <- split_lon[[xgrid$x[j]]]
    ys <- split_lat[[xgrid$y[j]]]
    crd <- expand.grid(x = lon[xs], y = lat[ys])
    ids <- seqcol$id[(seqcol$x %in% xs) & (seqcol$y %in% ys)]
    list(coords = crd, id = ids, grp = rep(j, length(ids)))
  })

  col_idx <- lapply(xarrg, function(x) x$id)
  col_id <- do.call(c, col_idx)
  col_grp <- do.call(c, lapply(xarrg, function(x) x$grp))
  xy_exp <- do.call(rbind, lapply(xarrg, function(x) x$coords))
  col_order <- order(col_id)

  datarepo <- file.path(output_dir, opts$dataset_name)
  datadir <- file.path(datarepo, "DATA")
  datafile_idx <- file.path(datarepo, paste0(opts$dataset_name, ".rds"))

  if (dir.exists(datarepo)) {
    if (!opts$overwrite) {
      stop(paste("Output exists:", datarepo, "(use --overwrite)"))
    }
    unlink(datarepo, recursive = TRUE, force = TRUE)
  }

  dir.create(datadir, recursive = TRUE, showWarnings = FALSE)

  nfiles <- length(files)
  chunk_buffers <- lapply(seq_along(col_idx), function(i) vector("list", nfiles))

  if (opts$verbose) {
    cat("Reading", nfiles, "files and building chunk buffers...\n")
  }

  for (t in seq_len(nfiles)) {
    g <- read_grid(files[[t]], varid, opts$lon_dim, opts$lat_dim, lon_ref = lon, lat_ref = lat, bbox = bbox)
    vec <- c(g$mat)
    for (j in seq_along(col_idx)) {
      chunk_buffers[[j]][[t]] <- vec[col_idx[[j]]]
    }
    if (opts$verbose && (t %% 25 == 0 || t == nfiles)) {
      cat("  processed", t, "/", nfiles, "\n")
    }
  }

  if (opts$verbose) {
    cat("Writing chunk files...\n")
  }

  for (j in seq_along(chunk_buffers)) {
    mat <- do.call(rbind, chunk_buffers[[j]])
    con <- gzfile(file.path(datadir, paste0(j, ".rds")), compression = 7)
    open(con, "wb")
    saveRDS(mat, con)
    close(con)
  }

  cdt_index <- list(
    TimeStep = opts$time_step,
    minhour = opts$minhour,
    chunksize = nxy_chunksize * nxy_chunksize,
    chunkfac = opts$chunk_fac,
    coords = list(
      mat = list(x = lon, y = lat),
      df = xy_exp
    ),
    colInfo = list(
      id = col_id,
      index = col_grp,
      order = col_order
    ),
    varInfo = varinfo,
    dateInfo = list(
      date = dates,
      index = seq_along(dates)
    )
  )

  attr(cdt_index$coords$df, "out.attrs") <- NULL

  con <- gzfile(datafile_idx, compression = 6)
  open(con, "wb")
  saveRDS(cdt_index, con)
  close(con)

  cat("Done. CDT Dataset written to:\n")
  cat("  ", datarepo, "\n", sep = "")
}

main()
