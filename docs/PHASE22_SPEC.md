# Phase 22: Excel intake (.xlsx / .xlsm)

Written **2026-11-17, before any code**. Additive: CSV, JSON, JSON Lines and Parquet behaviour must stay byte-identical (E0).

## Why
The product objective says "upload a dataset (CSV or Excel)". Intake (Phase 19) reads CSV/JSON/JSONL/Parquet only.

## Design (fixed now)
* Reader: `pandas.read_excel(engine="openpyxl")` behind `intake.load_table`, returning the same `(frame, info)` pair so everything
  downstream is format-blind. Cached cell values are read (formulas are not re-evaluated; a formula never calculated by Excel reads
  as missing, and that is reported).
* Detection: by extension (`.xlsx`, `.xlsm`) or by content (a ZIP holding `xl/workbook.xml`).
* **Sheets.** The first sheet that holds data is used by default. `info` carries `sheet`, `sheets` (all names) and a note when there is
  more than one. The caller may pick another sheet by name (`load_table(..., sheet=...)`); an unknown name is a clear error listing the names.
* **Header.** The first non-empty row is the header; fully empty leading rows/columns are skipped, fully empty rows are dropped (counted
  in a note); blank header cells become `Unnamed: n`, duplicate headers are made unique, as pandas does.
* **Types.** Numbers stay numeric, dates become datetime columns (the audit already leaves datetime columns out of analysis), text stays
  text; numbers stored as text are handled by the audit's own preparation.
* **Not supported, with a clear message:** legacy `.xls` ("save as .xlsx or CSV"), password-protected or corrupt workbooks, a workbook with
  no data. Charts, images, merged-cell formatting and macros are ignored (merged cells read as one value plus blanks, noted).
* Product: the Start page accepts the extensions; when a workbook has several sheets a sheet picker re-reads the stored upload
  (`POST /api/datasets/{id}/sheet`), no re-upload.

## Criteria
* **E0** every existing test suite passes unchanged.
* **E1** round trip: numbers, text, missing cells, dates and booleans survive an `.xlsx` written by pandas; the frame equals the source.
* **E2** sheets: default is the first sheet with data (an empty first sheet is skipped); a named sheet is read; an unknown name is an error
  that lists the sheets; `info["sheets"]` lists all.
* **E3** layout: leading blank rows/columns skipped, blank/duplicate headers made unique, empty rows dropped and counted.
* **E4** robustness: `.xls` gets the "save as .xlsx or CSV" message; random bytes named `.xlsx`, an empty workbook, and a truncated file are
  clear `ValueError`s, never a stack trace.
* **E5** end to end: an Excel upload through the backend gives the same audit result as the same data uploaded as CSV.

## Not claimed
Formula evaluation, legacy `.xls`, multi-sheet joins, pivot tables, or that Excel's own display formatting (e.g. a date shown as text) is preserved.

## Results
Measured 2026-11-17, `src/test_phase22.py` (6 tests), all pass.
* **E0** held: taskclean, selfbench and phases 14-21 pass unchanged.
* **E1-E5** pass. Two test mistakes were found and fixed while writing them, neither a reader bug: Excel stores whole numbers without a
  float/int distinction, so a whole-number float column reads back as integer (values equal, dtype differs; the test data now uses integers);
  and `openpyxl` ignores `cell(value=None)`, so a "blanked" row was never blank.
* Observed behaviour worth knowing: the default sheet is the first one with data, which can be a one-row cover sheet; the picker exists
  for that case. Leading blank rows count towards the "empty rows skipped" note.
* Verified in the browser with a three-sheet workbook (Readme, Customers, Small): upload reads "Readme", the picker switches to "Customers"
  (400 rows, target list updates); an unknown sheet returns a 400 that lists the sheets.
* Not run: files from real-world Excel exports (merged-heavy report layouts); only pandas/openpyxl-written workbooks were tested.
