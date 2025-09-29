"""Reusable toolkit for turning-movement count analysis.

This module centralises ingestion, adjustment, reporting, and plotting helpers
so project notebooks can stay lightweight while keeping the analytical logic in
one place.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, Sequence, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Constants and identifier normalisation helpers
# ---------------------------------------------------------------------------

DIRECTIONS: Sequence[str] = ("NB", "SB", "EB", "WB")
MOVEMENTS: Sequence[str] = ("Left", "Thru", "Right", "U-Turns")
DATASETS = ("combined", "passenger", "heavy")

APPROACH_ALIASES: Dict[str, str] = {
    "NB": "NB",
    "NORTHBOUND": "NB",
    "N": "NB",
    "SB": "SB",
    "SOUTHBOUND": "SB",
    "S": "SB",
    "EB": "EB",
    "EASTBOUND": "EB",
    "E": "EB",
    "WB": "WB",
    "WESTBOUND": "WB",
    "W": "WB",
}

MOVEMENT_ALIASES: Dict[str, str] = {
    "L": "L",
    "LEFT": "L",
    "T": "T",
    "THRU": "T",
    "THROUGH": "T",
    "R": "R",
    "RIGHT": "R",
    "U": "U",
    "UTURN": "U",
    "U-TURN": "U",
    "U-TURNS": "U",
}

# Peak period search windows used throughout the analysis helpers
PERIODS: Dict[str, Tuple[str, str]] = {
    "AM": ("06:00", "10:00"),
    "MD": ("10:00", "15:00"),
    "PM": ("15:00", "19:00"),
}


def _normalize_block_inputs(
    movements: Optional[Iterable[Any]] = None,
    approaches: Optional[Iterable[Any]] = None,
) -> Tuple[Set[str], Set[str]]:
    """Normalise user-supplied movement/approach identifiers."""

    mov_set: Set[str] = set()
    app_set: Set[str] = set()

    if approaches:
        for item in approaches:
            if item is None:
                continue
            key = str(item).strip().upper().replace("-", "").replace(" ", "")
            key = APPROACH_ALIASES.get(key, key)
            if key not in DIRECTIONS:
                raise ValueError(f"Unknown approach identifier: {item!r}")
            app_set.add(key)

    if movements:
        for item in movements:
            if item is None:
                continue
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                approach_raw = str(item[0]).strip().upper().replace("-", "").replace(" ", "")
                move_raw = str(item[1]).strip().upper().replace("-", "").replace(" ", "")
            else:
                token = str(item).strip().upper().replace("-", "_").replace(" ", "_")
                parts = [p for p in token.split("_") if p]
                if len(parts) < 2:
                    raise ValueError(
                        f"Movement identifier must include approach and movement: {item!r}"
                    )
                approach_raw, move_raw = parts[0], parts[1]
            approach = APPROACH_ALIASES.get(approach_raw, approach_raw)
            if approach not in DIRECTIONS:
                raise ValueError(f"Unknown approach identifier: {item!r}")
            move_key = MOVEMENT_ALIASES.get(move_raw, MOVEMENT_ALIASES.get(move_raw[:1], move_raw[:1]))
            move = move_key.upper()
            if move not in {"L", "T", "R", "U"}:
                raise ValueError(f"Unknown movement identifier: {item!r}")
            mov_set.add(f"{approach}_{move}")

    return mov_set, app_set


# ---------------------------------------------------------------------------
# Workbook ingestion helpers
# ---------------------------------------------------------------------------

def parse_direction_headers(header_row: Sequence[str]) -> Dict[str, str]:
    """Extract approach road names from the merged header row of the workbook."""

    mapping: Dict[str, str] = {}
    for value in header_row:
        if not isinstance(value, str):
            continue
        parts = [part.strip() for part in value.replace("\r", "").split("\n") if part.strip()]
        if len(parts) < 2:
            continue
        direction = parts[-1].lower()
        road = " ".join(parts[:-1])
        if direction == "northbound":
            mapping["NB"] = road
        elif direction == "southbound":
            mapping["SB"] = road
        elif direction == "eastbound":
            mapping["EB"] = road
        elif direction == "westbound":
            mapping["WB"] = road
    return mapping


def _rename_turn_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map successive Left/Thru/Right/U-Turn columns onto NB/SB/EB/WB."""

    rename: Dict[str, str] = {}
    data_cols = list(df.columns[1:])  # drop Start Time
    for approach, chunk in zip(DIRECTIONS, (data_cols[i : i + 4] for i in range(0, len(data_cols), 4))):
        for movement, old in zip(MOVEMENTS, chunk):
            code = movement.split("-")[0][0]
            rename[old] = f"{approach}_{code}"
    return df.rename(columns=rename)

@dataclass
class IntersectionData:
    """Container for passenger/heavy turning-movement data and metadata."""

    name: str
    count_date: pd.Timestamp
    northbound_road: str
    southbound_road: str
    eastbound_road: str
    westbound_road: str
    passenger: pd.DataFrame
    heavy: Optional[pd.DataFrame] = None
    blocked_movements: Set[str] = field(default_factory=set)
    blocked_approaches: Set[str] = field(default_factory=set)

    @classmethod
    def from_workbook(
        cls,
        source: Union[str, Path],
        *,
        name: str,
        northbound_road: Optional[str] = None,
        southbound_road: Optional[str] = None,
        eastbound_road: Optional[str] = None,
        westbound_road: Optional[str] = None,
        block_movements: Optional[Iterable[Any]] = None,
        block_approaches: Optional[Iterable[Any]] = None,
    ) -> "IntersectionData":
        """Load passenger/heavy sheets from the workbook and apply optional blocks."""

        source_path = Path(source)
        passenger = pd.read_excel(source_path, sheet_name="Passenger Vehicles", header=9)
        heavy = pd.read_excel(source_path, sheet_name="Heavy Trucks", header=9)

        passenger = _rename_turn_columns(passenger)
        heavy = _rename_turn_columns(heavy)

        header_sheet = pd.read_excel(source_path, sheet_name="Passenger Vehicles", header=None, nrows=1)
        road_names = parse_direction_headers(header_sheet.iloc[0])

        date_sheet = pd.read_excel(source_path, header=None)
        count_date = pd.to_datetime(date_sheet.iat[1, 2]).date()

        blocked_movements, blocked_approaches = _normalize_block_inputs(block_movements, block_approaches)

        instance = cls(
            name=name,
            count_date=pd.to_datetime(count_date),
            northbound_road=northbound_road or road_names.get("NB", ""),
            southbound_road=southbound_road or road_names.get("SB", ""),
            eastbound_road=eastbound_road or road_names.get("EB", ""),
            westbound_road=westbound_road or road_names.get("WB", ""),
            passenger=passenger,
            heavy=heavy,
            blocked_movements=blocked_movements,
            blocked_approaches=blocked_approaches,
        )
        instance._apply_blocking(drop=True)
        return instance

    def _apply_blocking_to_df(self, df: Optional[pd.DataFrame], drop: bool) -> Optional[pd.DataFrame]:
        if df is None:
            return None
        df = df.copy()
        blocked_cols: list[str] = []
        for col in df.columns:
            if col == "Start Time" or "_" not in col:
                continue
            approach, movement = col.split("_", 1)
            movement_code = movement.split("_", 1)[0]
            if approach in self.blocked_approaches or f"{approach}_{movement_code}" in self.blocked_movements:
                blocked_cols.append(col)
        if not blocked_cols:
            return df
        if drop:
            df = df.drop(columns=blocked_cols, errors="ignore")
        else:
            for col in blocked_cols:
                if col in df.columns:
                    df[col] = 0
        return df

    def _apply_blocking(self, drop: bool = True) -> None:
        if not self.blocked_movements and not self.blocked_approaches:
            return
        self.passenger = self._apply_blocking_to_df(self.passenger, drop)
        if self.heavy is not None:
            self.heavy = self._apply_blocking_to_df(self.heavy, drop)

    def block_movements(
        self,
        movements: Optional[Iterable[Any]] = None,
        approaches: Optional[Iterable[Any]] = None,
        drop: bool = True,
    ) -> "IntersectionData":
        mov_set, app_set = _normalize_block_inputs(movements, approaches)
        if mov_set:
            self.blocked_movements |= mov_set
        if app_set:
            self.blocked_approaches |= app_set
        if mov_set or app_set:
            self._apply_blocking(drop=drop)
        return self

    def _time_series_frame(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
        if "Start Time" not in frame.columns:
            raise ValueError("Expected a 'Start Time' column.")
        date_str = self.count_date.date().isoformat()
        ts = pd.to_datetime(date_str + " " + frame["Start Time"].astype(str))
        wide = frame.drop(columns=["Start Time"]).set_index(ts)
        return wide, ts

    @property
    def combined(self) -> pd.DataFrame:
        base = (
            self.passenger.set_index("Start Time")
            if "Start Time" in self.passenger.columns
            else self.passenger.copy()
        )
        if self.heavy is not None:
            hv = (
                self.heavy.set_index("Start Time")
                if "Start Time" in self.heavy.columns
                else self.heavy.copy()
            )
            hv = hv.reindex(base.index, fill_value=0)
            combined = base.add(hv, fill_value=0)
        else:
            combined = base
        return combined.reset_index()

    def _resolve_frame(self, dataset: Union[str, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(dataset, pd.DataFrame):
            return dataset.copy()
        if dataset == "combined":
            return self.combined.copy()
        if dataset == "passenger":
            return self.passenger.copy()
        if dataset == "heavy":
            if self.heavy is None:
                raise ValueError("No heavy-vehicle dataframe is available.")
            return self.heavy.copy()
        raise ValueError(f"Unknown dataset selection: {dataset!r}")

    def find_peaks_total_volume(
        self,
        dataset: Union[str, pd.DataFrame] = "combined",
        periods: Dict[str, Tuple[str, str]] = PERIODS,
        window: str = "60min",
    ) -> pd.DataFrame:
        frame = self._resolve_frame(dataset)
        wide, _ = self._time_series_frame(frame)
        wide["total"] = wide.sum(axis=1)

        if len(wide.index) >= 2:
            interval = (wide.index.to_series().diff().dropna().iloc[0])
        else:
            interval = pd.to_timedelta("15min")
        window_td = pd.to_timedelta(window)

        rows = []
        for label, (start_clock, end_clock) in periods.items():
            period_slice = wide.between_time(start_clock, end_clock, inclusive="left")
            if period_slice.empty:
                rows.append({"Period": label, "Peak Start": pd.NaT, "Peak End": pd.NaT, "Peak Total": float("nan")})
                continue
            totals = period_slice["total"].rolling(window_td).sum()
            peak_end = totals.idxmax()
            peak_start = peak_end - window_td + interval
            rows.append(
                {
                    "Period": label,
                    "Peak Start": peak_start,
                    "Peak End": peak_end,
                    "Peak Total": totals.loc[peak_end],
                }
            )
        return pd.DataFrame(rows)

    def peak_15min_by_movement(
        self,
        dataset: Union[str, pd.DataFrame] = "combined",
        periods: Dict[str, Tuple[str, str]] = PERIODS,
        window: str = "60min",
    ) -> pd.DataFrame:
        frame = self._resolve_frame(dataset)
        peaks = self.find_peaks_total_volume(dataset=frame, periods=periods, window=window)
        wide, _ = self._time_series_frame(frame)

        rows = []
        for _, peak in peaks.iterrows():
            label = peak["Period"] if "Period" in peak else peak.name
            peak_start = peak.get("Peak Start")
            peak_end = peak.get("Peak End")
            if pd.isna(peak_start) or pd.isna(peak_end):
                rows.append(pd.Series(dtype=float, name=label))
                continue
            window_slice = wide.loc[peak_start:peak_end]
            rows.append(window_slice.max().rename(label))
        return pd.DataFrame(rows)

    def peak_hour_totals_by_movement(
        self,
        dataset: Union[str, pd.DataFrame] = "combined",
        periods: Dict[str, Tuple[str, str]] = PERIODS,
        window: str = "60min",
        include_total: bool = True,
        peak_windows: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        frame = self._resolve_frame(dataset)
        if peak_windows is None:
            peak_windows = self.find_peaks_total_volume(dataset=frame, periods=periods, window=window)
        peak_windows = peak_windows.set_index("Period") if "Period" in peak_windows.columns else peak_windows
        wide, _ = self._time_series_frame(frame)

        rows = []
        for label, peak in peak_windows.iterrows():
            peak_start = peak.get("Peak Start")
            peak_end = peak.get("Peak End")
            if pd.isna(peak_start) or pd.isna(peak_end):
                rows.append(pd.Series(dtype=float, name=label))
                continue
            window_slice = wide.loc[peak_start:peak_end]
            totals = window_slice.sum().rename(label)
            if include_total:
                totals["Total"] = peak.get("Peak Total", window_slice.values.sum())
            rows.append(totals)
        return pd.DataFrame(rows)

    def peak_hour_factor_for_peak_period(
        self,
        dataset: Union[str, pd.DataFrame] = "combined",
        periods: Dict[str, Tuple[str, str]] = PERIODS,
        window: str = "60min",
        include_approach: bool = False,
        include_movement: bool = False,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        frame = self._resolve_frame(dataset)
        peaks = self.find_peaks_total_volume(dataset=frame, periods=periods, window=window)
        peak_windows = peaks.set_index("Period") if "Period" in peaks.columns else peaks
        wide, _ = self._time_series_frame(frame)

        if len(wide.index) >= 2:
            interval = (wide.index.to_series().diff().dropna().iloc[0])
        else:
            interval = pd.to_timedelta("15min")
        window_td = pd.to_timedelta(window)

        def _phf(series: pd.Series) -> float:
            series = series.dropna()
            if series.empty:
                return float("nan")
            max_bin = float(series.max())
            if max_bin <= 0:
                return float("nan")
            bins_per_window = int(round(window_td / interval))
            return float(series.sum()) / (bins_per_window * max_bin)

        intersection_rows = []
        approach_rows = []
        movement_rows = []

        for label, peak in peak_windows.iterrows():
            peak_start = peak.get("Peak Start")
            peak_end = peak.get("Peak End")
            if pd.isna(peak_start) or pd.isna(peak_end):
                intersection_rows.append(pd.Series({"PHF": float("nan")}, name=label))
                if include_approach:
                    approach_rows.append(pd.Series(name=label))
                if include_movement:
                    movement_rows.append(pd.Series(name=label))
                continue
            window_slice = wide.loc[peak_start:peak_end]
            intersection_rows.append(pd.Series({"PHF": _phf(window_slice.sum(axis=1))}, name=label))

            if include_approach:
                data = {}
                for approach in DIRECTIONS:
                    cols = [c for c in window_slice.columns if c.startswith(f"{approach}_")]
                    if cols:
                        data[approach] = _phf(window_slice[cols].sum(axis=1))
                approach_rows.append(pd.Series(data, name=label))

            if include_movement:
                data = {col: _phf(window_slice[col]) for col in window_slice.columns}
                movement_rows.append(pd.Series(data, name=label))

        intersection_df = pd.DataFrame(intersection_rows)
        if not include_approach and not include_movement:
            return intersection_df

        result: Dict[str, pd.DataFrame] = {"intersection": intersection_df}
        if include_approach:
            result["approach"] = pd.DataFrame(approach_rows)
        if include_movement:
            result["movement"] = pd.DataFrame(movement_rows)
        return result
    def apply_adjustments(
        self,
        *,
        factor: Optional[float] = None,
        movements: Optional[Sequence[str]] = None,
        copy: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Scale volumes by a factor and round up, preserving total ceilings."""

        def _apply(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if df is None:
                return None
            target = df.copy() if copy else df
            if factor is None:
                return target
            cols = [c for c in target.columns if c != "Start Time"]
            if movements is not None:
                allowed = set(movements)
                cols = [c for c in cols if c in allowed]
            if not cols:
                return target
            scaled = target.loc[:, cols].astype(float) * factor
            floors = np.floor(scaled)
            frac = scaled - floors
            shortfall = (np.ceil(scaled.sum(axis=0)) - floors.sum(axis=0)).astype(int)
            adjusted = floors.astype(int)
            for col in cols:
                extra = int(shortfall[col])
                if extra <= 0:
                    continue
                bump_rows = frac[col].sort_values(ascending=False).index[:extra]
                adjusted.loc[bump_rows, col] += 1
            target.loc[:, cols] = adjusted.astype(int)
            return target

        passenger_df = _apply(self.passenger)
        heavy_df = _apply(self.heavy)

        if copy:
            return passenger_df, heavy_df

        if passenger_df is not None:
            self.passenger = passenger_df
        if heavy_df is not None and self.heavy is not None:
            self.heavy = heavy_df
        return self.passenger, self.heavy

    def adjusted_frames(
        self,
        *,
        factor: Optional[float] = None,
        movements: Optional[Sequence[str]] = None,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """Return adjusted passenger/heavy copies without mutating the originals."""

        passenger_df, heavy_df = self.apply_adjustments(factor=factor, movements=movements, copy=True)
        return {"passenger_adjusted": passenger_df, "heavy_adjusted": heavy_df}

    def estimate_approach_aadt(
        self,
        *,
        dataset: Union[str, pd.DataFrame] = "combined",
        expansion_factor: Union[float, Dict[str, float]] = 1.0,
        seasonal_factor: float = 1.0,
        growth_factor: float = 1.0,
    ) -> pd.DataFrame:
        frame = self._resolve_frame(dataset)
        values = frame.drop(columns=["Start Time"]).astype(float)

        approach_totals: Dict[str, float] = {}
        for approach in DIRECTIONS:
            cols = [c for c in values.columns if c.startswith(f"{approach}_")]
            if cols:
                approach_totals[approach] = float(values[cols].sum().sum())

        def factor_for(approach: str) -> float:
            if isinstance(expansion_factor, dict):
                return float(expansion_factor.get(approach, expansion_factor.get("default", 1.0)))
            return float(expansion_factor)

        rows = []
        observed = pd.Series({approach: vol for approach, vol in approach_totals.items()}, name="Observed Approach Volume")
        rows.append(observed)

        expanded = {}
        seasonal = {}
        design = {}
        for approach, observed_vol in approach_totals.items():
            expansion = factor_for(approach)
            expanded[approach] = math.ceil(observed_vol * expansion)
            seasonal[approach] = math.ceil(observed_vol * expansion * seasonal_factor)
            design[approach] = math.ceil(observed_vol * expansion * seasonal_factor * growth_factor)

        rows.append(pd.Series(expanded, name="Expanded Volume"))
        if seasonal_factor != 1.0:
            rows.append(pd.Series(seasonal, name="Seasonally Adjusted AADT"))
        if growth_factor != 1.0:
            rows.append(pd.Series(design, name="Design Year AADT"))

        return pd.DataFrame(rows)
    def generate_turning_movement_report(
        self,
        *,
        dataset: Union[str, pd.DataFrame] = "combined",
        periods: Dict[str, Tuple[str, str]] = PERIODS,
        window: str = "60min",
        seasonal_factor: float = 1.0,
        growth_factor: float = 1.0,
        expansion_factor: Union[float, Dict[str, float]] = 1.0,
    ) -> Dict[str, Any]:
        frame = self._resolve_frame(dataset)
        period_windows = self.find_peaks_total_volume(dataset=frame, periods=periods, window=window)
        peak_windows_idx = period_windows.set_index("Period") if "Period" in period_windows.columns else period_windows.copy()

        peak15 = self.peak_15min_by_movement(dataset=frame, periods=periods, window=window)
        totals = self.peak_hour_totals_by_movement(
            dataset=frame,
            periods=periods,
            window=window,
            include_total=False,
            peak_windows=peak_windows_idx,
        )

        if self.heavy is not None:
            heavy_totals = self.peak_hour_totals_by_movement(
                dataset="heavy",
                periods=periods,
                window=window,
                include_total=False,
                peak_windows=peak_windows_idx,
            )
        else:
            heavy_totals = pd.DataFrame(0.0, index=totals.index, columns=totals.columns)

        phf_tables = self.peak_hour_factor_for_peak_period(
            dataset=frame,
            periods=periods,
            window=window,
            include_approach=True,
            include_movement=True,
        )
        movement_phf = phf_tables.get("movement", pd.DataFrame()) if isinstance(phf_tables, dict) else pd.DataFrame()

        base_cols = sorted({*totals.columns, *peak15.columns, *heavy_totals.columns, *movement_phf.columns})
        ordered_cols: list[str] = []
        for approach in DIRECTIONS:
            for mv_code in ("L", "T", "R", "U"):
                key = f"{approach}_{mv_code}"
                if key in base_cols:
                    ordered_cols.append(key)

        approach_names = {
            "NB": ("Northbound", self.northbound_road),
            "SB": ("Southbound", self.southbound_road),
            "EB": ("Eastbound", self.eastbound_road),
            "WB": ("Westbound", self.westbound_road),
        }
        movement_labels = {"L": "Left", "T": "Through", "R": "Right", "U": "U-Turn"}

        def format_columns(columns: Sequence[str]) -> pd.MultiIndex:
            tuples = []
            for col in columns:
                approach, mv = col.split("_", 1)
                direction_label, road = approach_names.get(approach, (approach, ""))
                top = f"{direction_label}\n{road}" if road else direction_label
                tuples.append((top, movement_labels.get(mv, mv)))
            return pd.MultiIndex.from_tuples(tuples, names=["Approach", "Movement"])

        tables: Dict[str, pd.DataFrame] = {}
        season_mult = float(seasonal_factor or 1.0)
        growth_mult = float(growth_factor or 1.0)

        for label in peak_windows_idx.index:
            total_series = totals.loc[label].reindex(ordered_cols).astype(float) if label in totals.index else pd.Series(0.0, index=ordered_cols)
            peak15_series = peak15.loc[label].reindex(ordered_cols).astype(float) if label in peak15.index else pd.Series(0.0, index=ordered_cols)
            heavy_series = heavy_totals.loc[label].reindex(ordered_cols).astype(float) if label in heavy_totals.index else pd.Series(0.0, index=ordered_cols)
            phf_series = movement_phf.loc[label].reindex(ordered_cols) if label in movement_phf.index else pd.Series(np.nan, index=ordered_cols)

            percent_series = heavy_series.divide(total_series.replace(0, np.nan))

            rows = [
                pd.Series(peak15_series, name="Peak 15-Min Volume"),
                pd.Series(total_series, name="Counted Total Volume"),
                pd.Series(heavy_series, name="Counted Heavy Vehicles"),
                pd.Series(percent_series, name="Heavy Vehicles Proportion"),
                pd.Series(phf_series, name="Peak Hour Factor"),
            ]

            if season_mult != 1.0:
                seasonal = np.ceil(total_series * season_mult)
                rows.append(pd.Series(seasonal, name="Seasonal Adjusted Volume"))
            if growth_mult != 1.0:
                projected = np.ceil(total_series * season_mult * growth_mult)
                rows.append(pd.Series(projected, name="Projected Volume"))

            table = pd.DataFrame(rows)
            table.columns = format_columns(ordered_cols)
            tables[label] = table

        approach_summary = self.estimate_approach_aadt(
            dataset=dataset,
            expansion_factor=expansion_factor,
            seasonal_factor=season_mult,
            growth_factor=growth_mult,
        )

        metadata = {
            "intersection": self.name,
            "count_date": str(self.count_date.date()),
            "northbound_road": self.northbound_road,
            "southbound_road": self.southbound_road,
            "eastbound_road": self.eastbound_road,
            "westbound_road": self.westbound_road,
            "seasonal_factor": season_mult,
            "growth_factor": growth_mult,
            "expansion_factor": expansion_factor,
        }

        return {
            "metadata": metadata,
            "period_windows": period_windows[["Peak Start", "Peak End", "Peak Total"]],
            "period_tables": tables,
            "approach_summary": approach_summary,
            "phf_tables": phf_tables,
            "intersection_obj": self,
        }

@dataclass
class IntersectionConfig:
    name: str
    source: Union[str, Path]
    seasonal_factor: float = 1.0
    growth_factor: float = 1.0
    expansion_factor: Union[float, Dict[str, float]] = 1.0
    northbound_road: Optional[str] = None
    southbound_road: Optional[str] = None
    eastbound_road: Optional[str] = None
    westbound_road: Optional[str] = None
    block_movements: Optional[Iterable[Any]] = None
    block_approaches: Optional[Iterable[Any]] = None


class BatchRunner:
    """Helper to load multiple intersections and produce reports in bulk."""

    def __init__(self, configs: Iterable[IntersectionConfig]):
        self.configs = list(configs)
        self.intersections: Dict[str, IntersectionData] = {}
        self.reports: Dict[str, Dict[str, Any]] = {}

    def load_all(self) -> Dict[str, IntersectionData]:
        for cfg in self.configs:
            ix = IntersectionData.from_workbook(
                cfg.source,
                name=cfg.name,
                northbound_road=cfg.northbound_road,
                southbound_road=cfg.southbound_road,
                eastbound_road=cfg.eastbound_road,
                westbound_road=cfg.westbound_road,
                block_movements=cfg.block_movements,
                block_approaches=cfg.block_approaches,
            )
            self.intersections[cfg.name] = ix
        return self.intersections

    def build_reports(self, **kwargs: Any) -> Dict[str, Dict[str, Any]]:
        if not self.intersections:
            self.load_all()
        for cfg in self.configs:
            ix = self.intersections[cfg.name]
            payload = ix.generate_turning_movement_report(
                seasonal_factor=cfg.seasonal_factor,
                growth_factor=cfg.growth_factor,
                expansion_factor=cfg.expansion_factor,
                **kwargs,
            )
            payload.setdefault("intersection_obj", ix)
            self.reports[cfg.name] = payload
        return self.reports


def export_report_to_excel(
    report: Dict[str, Any],
    out_path: Union[str, Path],
    *,
    include_metadata: bool = True,
    freeze_panes: Tuple[int, int] = (2, 2),
    decimal_rows: Tuple[str, ...] = ("Heavy Vehicles Proportion", "Peak Hour Factor"),
) -> None:
    """Persist a report payload to an Excel workbook with safe row-level formatting."""

    def _normalize(s: Any) -> str:
        return str(s).strip().lower()

    def _find_row_idx(df: pd.DataFrame, label: str) -> Optional[int]:
        """Return 0-based *data row* index in df.index for label (case/space-insensitive)."""
        target = _normalize(label)
        norm_idx = [_normalize(x) for x in df.index]
        for i, v in enumerate(norm_idx):
            if v == target:
                return i
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        wb = writer.book

        # Formats
        header_fmt  = wb.add_format({"bold": True, "bg_color": "#DDEBF7"})
        int_fmt     = wb.add_format({"num_format": "#,##0", "align": "right"})
        dec_fmt     = wb.add_format({"num_format": "0.000", "align": "right"})
        # If you prefer percent for the proportion row:
        # pct_fmt   = wb.add_format({"num_format": "0.0%", "align": "right"})

        # --- Metadata sheet (optional)
        if include_metadata:
            pd.Series(report["metadata"], name="Value").to_frame().to_excel(writer, sheet_name="Metadata")
            sh = writer.sheets["Metadata"]
            sh.set_column("A:A", 28, header_fmt)
            sh.set_column("B:B", 40)

        # --- Period tables
        for period, table in report["period_tables"].items():
            sheet_name = period.replace(" ", "_")
            table.to_excel(writer, sheet_name=sheet_name)
            sh = writer.sheets[sheet_name]

            # Header (top row with column titles written by pandas)
            sh.freeze_panes(*freeze_panes)
            sh.set_row(0, None, header_fmt)

            # Column widths only (NO default format so row formats can take effect)
            sh.set_column(0, 0, 28)                                # row labels column
            sh.set_column(1, table.shape[1], 14)                   # numeric columns

            # Where data starts: pandas writes (column header rows) + 1 for index label row
            header_rows = table.columns.nlevels + 1
            total_rows = len(table.index)

            # 1) Default: set **integer format** for all data rows (volumes etc.)
            for r in range(total_rows):
                sh.set_row(header_rows + r, None, int_fmt)

            # 2) Override: set **decimal format** on special rows (if present)
            for lbl in decimal_rows:
                ridx = _find_row_idx(table, lbl)
                if ridx is None:
                    # Not all periods must have every special row; skip gracefully
                    # You can log/print if helpful:
                    # print(f"[{sheet_name}] Row not found: {lbl}")
                    continue
                sh.set_row(header_rows + ridx, None, dec_fmt)
                # If you need *percent* formatting for the proportion row instead:
                # if _normalize(lbl) == _normalize("Heavy Vehicles Proportion"):
                #     sh.set_row(header_rows + ridx, None, pct_fmt)

        # --- Other sheets
        report["period_windows"].to_excel(writer, sheet_name="Peak_Windows")
        writer.sheets["Peak_Windows"].set_column("A:C", 22)

        report["approach_summary"].to_excel(writer, sheet_name="Approach_AADT")
        sh = writer.sheets["Approach_AADT"]
        sh.set_column("A:A", 28, header_fmt)
        sh.set_column("B:Z", 16, int_fmt)

    print(f"Wrote {out_path}")

def plot_peak_stacked_bars(
    report: Dict[str, Any],
    period: str = "AM",
    movements: Optional[Sequence[str]] = None,
    colors=None,
    ax=None,
):
    table = report["period_tables"][period]
    series = table.loc["Counted Total Volume"]
    if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels != 2:
        raise ValueError("Expected peak tables with (approach, movement) multi-index columns.")
    df = series.unstack(level=1).fillna(0)
    if movements:
        df = df[[m for m in df.columns if m in movements]]
    ax = df.plot(kind="bar", stacked=True, color=colors, figsize=(10, 5), ax=ax)
    ax.set_ylabel("Vehicles in Peak Hour")
    ax.set_title(f"{period} Peak Hour Stacked Movements")
    ax.set_xlabel("")
    ax.legend(title="Movement", loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    return ax


def plot_movement_time_series(
    source: Union[Dict[str, Any], IntersectionData],
    dataset: Union[str, pd.DataFrame] = "combined",
    movements: Optional[Sequence[str]] = None,
    ax=None,
):
    if isinstance(source, IntersectionData):
        intersection = source
    elif isinstance(source, dict) and "intersection_obj" in source:
        intersection = source["intersection_obj"]
    else:
        raise ValueError("Provide an IntersectionData instance or a report containing 'intersection_obj'.")

    df = intersection._resolve_frame(dataset)
    date_str = intersection.count_date.date().isoformat()
    ts = pd.to_datetime(date_str + " " + df["Start Time"].astype(str))
    wide = df.drop(columns=["Start Time"]).set_index(ts)
    if movements:
        wide = wide[movements]
    ax = wide.plot(figsize=(12, 5), ax=ax)
    ax.set_ylabel("Vehicles per 15 min")
    ax.set_xlabel("Time")
    ax.set_title(f"15-minute Volumes ({dataset})")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    return ax


def plot_heavy_percentage_heatmap(
    report: Dict[str, Any],
    periods: Sequence[str] = ("AM", "MD", "PM"),
    cmap: str = "Blues",
):
    """Render a heatmap of heavy-vehicle percentages by period and approach."""

    records: list[pd.DataFrame] = []
    for period in periods:
        table = report["period_tables"][period]
        series = table.loc["Heavy Vehicles Proportion"]
        if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels != 2:
            raise ValueError("Expected peak tables with (approach, movement) columns.")
        pct = series.unstack(level=1).fillna(0.0)
        pct.index = pd.Index(pct.index, name="Approach")
        pct = pct.reset_index()
        pct["Period"] = period
        records.append(pct)

    if not records:
        raise ValueError("No periods supplied for heatmap.")

    heat = pd.concat(records, ignore_index=True)
    heat = heat.set_index(["Period", "Approach"]).sort_index()

    # Convert from decimals to percentages
    numeric = heat.select_dtypes(include=[float, int]) * 100.0

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(
        numeric,
        annot=True,
        fmt=".1f",  # e.g., 18.0
        cmap=cmap,
        cbar_kws={"label": "% Heavy"},
        ax=ax,
    )
    ax.set_xlabel("Movement")
    ax.set_ylabel("Peak Period / Approach")
    ax.set_yticklabels(
        [f"{period} / {approach}" for period, approach in numeric.index],
        rotation=0,
    )
    plt.tight_layout()
    plt.title("Heavy Vehicle Percentage (%)")
    return ax


def plot_phf_bars(
    source: Union[Dict[str, Any], IntersectionData],
    level: str = "approach",
    periods: Sequence[str] = ("AM", "MD", "PM"),
):
    if isinstance(source, IntersectionData):
        phf_data = source.peak_hour_factor_for_peak_period(include_approach=True, include_movement=True)
    elif isinstance(source, dict):
        if "phf_tables" in source:
            phf_data = source["phf_tables"]
        elif "intersection_obj" in source:
            phf_data = source["intersection_obj"].peak_hour_factor_for_peak_period(include_approach=True, include_movement=True)
        else:
            raise ValueError("Report dict must contain 'phf_tables' or 'intersection_obj'.")
    else:
        raise TypeError("Pass an IntersectionData instance or a report dictionary.")

    phf_frame = phf_data[level] if isinstance(phf_data, dict) else phf_data
    phf_frame = phf_frame.loc[list(periods)]
    phf_frame.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("PHF")
    plt.title(f"Peak Hour Factor by {level.capitalize()}")
    plt.tight_layout()


def plot_adjusted_volume_comparison(
    report: Dict[str, Any],
    period: str = "AM",
    include_movements: Optional[Sequence[str]] = None,
):
    table = report["period_tables"][period]
    base = table.loc["Counted Total Volume"]
    seasonal = table.loc["Seasonal Adjusted Volume"] if "Seasonal Adjusted Volume" in table.index else None
    projected = table.loc["Projected Volume"] if "Projected Volume" in table.index else None

    cols = base.index if include_movements is None else [c for c in base.index if any(mv in c for mv in include_movements)]
    x = range(len(cols))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, base[cols], width, label="Counted")
    offset = width
    if seasonal is not None:
        ax.bar([i + offset for i in x], seasonal[cols], width, label="Seasonal Adj.")
        offset += width
    if projected is not None:
        ax.bar([i + offset for i in x], projected[cols], width, label="Projected")

    ax.set_xticks([i + width for i in x])
    ax.set_xticklabels([f"{a}\n{m}" for a, m in cols], rotation=90)
    ax.set_ylabel("Vehicles (Peak Hour)")
    ax.set_title(f"{period} Peak Hour Adjustment Comparison")
    ax.legend()
    plt.tight_layout()
    return ax
