"""Pydantic response models — these define the public API contract."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Segment(BaseModel):
    start: float = Field(..., description="Window start in seconds")
    end: float = Field(..., description="Window end in seconds")
    spoof_probability: float = Field(..., ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    verdict: str = Field(..., description='"HUMAN" or "FAKE"')
    confidence: int = Field(..., ge=0, le=100, description="Confidence in the predicted class")
    spoof_probability: float = Field(..., ge=0.0, le=1.0, description="Mean over all windows")
    max_segment_probability: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(..., ge=0.0, le=1.0, description="Decision boundary used")
    duration_sec: float = Field(..., description="Duration of the uploaded clip")
    analysed_sec: float = Field(..., description="Duration actually analysed (after silence trim)")
    backend: str = Field(..., description='"ml" or "heuristic"')
    model: str = Field(..., description="Model id used, or 'heuristic-fallback'")
    filename: str
    segments: List[Segment] = []
    reasons: List[str] = Field(default_factory=list, description="Heuristic backend only")
    warnings: List[str] = Field(default_factory=list, description="Caveats about this specific clip")
    spectrogram_png_base64: Optional[str] = None
    note: str
    processing_ms: int


class HealthResponse(BaseModel):
    status: str
    backend: str
    model: str
    mode: str
    device: str
    threshold: float
    ffmpeg: bool
    ml_load_error: Optional[str] = None
    version: str


class ErrorResponse(BaseModel):
    detail: str
