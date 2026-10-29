from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Series:
    uid: str
    modality: str
    description: str
    frame_of_reference_uid: str | None
    slice_thickness: float | None
    n_files: int
    paths: list[str] = field(default_factory=list, repr=False)


@dataclass
class Study:
    uid: str
    patient_id: str
    series: list[Series] = field(default_factory=list)

    def by_modality(self, modality: str) -> list[Series]:
        return [s for s in self.series if s.modality == modality]


@dataclass
class Patient:
    patient_id: str
    studies: list[Study] = field(default_factory=list)
