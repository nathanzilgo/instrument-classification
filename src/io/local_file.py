from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class LocalFile(BaseModel):
    kind: str | None
    path: Path
    name: str

    def copy_with(
        self, kind: Optional[str] = None, path: Optional[Path] = None
    ) -> 'LocalFile':
        return LocalFile(
            kind=kind or self.kind,
            path=path or self.path,
        )
