from pathlib import Path

from databricks.sdk.service.catalog import (
  VolumeInfo, 
)


def _volume_as_path(self: VolumeInfo) -> Path:
  return Path(f"/Volumes/{self.catalog_name}/{self.schema_name}/{self.name}")


# monkey patch
VolumeInfo.as_path = _volume_as_path
