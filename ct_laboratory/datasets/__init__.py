# datasets/__init__.py
#
# NOTE: Only the dataset base classes (`core`) were merged from gmi. The concrete
# dataset implementations (mnist, medmnist, synthrad2023, kagglehub, sklearn) were
# intentionally NOT brought over — they pull external data dependencies and were
# excluded from the gmi merge. `core` is vendored so that `tasks` / `commands`
# can resolve their `GMI_Dataset` base-class references.
from .core import GMI_Dataset, GeneralPurposeDataset
