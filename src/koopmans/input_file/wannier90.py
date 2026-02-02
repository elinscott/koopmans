"""Input parameters for ``wannier90.x`` calculations."""

from pydantic import create_model
from wannier90_input.models.latest import Wannier90Input

from koopmans.base import BaseModel

fields = {
    field_name: (field_info.annotation, field_info)
    for field_name, field_info in Wannier90Input.model_fields.items()
}

# A modified input class which removes input parameters (such as k-points) that are stored
# centrally in a workflow rather than as calculator parameters
Wannier90InputParameters = create_model(  # type: ignore[call-overload]
    "Wannier90InputParameters",
    __base__=BaseModel,
    **{k: v for k, v in fields.items() if k not in ["unit_cell_cart", "mp_grid", "kpoints"]},
)

# As above, but also excludes auto-generated keywords
RestrictedWannier90InputParameters = create_model(  # type: ignore[call-overload]
    "RestrictedWannier90InputParameters",
    __base__=BaseModel,
    **{
        k: v
        for k, v in fields.items()
        if k
        not in [
            "num_wann",
            "num_bands",
            "exclude_bands",
            "unit_cell_cart",
            "mp_grid",
            "kpoints",
            "projections",
        ]
    },
)
