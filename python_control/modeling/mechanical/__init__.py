from .translation import (
    TranslationalImpedance,
    TranslationalComponent,
    Mass,
    Spring,
    Damper
)

from .rotation import (
    RotationalImpedance,
    RotationalComponent,
    Inertia,
    TorsionSpring,
    TorsionDamper,
    GearRatio
)

MechanicalComponent = TranslationalComponent | RotationalComponent
