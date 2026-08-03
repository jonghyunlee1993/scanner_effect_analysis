"""Public entry point for the generic E0 rigid native fallback builder."""

from build_e0_akoya_rigid_native_fallback import (
    MOVING_SCANNERS,
    fallback_location_checks,
    main,
    rigid_branch,
    truth,
)

__all__ = [
    "MOVING_SCANNERS",
    "fallback_location_checks",
    "main",
    "rigid_branch",
    "truth",
]


if __name__ == "__main__":
    main()
