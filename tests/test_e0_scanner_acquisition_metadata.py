import pandas as pd

from audit_e0_scanner_acquisition_metadata import parse_description, summarize_scanners


def test_parse_aperio_and_qptiff_description_fields():
    aperio = parse_description(
        "Aperio Leica GT450 v1.5.1 141074x89515 |Q = 91|AppMag = 40|MPP = 0.262407|ScannerType = GT450"
    )
    assert aperio["description_jpeg_quality"] == "91"
    assert aperio["description_app_mag"] == "40"
    assert aperio["description_mpp"] == "0.262407"
    assert aperio["description_scanner_type"] == "GT450"
    assert aperio["description_product"] == "Aperio Leica GT450 v1.5.1"

    qptiff = parse_description(
        "<AcquisitionSoftware>Fusion 2.2.0</AcquisitionSoftware>"
        "<Objective>10x</Objective><Name>H&amp;E brightfield 20x</Name>"
        "<PixelSizeMicrons>0.5</PixelSizeMicrons><Magnification>20</Magnification>"
        "<Compression>JPEG</Compression><JPEGQuality>70</JPEGQuality>"
    )
    assert qptiff["acquisition_software"] == "Fusion 2.2.0"
    assert qptiff["description_objective"] == "10x"
    assert qptiff["description_magnification"] == "20"
    assert qptiff["description_jpeg_quality"] == "70"


def test_scanner_summary_marks_consistent_mpp_and_preserves_missing_fields():
    scanners = ("at2", "gt450", "versa", "akoya", "s60", "s360")
    rows = []
    for scanner in scanners:
        for slide in range(109):
            rows.append(
                {
                    "scanner": scanner,
                    "slide_id": str(slide),
                    "native_path": f"/{scanner}/{slide}.svs",
                    "header_status": "pass",
                    "suffix": ".svs",
                    "openslide_vendor": "vendor",
                    "effective_mpp_x": 0.5 + slide * 1e-6,
                    "openslide_objective_power": 20.0,
                    "description_app_mag": "20",
                    "description_objective": "",
                    "description_magnification": "",
                    "tiff_compression": "JPEG",
                    "description_jpeg_quality": "70",
                    "tiff_photometric": "RGB",
                    "tile_width": 256,
                    "tile_height": 256,
                    "tiff_make": "",
                    "tiff_model": "",
                    "tiff_software": "software",
                    "acquisition_software": "",
                    "scan_profile_name": "",
                    "description_product": "product",
                }
            )
    result = summarize_scanners(pd.DataFrame(rows))
    assert result["scanner_slides"].eq(109).all()
    assert result["mpp_consistent_within_0_005"].all()
    assert result["objective_na_status"].str.contains("not present").all()
