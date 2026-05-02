"""
Run all 10 geospatial analysis projects in sequence.
"""
import subprocess
import sys
import os
import time

projects = [
    ("01-urban-heat-island", "Urban Heat Island Mapping"),
    ("02-flood-risk", "Flood Risk & Early Warning"),
    ("03-healthcare-access", "Healthcare Accessibility"),
    ("04-food-desert", "Food Desert Identification"),
    ("05-deforestation", "Deforestation Monitoring"),
    ("06-road-safety", "Road Safety Hotspot Analysis"),
    ("07-air-quality", "Air Quality & Pollution Mapping"),
    ("08-crop-yield", "Crop Yield Optimisation"),
    ("09-disaster-response", "Disaster Response Routing"),
    ("10-transit-gap", "Public Transit Gap Analysis"),
]

results = []
base_dir = os.path.dirname(os.path.abspath(__file__))

print("=" * 70)
print("  GEOSPATIAL DATA ANALYSIS PROJECTS — FULL RUN")
print("=" * 70)

for folder, name in projects:
    project_dir = os.path.join(base_dir, folder)
    print(f"\n[{folder}] Running: {name}")
    print("-" * 50)
    start = time.time()
    result = subprocess.run(
        [sys.executable, "analysis.py"],
        cwd=project_dir, capture_output=False, text=True
    )
    elapsed = time.time() - start
    status = "✓ PASSED" if result.returncode == 0 else "✗ FAILED"
    results.append((folder, name, status, f"{elapsed:.1f}s"))
    print(f"\n  {status} in {elapsed:.1f}s")

print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
for folder, name, status, duration in results:
    print(f"  {status}  [{folder}]  {name}  ({duration})")
passed = sum(1 for *_, s, _ in results if "PASSED" in s)
print(f"\n  {passed}/{len(projects)} projects completed successfully.")
print("=" * 70)
