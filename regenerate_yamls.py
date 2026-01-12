"""
Script temporaire pour regénérer uniquement les YAMLs des méthodes problématiques
"""
import subprocess
import sys

# Générer uniquement les YAMLs (dry_run) pour SD3, en excluant SDEdit qui est correct
print("Regénération des YAMLs pour flowedit, irfds et ode_inv...")
cmd = [
    sys.executable, 
    "fig7/generate_edits.py",
    "--model", "sd3",
    "--dry_run",  # Ne pas exécuter run_script.py
]

result = subprocess.run(cmd, check=True, capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

print("\n✅ YAMLs regénérés avec succès!")
print("\nPour vérifier un des YAMLs générés, regardez:")
print("  scripts/fig7/exp_generated/sd3/flowedit/sd3_flowedit_011_Ours_cfg13.5.yaml")
print("  scripts/fig7/exp_generated/sd3/flowedit/sd3_flowedit_012_Ours_cfg16.5.yaml")
print("  scripts/fig7/exp_generated/sd3/flowedit/sd3_flowedit_013_Ours_cfg19.5.yaml")
