"""
Script de test rapide pour vérifier que les 3 méthodes sont bien différentes
"""
import yaml

# Charger et afficher les méthodes de chaque YAML
yamls = [
    ("FloweEdit cfg13.5", "scripts/fig7/exp_generated/sd3/flowedit/sd3_flowedit_011_Ours_cfg13.5.yaml"),
    ("ODE Inv cfg13.5", "scripts/fig7/exp_generated/sd3/ode_inv/sd3_ode_inv_007_ODEInv_cfg13.5.yaml"),
    ("iRFDS", "scripts/fig7/exp_generated/sd3/irfds/sd3_irfds_010_iRFDS_official.yaml"),
]

print("=" * 80)
print("VÉRIFICATION DES CONFIGURATIONS YAML")
print("=" * 80)

for name, path in yamls:
    with open(path, 'r') as f:
        config = yaml.safe_load(f)[0]
    
    print(f"\n{name}:")
    print(f"  method: {config.get('method', 'NOT SET')}")
    print(f"  tar_guidance_scale: {config.get('tar_guidance_scale', 'NOT SET')}")
    print(f"  n_max: {config.get('n_max', 'NOT SET')}")

print("\n" + "=" * 80)
print("✅ Si vous voyez 'method: flowedit/ode_inv/irfds', c'est bon!")
print("=" * 80)
