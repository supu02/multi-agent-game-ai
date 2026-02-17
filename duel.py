import subprocess
import time

IP = "127.0.0.1"
PORT = 5555

# --- paramètres des deux IA ---
bots = [
    #("MeriemBot1", "starterkit/best_strat_2.py"),
    #("MeriemBot2", "starterkit/Sneha Basker Vampire.py"),
    ("MeriemBot2", "starterkit/sydney_5.py"),
    ("MeriemBot2", "starterkit/sydney_6.py"),
]

# --- Lancer les deux clients ---
processes = []

for name, script in bots:
    print(f"Lancement de {name} ...")
    p = subprocess.Popen(
        ["python", script, "--ip", IP, "--port", str(PORT), "--name", name],
    )
    processes.append(p)
    time.sleep(2)  # petite pause entre les deux connexions

print("Les deux IA sont lancées. La partie devrait démarrer automatiquement.")
print("Appuie sur Ctrl+C pour arrêter toutes les IA.")

try:
    # Attente de la fin (jusqu’à Ctrl+C)
    for p in processes:
        p.wait()
except KeyboardInterrupt:
    print("\nArrêt demandé. Fermeture des IA...")
    for p in processes:
        p.terminate()
