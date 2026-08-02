import multiprocessing
import time

DURATION_SECONDS = 90  # durée du test -> modifie ce chiffre si tu veux plus court/long


def burn_cpu(duration):
    end = time.time() + duration
    while time.time() < end:
        x = 2 ** 1000000  # calcul volontairement lourd, juste pour occuper le CPU


if __name__ == "__main__":
    print(f"[*] Démarrage du stress test pour {DURATION_SECONDS} secondes...")
    print("[*] Regarde Activity Monitor pendant ce temps pour surveiller CPU/température ressentie.")

    processes = [multiprocessing.Process(target=burn_cpu, args=(DURATION_SECONDS,)) for _ in range(4)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print("[*] Stress test terminé. Tous les processus sont arrêtés automatiquement.")