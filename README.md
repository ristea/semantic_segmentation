# semantic_segmentation
SCCV Project

Pentru a putea rula proiectul va trebuie un env cu python3.x
Pentru asta recomandam urmatoarele comenzi:
python3 -m venv virt_env
. virt_env/bin/activate
pip install -r requirements.txt

Pentru a rula prima data aveti nevoie sa aveti deschis un server de visdom:
    python -m visdom.server

Dupa ce ati deschis serverul de visdom intrati pe pagina de local host indicata in terminal, acolo veti putea
vedea rezultatele si antreanarea.

Pentru a rula proiectul (antrenarea cu ultima arhitectura de retea):
    python main.py

IMPORTANT: in config.json aveti configurarile pentru retea si diverse alte lucruri, spre exemplu caile catre dataset.
Aveti nevoie sa configurati:
    "exp_path" = locul unde doriti sa fie salvate informatiile legate de antrenare, inclusiv modelele
    "data_path" = path-ul catre setul de date, atentie sa fie catre VOC2012!
Restul configurarilor depind de sistemul de pe care dvs lucrati.

