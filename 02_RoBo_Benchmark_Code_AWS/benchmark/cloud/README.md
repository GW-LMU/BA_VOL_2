# Benchmark auf AWS rechnen (SQS-Queue + EC2-Worker)

Dieser Ordner ist eine Kopie von `01_RoBo_Benchmark_Code`. **Der Benchmark-Code selbst
(`preprocessing/`, `main_processing/`, `postprocessing/`, `main.py`) ist unveraendert.**
Neu ist nur der Ordner `cloud/`, der denselben Code auf AWS ausfuehrt:

- Experimentplan: identisch zu `main.py` (`create_experiment_table` + `remove_duplicate_nominal_runs`)
- Rechnen: unveraendertes `run_single_experiment` aus `experiment_runner.py`
- Ergebnis: dieselbe CSV (gleiche Spalten) + Excel ueber `export_excel`

Jedes Experiment wird als eigene Nachricht in eine **SQS-Queue** gestellt. **EC2-Instanzen**
holen sich die Experimente ab, rechnen sie (ein Prozess pro CPU-Kern) und legen das
Ergebnis jedes Experiments als CSV in **S3** ab. Ist die Queue leer, beenden sich die
Instanzen selbst. Lokal werden die Ergebnisse zu einer CSV + Excel zusammengefuehrt.

```
submit.py ──► SQS-Queue ──► worker.py (EC2: n Instanzen × n Kerne) ──► S3 (tasks/*.csv) ──► collect.py ──► results/raw/<RUN>.csv + .xlsx
                   └─ nach 3 Fehlversuchen ──► Dead-Letter-Queue
```

| Datei | Zweck |
|---|---|
| `aws_settings.py` | Region, Queue-, Bucket- und Laufname (per Umgebungsvariablen) |
| `setup_infra.py` | legt einmalig Bucket, Queue, DLQ, IAM-Rolle an |
| `submit.py` | Experimentplan erstellen, Code nach S3 hochladen, Experimente in die Queue |
| `launch_workers.py` | EC2-Worker starten / anzeigen / beenden |
| `ec2_user_data.sh` | Startskript der Instanzen (Python, Abhaengigkeiten, Worker) |
| `worker.py` | arbeitet die Queue ab (laeuft auf EC2, aber auch lokal) |
| `collect.py` | Fortschritt anzeigen, Ergebnisse herunterladen und zusammenfuehren |
| `requirements-aws.txt` | zusaetzliche Abhaengigkeit `boto3` |

## Einmalige Vorbereitung

1. AWS-Konto, AWS CLI installieren und `aws configure` ausfuehren (Access Key eines
   Benutzers mit Rechten fuer S3, SQS, IAM, EC2).
2. Lokal zusaetzlich boto3 installieren:

```bash
pip install -r cloud/requirements-aws.txt
```

3. Bucket-Namen festlegen (weltweit eindeutig, nur Kleinbuchstaben/Ziffern/Bindestriche)
   und Infrastruktur anlegen. Alle Befehle im Ordner `benchmark/`. PowerShell:

```powershell
$env:ROBO_BUCKET = "robo-benchmark-<dein-name>"
```

```bash
python -m cloud.setup_infra
```

Standardregion ist `eu-central-1` (Frankfurt), aenderbar mit `ROBO_AWS_REGION`.
Die Umgebungsvariablen gelten nur im aktuellen Terminal – in einem neuen Terminal erneut setzen.

## Ablauf eines Laufs

`ROBO_RUN_NAME` trennt verschiedene Laeufe im Bucket (Standard: `results`). Fuer jeden
neuen Plan (z. B. nach Aenderung von `config.py`) einen neuen Namen waehlen:

```powershell
$env:ROBO_RUN_NAME = "lauf_01"
```

1. **Plan pruefen** (sendet nichts):

```bash
python -m cloud.submit --dry-run
```

2. **Experimente einstellen** (laedt auch den aktuellen Code inkl. `config.py` hoch):

```bash
python -m cloud.submit
```

3. **Worker starten**, z. B. 4 Spot-Instanzen mit je 16 Kernen = 64 parallele Experimente:

```bash
python -m cloud.launch_workers --count 4 --instance-type c7i.4xlarge --spot
```

4. **Fortschritt** ansehen:

```bash
python -m cloud.collect --status
```

5. **Ergebnisse** zusammenfuehren → `results/raw/<RUN_NAME>.csv` und `.xlsx`:

```bash
python -m cloud.collect
```

## Empfohlener erster Test

Erst klein starten und pruefen, ob die Instanz sauber durchlaeuft:

```bash
python -m cloud.submit --max-experiments 3
```

```bash
python -m cloud.launch_workers --count 1 --instance-type c7i.large
```

Nach ca. 5–10 min (Einrichtung) sollte `python -m cloud.collect --status` die ersten fertigen
Experimente zeigen. Fuer die Fehlersuche mit `--key-name <schluesselpaar>` starten, per SSH
verbinden und `/var/log/robo-worker.log` ansehen.

## Wichtig zu wissen

- **Code-Aenderungen aus 01 uebernehmen:** Den Ordner `benchmark/` aus `01_...` hierher
  kopieren (alles ausser `cloud/`), dann `submit` erneut ausfuehren – erst dann rechnen
  die Worker den neuen Stand. `cloud/` haengt nur von `run_single_experiment`,
  `RESULT_COLUMNS`, `create_experiment_table`, `remove_duplicate_nominal_runs` und
  `export_excel` ab.
- **Reproduzierbarkeit:** Die Seeds haengen nur von `MASTER_SEED` und der Wiederholung ab,
  nicht von Maschine oder Reihenfolge. Getestet: lokal (`run_experiments`) und ueber die
  Queue gerechnete Experimente liefern identische Werte (bis auf `iter_time_s`).
- **Fortsetzen:** `submit` stellt nur Experimente ein, deren Ergebnis noch nicht in S3
  liegt. Nach einem Abbruch also einfach erneut `submit` + `launch_workers`.
  Nicht `submit` aufrufen, waehrend noch Nachrichten in der Queue warten – sonst werden
  diese Experimente doppelt gerechnet (das Ergebnis wird nur ueberschrieben, kostet aber Zeit).
- **Fehler:** Ein fehlgeschlagenes Experiment wird bis zu 3-mal versucht, der Traceback liegt
  in `s3://<bucket>/<RUN_NAME>/errors/`. Danach landet es in der Queue
  `robo-benchmark-experiments-failed` (in der AWS-Konsole einsehbar).
- **Spot-Instanzen** sind meist 60–70 % guenstiger. Wird eine unterbrochen, kommt das
  laufende Experiment nach max. 15 min automatisch wieder in die Queue.
- **Kosten im Blick behalten:** Die Instanzen beenden sich nach 10 min leerer Queue selbst
  (`--idle-minutes`). Zur Sicherheit: `python -m cloud.launch_workers --list` bzw.
  `--terminate` beendet alle Worker sofort. Einen Budget-Alarm in der AWS-Konsole
  (Billing → Budgets) einrichten. Neue AWS-Konten haben oft ein niedriges vCPU-Limit fuer
  EC2/Spot – ggf. unter Service Quotas eine Erhoehung beantragen.
- **Lokal mitrechnen:** Der Worker laeuft auch auf dem eigenen Rechner und arbeitet dieselbe
  Queue ab: `python -m cloud.worker --processes 4 --idle-minutes 1`.
