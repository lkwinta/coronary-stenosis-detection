# Raport analizy naczyń wieńcowych

**Obraz źródłowy:** 12.png
**Data analizy:** 2026-06-11 13:20:58
**Wymiary obrazu:** 512 x 512

## Obraz wejściowy

![Obraz wejściowy](images/12_input.png)

## Segmentacja

![Maska segmentacji](images/12_mask.png)

## Graf naczyń

![Graf naczyń](images/12_graph.png)

## Statystyki topologii

| Metryka | Wartość |
|---|---|
| Całkowita długość naczyń | 1011.9 px |
| Liczba gałęzi | 20 |
| Najdłuższa gałąź | 138.0 px |
| Najkrótsza gałąź | 11.0 px |
| Średnia krętość | 1.106 |
| Pokrycie naczyniowe | 2.47 % |
| Punkty końcowe | 13 |

## Szczegóły gałęzi

| ID | Długość (px) | Średnia Ø (px) | Min Ø (px) | Max Ø (px) |
|---|---|---|---|---|
| 0 | 51.7 | 14.4 | 4.0 | 20.0 |
| 1 | 11.0 | 17.2 | 16.0 | 18.9 |
| 2 | 23.9 | 10.1 | 7.2 | 18.9 |
| 3 | 90.5 | 9.3 | 6.0 | 17.9 |
| 4 | 45.9 | 9.2 | 6.0 | 17.9 |
| 5 | 87.0 | 5.5 | 2.8 | 8.0 |
| 6 | 12.9 | 10.4 | 8.0 | 14.0 |
| 7 | 53.3 | 4.8 | 2.0 | 8.0 |
| 8 | 33.7 | 6.9 | 2.8 | 14.0 |
| 9 | 68.3 | 5.4 | 2.0 | 10.0 |
| 10 | 32.8 | 7.3 | 6.0 | 10.0 |
| 11 | 22.0 | 6.3 | 2.8 | 10.0 |
| 12 | 36.2 | 3.9 | 2.0 | 5.7 |
| 13 | 16.5 | 5.6 | 2.0 | 10.0 |
| 14 | 67.1 | 8.3 | 6.0 | 11.3 |
| 15 | 18.9 | 6.1 | 4.0 | 10.2 |
| 16 | 27.6 | 7.0 | 2.0 | 12.0 |
| 17 | 93.2 | 5.3 | 2.0 | 10.0 |
| 18 | 138.0 | 5.5 | 2.0 | 11.3 |
| 19 | 81.5 | 5.9 | 4.0 | 11.3 |

## Klasyfikacja wierzchołków

- Pewne rozgałęzienia: 9
- Fałszywe rozgałęzienia: 0
- Nie-rozgałęzienia: 0

| ID | Środek (row, col) | Klasyfikacja | Confidence |
|---:|---|---|---:|
| 0 | (161.6, 135.4) | certain | - |
| 1 | (162.0, 146.0) | certain | - |
| 2 | (164.0, 238.0) | certain | - |
| 3 | (172.0, 228.6) | certain | - |
| 4 | (183.0, 128.0) | certain | - |
| 5 | (190.0, 178.0) | certain | - |
| 6 | (198.4, 142.0) | certain | - |
| 7 | (209.0, 113.0) | certain | - |
| 8 | (217.4, 231.4) | certain | - |

## XGBoost — predykcja stenozy na fragmentach

Poniższa wizualizacja pokazuje maskę, szkielet oraz fragmenty wskazane przez XGBoosta. Czerwone prostokąty oznaczają fragmenty dodatnie, a liczba przy fragmencie to P(stenosis).

![XGBoost overlay](images/12_xgboost_overlay.png)

| Metryka | Wartość |
|---|---:|
| Liczba fragmentów | 101 |
| Fragmenty dodatnie | 45 |
| Maksymalne prawdopodobieństwo | 0.8795 |
| Threshold | 0.4758 |

Top fragmenty według prawdopodobieństwa:

| Branch ID | Segment ID | Center X | Center Y | P(stenosis) | Pred |
|---:|---:|---:|---:|---:|---:|
| 3 | 7 | 214.2 | 170.3 | 0.8795 | 1 |
| 14 | 2 | 200.8 | 195.4 | 0.8512 | 1 |
| 11 | 0 | 131.0 | 186.7 | 0.8492 | 1 |
| 4 | 3 | 169.4 | 184.3 | 0.8445 | 1 |
| 3 | 2 | 168.9 | 159.0 | 0.8342 | 1 |
| 3 | 6 | 205.3 | 167.7 | 0.8329 | 1 |
| 11 | 1 | 136.8 | 194.4 | 0.8325 | 1 |
| 4 | 1 | 153.9 | 173.4 | 0.8324 | 1 |
| 3 | 1 | 158.9 | 159.0 | 0.8306 | 1 |
| 4 | 2 | 161.5 | 179.2 | 0.8301 | 1 |

---

*Raport wygenerowany automatycznie przez coronary_analysis*

