# Raport analizy naczyń wieńcowych

**Obraz źródłowy:** {image_name}
**Data analizy:** {date}
**Wymiary obrazu:** {width} x {height}

## Obraz wejściowy

![Obraz wejściowy]({input_image_path})

## Segmentacja

![Maska segmentacji]({mask_image_path})

## Graf naczyń

![Graf naczyń]({graph_image_path})

## Statystyki topologii

| Metryka | Wartość |
|---|---|
| Całkowita długość naczyń | {total_vessel_length} px |
| Liczba gałęzi | {num_branches} |
| Najdłuższa gałąź | {longest_branch} px |
| Najkrótsza gałąź | {shortest_branch} px |
| Średnia krętość | {mean_tortuosity} |
| Pokrycie naczyniowe | {vessel_coverage} % |
| Punkty końcowe | {num_endpoints} |

## Szczegóły gałęzi

| ID | Długość (px) | Średnia Ø (px) | Min Ø (px) | Max Ø (px) |
|---|---|---|---|---|
{branch_rows}

---

*Raport wygenerowany automatycznie przez coronary_analysis*

