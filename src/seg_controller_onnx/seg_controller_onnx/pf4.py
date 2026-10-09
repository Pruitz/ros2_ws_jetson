import numpy as np
import cv2



def _line(model, y):
    """Valuta il modello lineare x(y) = m*y + q."""
    return model["m"] * y + model["q"]



def _irls(ys, xs, iters=4, sigma=18.0):
    """Fitta una retta con IRLS, riducendo il peso dei punti anomali."""
    ys = np.asarray(ys, np.float32)
    xs = np.asarray(xs, np.float32)
    if len(ys) < 2:
        return None, None


    weights = np.ones(len(ys), np.float32)
    coeff = np.polyfit(ys, xs, 1, w=weights) #coef[0] = m, coef[1] = q


    for _ in range(iters):
        residuals = xs - (coeff[0] * ys + coeff[1]) # residuals = observed line's x points - predicted line's x points (points far from the observed edge (segmentation) have large residuals)
        weights = 1.0 / (1.0 + (residuals / max(sigma, 1e-6)) ** 2) # v = 1 / (1 + (residual / sigma)^2), if residual is large, weight is small and vice versa. So the outliner points have small influence on the fit.
        coeff = np.polyfit(ys, xs, 1, w=weights) # new fit with updated weights


    model = {"m": float(coeff[0]), "q": float(coeff[1])}
    residuals = np.abs(xs - _line(model, ys))
    return model, residuals



def path_finder(
    pred_mask, #input of the function, is a 2D binarymatrix (predicted mask) with shape 224x224
    path_class=1, # pixel with value 1 representing the filare, 0 all others
    min_row_ratio=0.30, # top portion of the image excluded from analysis
    n_strips=24, # number of horizontal strips used for tracking
    min_component_area=500, # remove small components
    min_interval_width=8, # remove intervals narrower than this width
    max_interval_gap=30, #join intervals with a gap smaller than this value
    border_margin=2, # margin to recognize intervals cut at the edges
    min_edge_jump=28.0, # minimum allowed jump between edges of consecutive strips
    robust_iters=4, # iterations of the robust IRLS fit
    robust_sigma=18.0, # determines how much the IRLS fit downweights outliers
    local_band_half_width=5, # semi-width of the band used for edge support
    seed_bottom_strips=6, # bottom strips considered for the initial seed
    min_edge_support=0.20, # reserved for future use, currently represents the minimum edge support required for a valid path
    max_gap_strips=3, # maximum number of consecutive strips without tracking
    ok_conf_threshold=0.60, # state threshold for considering the path as "ok"
    degraded_conf_threshold=0.38, # state threshold for considering the path as "degraded"
):
    """Trova e traccia il centro di un filare nella maschera segmentata.


    La funzione seleziona intervalli nelle strisce orizzontali, li segue dal
    basso verso l'alto e fitta separatamente i bordi sinistro e destro.


    Args:
        pred_mask: Maschera 2D con le classi predette dal modello.
        path_class: Valore della classe che rappresenta il filare.
        min_row_ratio: Quota superiore esclusa dall'analisi.
        n_strips: Numero di strisce orizzontali usate per il tracking.
        min_component_area: Area minima per mantenere una componente connessa.
        min_interval_width: Larghezza minima di un intervallo candidato.
        max_interval_gap: Massimo vuoto orizzontale unito a un intervallo.
        border_margin: Margine per riconoscere intervalli tagliati ai bordi.
        min_edge_jump: Salto minimo consentito tra bordi di strisce consecutive.
        robust_iters: Iterazioni del fit robusto IRLS.
        robust_sigma: Scala dei residui per ridurre il peso degli outlier.
        local_band_half_width: Semi-larghezza della fascia usata per il supporto.
        seed_bottom_strips: Strisce inferiori candidate per il seed iniziale.
        min_edge_support: Parametro riservato per una futura soglia minima
        di supporto dei bordi. Non viene ancora usato nel codice.
        max_gap_strips: Numero massimo di strisce consecutive senza tracking.
        ok_conf_threshold: Soglia minima di confidence per lo stato ``ok``.
        degraded_conf_threshold: Sotto questa soglia il risultato e' ``lost``.


    Returns:
        Una tupla con la maschera della centerline e un dizionario diagnostico.
        La maschera e' vuota quando il tracking o la validazione falliscono.
    """
    pred_mask = np.asarray(pred_mask)
    H, W = pred_mask.shape # 224x224
    output = np.zeros((H, W), np.float32) # initialized output mask for the centerline (224x224)


    # 1. ROI inferiore e binarizzazione: esclude la parte alta dell'immagine
    # e mantiene solo i pixel della classe path.
    y0 = int(H * min_row_ratio) # y0 = 67 (for H = 224 and min_row_ratio = 0.30), so the ROI starts from row 67 from the top
    path = (pred_mask[y0:] == path_class).astype(np.uint8) # binary mask of the path class within the ROI, 157x224 (0-66 ingnored rows,67-223 analyzed in the ROI)


    # 2. Morfologia e componenti: chiude/apre piccoli buchi e rimuove le
    # componenti con area inferiore alla soglia.


    # Opening removes small foreground noise; closing fills small holes and reconnects narrow gaps.
    path = cv2.morphologyEx(
        path, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    path = cv2.morphologyEx(
        path, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
    )


    # removes small components based on the minimum component area threshold
    n, labels, stats, _ = cv2.connectedComponentsWithStats(path, 8)
    clean = np.zeros_like(path)
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
            clean[labels == label] = 1
    path = clean


    # after 2. path only contains sufficiently large connected components that can rappresent the real path


    # 3. Suddivisione in strip: le strip sono indicizzate dal basso verso l'alto.
    # Con pred_mask.shape == (224, 224) e min_row_ratio == 0.30:
    # y0 = 67, path.shape == (157, 224), quindi roi_h == 157.    
    roi_h = path.shape[0]
    strip_h = max(1, roi_h // n_strips) # for roi_h=157 and n_strips=24, strip_h=6
    strips = []


    # vertical limits for each strip within the ROI
    for idx in range(n_strips):
        # idx == 0 corrisponde alla strip più bassa.
        # ys: bordo superiore della strip, incluso.
        # ye: bordo inferiore della strip, escluso.
        ys = max(0, roi_h - (idx + 1) * strip_h)
        ye = min(roi_h, roi_h - idx * strip_h)


        if ye <= ys:
            continue


        # Esempio con roi_h=157, n_strips=24, strip_h=6:
        # idx=0 -> path[151:157], la strip più bassa.
        # idx=1 -> path[145:151].
        # idx=2 -> path[139:145].
        strip = path[ys:ye] 


        # Count path pixels in each strip column.
        profile = strip.sum(axis=0)
        # Mark columns containing at least one path pixel.
        # Example: profile = [0, 0, 3, 6, 4, 0] -> binary = [0, 0, 1, 1, 1, 0]
        binary = (profile > 0).astype(np.int8)
        # Detect background-to-path (+1) and path-to-background (-1) transitions.
        # Example: padded = [0, 0, 0, 1, 1, 1, 0, 0] -> diff = [0, 0, 1, 0, 0, -1, 0]
        diff = np.diff(np.pad(binary, (1, 1)))
        # Get path interval boundaries: example -> start = 2, end = 4.
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1


        # 4. Estrazione e fusione: ricava gli intervalli della strip e fonde
        # quelli separati da piccoli buchi nella maschera.
        spans = []
        for start, end in zip(starts, ends): # zip example: starts = [20, 76, 160] ends = [60, 140, 190] list(zip(starts, ends)) = [(20, 60), (76, 140), (160, 190)]
            if spans and start - spans[-1][1] - 1 <= max_interval_gap: #an empty list is False, true if there is not empty; spans[-1][1] is the end of the last span span=[(start,end),(start,end)...]
                spans[-1] = (spans[-1][0], end)
            else:
                spans.append((start, end))


        # Scarta intervalli troppo stretti e quelli che coprono quasi tutta la
        # strip senza sufficiente densita'.
        intervals = []
        for start, end in spans: #analizza ogni intervallo trovato nella strip (ciclo annidato)
            if end - start + 1 < min_interval_width: # elimina intervalli troppo stretti
                continue
            if start <= border_margin and end >= W - 1 - border_margin: #elimina intervalli che coprono quasi tutta la strip senza sufficiente densita'
                if np.mean(strip) < 0.7:
                    continue


            # 5. Supporto locale dei bordi: misura separatamente quanto sono presenti i
            # pixel della classe path intorno al bordo sinistro e destro.
            band = max(2, local_band_half_width)
            left = strip[:, max(0, start - band):min(W, start + band + 1)] #estrae una piccola regione rettangolare intorno al bordo sinistro, esempio: start=50 band=5 -> left = strip[:, 45:56] (11 colonne)
            right = strip[:, max(0, end - band):min(W, end + band + 1)] #estrae una piccola regione rettangolare intorno al bordo destro, esempio: end=100 band=5 -> right = strip[:, 95:106] (11 colonne)
            left_support = float(np.mean(left)) # num pixel banda sinistra/total pixel banda sinistra (value=[0:1])
            right_support = float(np.mean(right)) # num pixel banda destra/total pixel banda destra (value=[0:1])
            
            # the check on the minimum support can be added here if needed, e.g., skip intervals with very low support


            intervals.append({
                "start": float(start),
                "end": float(end),
                "width": float(end - start + 1),
                "left_support": left_support,
                "right_support": right_support,
                "touches_left": start <= border_margin,
                "touches_right": end >= W - 1 - border_margin,
            })


        strips.append({
            "idx": idx,
            "y": y0 + 0.5 * (ys + ye),
            "ys": ys,
            "ye": ye,
            "intervals": intervals,
        })


    if not strips:
        return output, {"status": "lost", "reason": "no_strips"}


    # 6. Scelta del seed: tra le strip inferiori seleziona l'intervallo con il
    # miglior supporto minimo dei due bordi.
    seeds = []
    for strip in strips[:min(seed_bottom_strips, len(strips))]: # consider only the bottom 6 (or fewer if not enough) strips for seed selection
        for interval in strip["intervals"]:
            support = min(interval["left_support"], interval["right_support"])
            seeds.append((support, interval, strip))


    if not seeds:
        return output, {"status": "lost", "reason": "no_seed_candidate"}

    # scelta del seed finale
    # prima massimizza item[0] = support
    # poi, a parità di support, massimizza item[1]["width"]
    # esempio:
    # seed_A = (0.45, width=80)
    # seed_B = (0.42, width=120)
    # seed_C = (0.45, width=95)
    # vince seed_C
    _, current, seed_strip = max(
        seeds, key=lambda item: (item[0], item[1]["width"])
    )
    # Scegli il seed: l'intervallo con supporto minimo più alto (e, a parità
    # di supporto, il più largo). current = dizionario dell'intervallo seed
    # (start, end, width, left_support, right_support, touches_left/right).
    # seed_strip = dizionario della strip che lo contiene (idx, y, ys, ye, intervals).


    # 7. Tracking bottom-up: inizializza dal seed (current) e conserva i due bordi,
    # aggiornando la posizione e la larghezza tra strip consecutive.
    points = [{  #points è la lista che conterra tutti i punti del corridoio, uno per strip valida
        "y": seed_strip["y"],
        "strip_idx": seed_strip["idx"],
        "x0": current["start"],
        "x1": current["end"],
        "left_support": current["left_support"],
        "right_support": current["right_support"],
        "touches_left": current["touches_left"],
        "touches_right": current["touches_right"],
    }]

    #Variabili di stato per il tracking
    # Servono per tenere traccia dello stato "corrente" del corridoio mentre sali di strip
    prev_left = current["start"] #posizione del bordo sinistro nell’ultima strip accettata
    prev_right = current["end"] #posizione del bordo destro nell’ultima strip accettata
    prev_width = current["width"] #larghezza dell’intervallo nell’ultima strip accettata
    missed = 0 ##contatore di strip consecutive senza match (resetta a 0 ogni volta che trovi un intervallo valido);
    max_missed = 0 # valore massimo raggiunto da missed durante tutto il tracking (usato dopo per le metriche di qualità).


    # Nelle strip successive associa l'intervallo piu' vicino; consente un
    # numero limitato di strip senza candidati compatibili.
    for i in range(seed_strip["idx"] + 1, len(strips)): #inizio del tracking dalla strip successiva a quella che contiene il seed
        jump_limit = max(min_edge_jump, 0.5 * prev_width)

        #Costruzione delle scelte possibili (choices)
        choices = []
        for interval in strips[i]["intervals"]: #guardo tutti gli intervalli nella strip corrente i
            #Calcolo del “salto” laterale
            jump = max(
                abs(interval["start"] - prev_left),
                abs(interval["end"] - prev_right),
            )
            #se il salto laterale è entro il limite, considero l'intervallo come scelta possibile e calcolo il costo
            if jump <= jump_limit:
                cost = jump + 0.4 * abs(interval["width"] - prev_width) #una variazione di 20 px di larghezza costa come un salto laterale di 8 px;
                choices.append((cost, interval)) #Accumulo delle scelte per questa strip (choices è una lista di tuple, [(costo, intervallo),..]. dove ogni interval è un dizionario con start, end, width, left_support, right_support, touches_*.

        #Se choices è vuota, significa che nessun intervallo nella strip corrente è abbastanza vicino al corridoio precedente.
        if not choices:
            missed += 1 #Incrementi il contatore di strip perse di fila
            max_missed = max(max_missed, missed) #Aggiorni il massimo storico di gap
            if missed > max_gap_strips: #Se il numero di strip perse consecutive supera il limite massimo, interrompo il tracking
                break
            continue # se non ci sono intervalli buoni in una strip ma non abbiamo raggiunto il limite massimo di strip perse, salta il resto del ciclo for e passa alla strip successiva
        #Se supero max_gap_strips il corridoio è considerato perso;
        #Se no, continui con la strip successiva, mantenendo invariati prev_left, prev_right, prev_width.

        # Scelta dell’intervallo migliore
        # Questo è il passo greedy: prendi il candidato localmente migliore, senza guardare le strip future.
        _, current = min(choices, key=lambda item: item[0]) #sceglie la tupla con costo minimo
        strip = strips[i]
        #Aggiunta del punto alla catena
        points.append({
            "y": strip["y"],
            "strip_idx": strip["idx"],
            "x0": current["start"],
            "x1": current["end"],
            "left_support": current["left_support"],
            "right_support": current["right_support"],
            "touches_left": current["touches_left"],
            "touches_right": current["touches_right"],
        })
        prev_left = current["start"]
        prev_right = current["end"]
        prev_width = current["width"]
        missed = 0 #Resetti il contatore di strip perse consecutive dopo aver trovato un intervallo valido

    # se il numero di punti trovati è inferiore a 3, considera il tracking fallito
    if len(points) < 3:
        return output, {"status": "lost", "reason": "tracking_failed"}


    # 8. Fit IRLS e verifica geometrica: prepara campioni indipendenti per i
    # bordi non tagliati, poi stima le due rette in modo robusto agli outlier.

    #salvo che non toccano i bordi per avere punti affidabili per il fit dei bordi del path
    left = [p for p in points if not p["touches_left"]]
    right = [p for p in points if not p["touches_right"]]

    # se ho meno di due punti non posso costruire la retta dei bordi, quindi considero il tracking fallito
    if len(left) < 2 or len(right) < 2:
        return output, {"status": "lost", "reason": "insufficient_edge_points"}

    # Preparazione degli array per il fit ()
    yl = np.array([p["y"] for p in left], np.float32)
    xl = np.array([p["x0"] for p in left], np.float32)
    yr = np.array([p["y"] for p in right], np.float32)
    xr = np.array([p["x1"] for p in right], np.float32)

    # Fit IRLS dei bordi. 
    # Modello: {"m", "q"} per x(y) = m*y + q.
    # Residui: array di |x_obs - x_fit| per ciascun punto (yl, xl) e (yr, xr).
    left_model, left_residuals = _irls(yl, xl, robust_iters, robust_sigma)
    right_model, right_residuals = _irls(yr, xr, robust_iters, robust_sigma)
    if left_model is None or right_model is None:
        return output, {"status": "lost", "reason": "fit_failed"}


    #Verifica geometrica del modello (rette)
    
    # Verifica l'ordine dei bordi e misura larghezza e residui del fit.
    # Campioni 16 valori di y tra il minimo e il massimo delle osservazioni dei due bordi;
    # Per ogni y, calcoli la larghezza del corridoio dal modello:
    y_eval = np.linspace(min(yl.min(), yr.min()), max(yl.max(), yr.max()), 16)
    widths = _line(right_model, y_eval) - _line(left_model, y_eval)
    width_ok = bool(np.all(widths > 0))  # True se il bordo destro è sempre a destra del sinistro
    width = float(np.median(widths))
    
    # Calcolo di width_error: usa i punti osservati per rilevare artefatti locali,
    # non il modello (che è già smoothato).
    # Usa la derivata seconda per essere insensibili alla prospettiva lineare,
    # ma sensibili a cambi bruschi (gradini).
    widths_obs = np.array([p["x1"] - p["x0"] for p in points], dtype=np.float32)
    
    # Derivata seconda approssimata (cambio del cambio di larghezza)
    if len(widths_obs) >= 3:
        diff1 = np.diff(widths_obs)
        diff2 = np.diff(diff1)
        # Usa il 90° percentile per catturare i casi peggiori (outlier)
        width_error_90 = float(np.percentile(np.abs(diff2), 90))
        # Normalizza per la larghezza tipica e il numero di punti
        width_error = width_error_90 / max(np.median(np.abs(widths_obs)), 1.0)
    else:
        width_error = 0.0
    # width_error basso = corridoio coerente (prospettiva regolare o costante);
    # width_error alto = presenza di artefatti locali o instabilità.


    #calcolo metriche di qualità in un unico punteggio "confidence"
    residual = float(np.mean(np.r_[left_residuals, right_residuals])) #Residuo medio del fit, misura quanto i punti osservati si discostano dalle rette stimate
    support = float(np.mean([
        *[p["left_support"] for p in left],
        *[p["right_support"] for p in right],
    ]))
    tracking = len(points) / max(len(strips), 1)
    fit_score = np.clip(1.0 - residual / max(robust_sigma, 1e-6), 0.0, 1.0)
    width_error_scale=0.30
    width_score = np.clip(1.0 - width_error / max(width_error_scale, 1e-6),0.0,1.0,)    
    gap_score = np.clip(1.0 - max_missed / max(max_gap_strips, 1), 0.0, 1.0)

    confidence = float(
        0.25 * tracking      # % di strip tracciate
        + 0.20 * support     # densità media ai bordi
        + 0.25 * fit_score   # coerenza geometrica (fit IRLS)
        + 0.20 * width_score # regolarità larghezza
        + 0.10 * gap_score   # continuità tracking
    )


    # Classifica il risultato combinando validita' geometrica, residuo e
    # confidence; un fit non valido viene marcato come perso.
    if not width_ok or residual > 2.0 * robust_sigma:
        status = "lost"
    elif confidence < degraded_conf_threshold:
        status = "lost"
    elif confidence < ok_conf_threshold:
        status = "degraded"
    else:
        status = "ok"

    # uscita anticipata se lo status è "lost"
    if status == "lost":
        roi_binary = np.zeros((H, W), np.uint8)
        roi_binary[y0:] = path * 255

        return output, {
            "status": status,
            "confidence": confidence,
            "mean_residual": residual,
            "support": support,
            "tracking": float(tracking),
            "width": width,
            "width_error": width_error,
            "max_gap": max_missed,
            "left_edge_model": left_model,
            "right_edge_model": right_model,
            "centerline_model": None,
            "line_start_y": None,
            "line_end_y": None,
            "roi_binary": roi_binary,
        }


    # 9. Centerline, rendering e dilatazione: fa la media dei modelli dei
    # bordi e rasterizza il tracciato nel tratto seguito fino al fondo.
    center_model = {
        "m": 0.5 * (left_model["m"] + right_model["m"]),
        "q": 0.5 * (left_model["q"] + right_model["q"]),
    }
    last_tracked_strip = strips[points[-1]["strip_idx"]] if "strip_idx" in points[-1] else None
    y_start = (
        y0 + last_tracked_strip["ys"]
        if last_tracked_strip is not None
        else int(np.floor(y_eval.min()))
    )
    y_end = H
    for y in range(y_start, y_end):
        x = int(np.clip(_line(center_model, y), 0, W - 1))
        output[y, x] = 1.0


    # Dilata la linea di un pixel per renderla piu' visibile nella maschera.
    output = cv2.dilate(
        output,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )


    # Completa l'output diagnostico con ROI, metriche e modelli dei bordi.
    roi_binary = np.zeros((H, W), np.uint8)
    roi_binary[y0:] = path * 255


    return output, {
        "status": status,
        "confidence": confidence,
        "mean_residual": residual,
        "support": support,
        "tracking": float(tracking),
        "width": width,
        "width_error": width_error,
        "max_gap": max_missed,
        "left_edge_model": left_model,
        "right_edge_model": right_model,
        "centerline_model": center_model,
        "line_start_y": y_start,
        "line_end_y": y_end,
        "roi_binary": roi_binary,
    }