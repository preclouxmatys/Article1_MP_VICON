# Idée pour un article ultérieur : coordination interpersonnelle (mêmes données)

Analyse réalisée en octobre 2026 pendant la révision de l'article 1, puis **retirée de l'article 1**
(choix de M. Precloux) pour être développée dans un article séparé.

## Question
MediaPipe permet-il de retrouver les mêmes effets de coordination interpersonnelle que le Vicon
en conversation dyadique spontanée ? (validité au niveau du construit interactionnel)

## Méthode (déjà codée)
- Scripts : `src/revision/r06_coordination.py` (indices) et `src/revision/r06b_coordination_stats.py` (stats).
- Prérequis : `r01` → `r02` → `r04` (séries temporelles synchronisées dans `data/intermediate/revision/timeseries/`).
- Signaux : vitesse des poignets (G+D) et de la tête (nez MP ; milieu des tempes Vicon), pour 3 systèmes :
  MediaPipe (vidéo de chaque partenaire), Vicon projeté dans chaque vidéo, Vicon 3D.
- Base de temps commune : horloge Vicon, via le décalage estimé de chaque vidéo.
- Indice : corrélation croisée fenêtrée (Boker et al., 2002) — fenêtres 5 s, pas 1 s, décalages ±2 s ;
  pic de corrélation par fenêtre, moyenné en z de Fisher.
- Référence de hasard : pseudo-dyades (P1 d'une dyade × P2 d'une autre, même condition ; 5 par participant = 300).
- Stats : accord MP–Vicon (r, ICC(A,1)) ; réelles vs pseudo (d de Cohen, permutation 5 000 tirages).

## Résultats préliminaires (60 dyades × conditions)
| Signal | Accord MP–Vicon projeté | Réelles vs pseudo |
|---|---|---|
| Tête | r = .72, ICC = .67 | MP d = 0.48, p = .001 ; Vicon proj. d = 0.61, p < .001 ; Vicon 3D d = 0.59, p < .001 |
| Poignets | r = .82, ICC = .74 | aucun effet, quel que soit le système (d ≤ 0.12) |
Fichiers : `data/final/revision/coordination_indices.csv`, `T_coordination_*.csv`, `figures/fig7_coordination.png`.

## À développer avant publication
- Cadre théorique (coordination tête vs corps : Kodama et al., 2024 ; Marin et al., 2009 ; Varlet et al., 2011).
- Analyse de sensibilité des paramètres (fenêtre, décalage max) ; effet de la posture.
- Méthodes temps-fréquence : cross-wavelet (Issartel et al., 2015), éventuellement CRQA.
- Leader/suiveur (signe du décalage au pic), lien avec les tours de parole si l'audio est disponible.
- Références utiles : Boker et al. (2002) Psychol Methods 7(3):338–355 ; Fujiwara & Yokomitsu (2021) BRM 53(6):2700–2711 ;
  Kodama, Shimizu & Fujiwara (2024) Front Psychol 15:1296521 ; Ramseyer & Tschacher (2011) ; Paxton & Dale (2017).
