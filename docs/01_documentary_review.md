# Revue documentaire — sélection des stratégies candidates

## Constat sur la source (scripts TradingView)

Le flux "Populaire/Récent" de https://www.tradingview.com/scripts/?script_type=strategies est dominé par des
scripts publiés par des comptes récents, sans historique, affichant des statistiques de backtest
extraordinaires (ex. "124% CAGR", "profit factor 7.6") calculées sur **un seul actif, une seule période, avec
des dizaines de paramètres visiblement ajustés a posteriori**. Exemple typique repéré pendant la revue :
"Sovereign Horizon Matrix v8.0" (BTC daily, 56 trades, CAGR annoncé 124%) — un échantillon de 56 trades sur un
seul actif ne permet aucune conclusion statistique fiable, et un tel rendement est un signal classique de
*curve-fitting* (paramètres optimisés sur les données historiques précises qui ont servi au test).

**Décision méthodologique** : plutôt que de faire confiance aux chiffres affichés par un auteur anonyme, on
retient uniquement des **mécanismes de marché documentés par ailleurs** (littérature académique ou
manuels de référence reconnus), et on les réimplémente nous-mêmes pour les tester de façon indépendante,
sur plusieurs actifs et plusieurs périodes. Le code Pine trouvé sur TradingView sert tout au plus
d'inspiration pour les conventions de paramètres, jamais de preuve de performance.

Un exemple concret de ce risque : l'auteur "ortizbruno115" (compte créé le 20/07/2025, 3 scripts, 5 abonnés)
publie une série de stratégies présentées comme "la stratégie de day trading la plus validée
académiquement" — la référence académique citée (Zarattini & Aziz, 2023) existe réellement, mais rien ne
garantit que son implémentation Pine soit fidèle, non répaintée, ou testée correctement. On garde l'idée,
pas le code.

## Stratégies candidates retenues

### 1. Golden/Death Cross — croisement de moyennes mobiles (référence de base)
- **Mécanisme** : long quand SMA rapide (ex. 50j) > SMA lente (ex. 200j), plat/short sinon.
- **Pourquoi ça a une chance de marcher** : capture les tendances de fond ; documenté depuis des décennies,
  utilisé comme filtre de tendance dans la quasi-totalité des systèmes trend-following institutionnels
  (CTA/managed futures). Sert de **benchmark** : si une stratégie plus complexe ne bat pas ce croisement
  simple après coûts, elle n'apporte rien.
- **Faiblesses connues** : nombreux faux signaux (whipsaws) en marché sans tendance ; toujours en retard
  (lagging) sur les retournements.
- **Marchés adaptés** : tout actif qui tend sur le long terme (indices actions, BTC/ETH en cycle haussier).
- **Automatisation IBKR** : très simple — un ordre au plus tous les quelques jours, aucune contrainte de
  latence.

### 2. Donchian Channel Breakout / "Turtle Trading System"
- **Mécanisme** : achat sur nouveau plus haut sur N jours, vente à découvert sur nouveau plus bas sur N jours ;
  stop et taille de position basés sur l'ATR (volatilité).
- **Pourquoi ça a une chance de marcher** : c'est le système systématique le plus documenté de l'histoire du
  trading — popularisé par Richard Dennis et William Eckhardt en 1983 (les "Turtle Traders"), analysé en
  détail dans la littérature sur le trend-following (ex. M. Covel, *The Complete TurtleTrader* ; études
  académiques sur la persistance du trend-following en matières premières/futures). L'edge repose sur un
  fait statistique répliqué sur des décennies : les marchés ont des queues de distribution (grosses
  tendances) que les stratégies de rupture captent, moyennant beaucoup de petites pertes.
- **Faiblesses connues** : taux de réussite souvent < 40-45%, gain concentré sur peu de trades ("hit une
  grosse tendance de temps en temps") — psychologiquement difficile à tenir en discrétionnaire, mais adapté
  à l'automatisation justement pour cette raison.
- **Marchés adaptés** : futures/matières premières historiquement, mais fonctionne aussi sur crypto et
  indices en version daily/swing.
- **Automatisation IBKR** : bien adaptée en swing (signal une fois par jour à la clôture).

### 3. Opening Range Breakout (ORB)
- **Mécanisme** : on mesure le range des N premières minutes de la séance (ex. 5 min) ; achat sur cassure du
  plus haut du range, vente à découvert sur cassure du plus bas ; sortie systématique en fin de séance
  (flat by EOD).
- **Pourquoi ça a une chance de marcher** : référence académique réelle — Carlo Zarattini & Andrew Aziz,
  *"Can Day Trading Really Be Profitable?"* (SSRN, 2023), qui montre un edge statistiquement significatif de
  l'ORB sur les grandes capitalisations US entre 1995 et 2020, avec un couple rendement/risque supérieur au
  buy-and-hold sur l'échantillon étudié. C'est l'un des rares papiers de day trading avec une méthodologie
  transparente et un vrai historique multi-décennal (pas un backtest d'un an sur un seul ticker).
- **Faiblesses connues** : sensible aux coûts de transaction et au slippage (beaucoup de trades) ; les
  résultats du papier concernent des large caps très liquides sur une longue période — rien ne garantit la
  même performance sur d'autres actifs/périodes ; nécessite un système qui tourne pendant les heures de
  marché (contrainte d'automatisation plus forte que du swing).
- **Marchés adaptés** : actions/ETF liquides US (le papier original), transposable aux indices.
- **Automatisation IBKR** : faisable (IBKR API + script tournant pendant les heures de bourse), mais plus
  exigeant qu'un signal quotidien.

### 4. RSI(2) Mean Reversion ("Connors RSI")
- **Mécanisme** : sur un actif en tendance haussière de fond (prix > SMA200), on achète un repli extrême de
  très court terme (RSI(2) très bas, ex. < 5-10) et on revend sur un retour à la normale (RSI(2) élevé ou
  clôture au-dessus du plus haut des N derniers jours).
- **Pourquoi ça a une chance de marcher** : popularisé et documenté par Larry Connors (*Short Term Trading
  Strategies That Work*, 2008) avec des tests étendus sur indices et actions US ; repose sur un phénomène de
  marché bien étudié (la sur-réaction à court terme suivie d'un retour à la moyenne, particulièrement
  visible sur indices/ETF larges).
- **Faiblesses connues** : stratégie très connue et largement répliquée depuis 15+ ans → edge probablement
  érodé (arbitré) sur les actifs les plus liquides depuis sa publication ; ne fonctionne pas en tendance
  baissière forte (le "repli" continue de creuser).
- **Marchés adaptés** : ETF actions larges (SPY, QQQ) plutôt que crypto (trop volatile, les reculs ne sont
  pas de simples "pullbacks" statistiques).
- **Automatisation IBKR** : simple, signal quotidien.

### 5. VWAP Band Mean Reversion
- **Mécanisme** : intraday, on fade les écarts extrêmes par rapport au VWAP de la séance (bande en écart-type)
  quand le marché n'est pas en tendance forte (filtre ADX) ; sortie au retour vers le VWAP.
- **Pourquoi ça a une chance de marcher** : le VWAP est un benchmark d'exécution institutionnel réel (utilisé
  pour mesurer la qualité d'exécution des gros ordres) — c'est une des raisons pour lesquelles le prix a
  tendance à graviter autour ; technique répandue chez les traders intraday sur futures indiciels (pas un
  papier académique unique, mais une pratique de desk documentée dans plusieurs manuels de microstructure).
- **Faiblesses connues** : ne marche pas les jours de forte tendance (d'où le filtre ADX, imparfait) ; edge
  plus fragile et plus dépendant des coûts d'exécution que les stratégies swing.
- **Marchés adaptés** : indices actions liquides (ES/SPY, NQ/QQQ).
- **Automatisation IBKR** : la plus exigeante des 5 (intraday, réactivité).

## Ce qu'on ne retient pas (et pourquoi)

- **Scripts "Popular/Recent" avec stats auto-déclarées extrêmes** : rejetés — pas de moyen de vérifier
  l'absence de surapprentissage, souvent testés sur un seul actif/une seule période.
- **TASC "Ag Selling Model"** (futures agricoles, 1-3 trades/an) : logique cohérente mais échantillon bien
  trop petit pour en tirer une conclusion statistique, et hors-sujet par rapport à IBKR CFD/actions.
- **TASC "One Percent A Week" (TQQQ)** : logique intéressante et documentée, mais mono-actif à effet de
  levier 3x — trop spécifique pour une étude comparative généraliste ; pourra être creusé plus tard si le
  style "mean-reversion hebdomadaire" ressort comme prometteur.

## Prochaine étape

Implémentation indépendante en Python des 5 stratégies ci-dessus, backtest sur données réelles
(crypto, actions/ETF US, forex) via `yfinance`, comparaison sur métriques robustes (CAGR, Sharpe, max
drawdown, profit factor) avec coûts de transaction réalistes — **pas** sur les chiffres publiés par les
auteurs TradingView.

## Avertissement

Cette étude est un travail de recherche/backtesting à but éducatif. Les performances passées (réelles ou
simulées) ne préjugent pas des performances futures. Ceci ne constitue pas un conseil en investissement
personnalisé.
