# proj_modele
 Porównanie modeli klasyfikacyjnych na podstawie danych z firmy ubezpieczeniowej
# 1. Wstęp
Ubezpieczenie to umowa zobowiązująca do wypłacenia świadczenia przy zaistnieniu zdarzeń zawartych w umowie zawieranej między ubezpieczonym, a ubezpieczycielem – firmą ubezpieczeniową. Zdarzenia występujące w życiu, zdrowiu lub mieniu ubezpieczonego implikują realizację umowy wypłacając odszkodowania lub świadczenia z funduszu utworzonego ze zgromadzonych składek wnoszonych przez ubezpieczonych zgodnie z umową.</br>
Firmy ubezpieczeniowe oferują różnego rodzaju pakiety kierowane do szerokiego grona klientów indywidualnych jak i korporacyjnych. Klienci indywidualni mogą skorzystać m.in. z ubezpieczeń komunikacyjnych, majątkowych, osobowo-podróżnych, od następstw nieszczęśliwych wypadków oraz OC. W zależności od rodzaju umowy jaki obszar życia, zdrowia, bądź mienia klienta ma być objęty ubezpieczeniem, obliczana jest składka świadczenia. Składka to suma pieniędzy, którą klient musi regularnie płacić firmie ubezpieczeniowej za tę gwarancję.</br>
W przypadku ubezpieczeń zdrowotnych taka umowa pozwala również na obniżenie kosztów w przypadku potrzebnej hospitalizacji. Przy regularnej płatności wielu ubezpieczonych, firma ubezpieczeniowa jest w stanie wypłacać wysokie kwoty w oparciu o umowę ubezpieczeniową bez nadwyrężania swoich zasobów pieniężnych. Dzieje się tak, gdyż statystycznie jedynie niewielka część klientów będzie faktycznie potrzebować wypłacenia świadczenia w danym miesiącu rozliczeniowym. W ten sposób wszyscy ubezpieczeni dzielą pulę funduszu dla potrzebujących oraz utrzymują firmę.</br>
Większość rodzajów ubezpieczeń jest dobrowolna i zależy to jedynie od klienta, czy decyduje się na ubezpieczenie, czy może woli ponosić koszty bez płacenia firmie ubezpieczeniowej. W Polsce zgodnie z ustawą z dnia 22 maja 2003 r. o ubezpieczeniach obowiązkowych, Ubezpieczeniowym Funduszu Gwarancyjnym i Polskim Biurze Ubezpieczycieli Komunikacyjnych uregulowano obowiązkowy wykup ubezpieczeń. Zgodnie z przepisami do ubezpieczeń obowiązkowych zalicza się:

<ol type="a">
  <li>ubezpieczenie odpowiedzialności cywilnej posiadaczy pojazdów mechanicznych (tzw. polisa komunikacyjna OC),</li>
  <li>ubezpieczenie odpowiedzialności cywilnej rolników z tytułu posiadania gospodarstwa rolnego,</li>
  <li>ubezpieczenie budynków wchodzących w skład gospodarstwa rolnego od ognia i innych zdarzeń losowych.</li>
</ol>

Do wykupienia ubezpieczenia OC zobowiązani są właściciele wszystkich zarejestrowanych pojazdów – samochodów, motocykli, traktorów, a nawet quadów. Ubezpieczenie OC ma na celu ochronę poszkodowanych w przypadku kolizji lub wypadku drogowego. Gdy kierowca ubezpieczonego pojazdu spowoduje wypadek lub kolizję, to pieniądze dla poszkodowanych wypłacane są z polisy zakładu, w którym się ubezpieczył. Dotyczy to nie tylko pokrycia należności związanych z naprawą auta, ale obejmuje też zwrot kosztów leczenia ofiar wypadku.
 
# 2. Cel
W trosce o ubezpieczonych firmy starają się dopasowywać pakiety, aby zachęcić do skorzystania z ich oferty. Moim zleceniodawcą jest towarzystwo ubezpieczeniowe (określane tutaj jako firma TU), które zapewniło swoim klientom ubezpieczenie zdrowotne. Aktualnie zmieniają strategię sprzedażową i chcieliby przewidzieć, czy ubezpieczeni (klienci) z minionego roku będą również zainteresowani ubezpieczeniem komunikacyjnym OC oferowanym przez firmę.</br>
Zbudowanie modelu w oparciu o uczenie maszynowe pozwalającego przewidzieć, czy klient byłby zainteresowany ubezpieczeniem pojazdów, jest niezwykle pomocne dla firmy TU, bo dzięki temu jest w stanie odpowiednio zaplanować strategię komunikacji, aby dotrzeć do tych klientów, zoptymalizować swój model biznesowy i przychody. Jeśli klienci firmy TU będą coraz bardziej zainteresowani kupnem, zostaną opracowane nowe pakiety obejmujące ubezpieczenie zdrowotne wraz z ubezpieczeniem OC o obniżonej sumarycznej składce, co będzie bardziej atrakcyjne dla klientów.</br>
Głównym celem mojej pracy jest dokonanie odpowiedniej klasyfikacji ubezpieczonych firmy TU przy użyciu algorytmów uczenia maszynowego, po odpowiednim zanalizowaniu bazy klientów i wybraniu jednego z algorytmów. Wybór jednego z przetestowanych algorytmów uczenia maszynowego będzie podyktowany jakościową oceną możliwości klasyfikacyjnych oraz predykcyjnych. W ten sposób zaprezentowany i przetestowany model może zostać wykorzystany przez firmę TU do maksymalizacji zysków oraz do zaoszczędzenia czasu przy szukaniu klientów zainteresowanych nowym pakietem ubezpieczeniowym.</br>
Moja analiza jest też istotna pod względem dopasowania nowej kampanii reklamowej firmy TU do typowego niezdecydowanego klienta, który mógłby zmienić zdanie i zdecydować się na nowy pakiet pod wpływem odpowiednio dopasowanej do niego reklamy.
 
# 3. Wyniki badań

## 3.1 Wstępna analiza danych
Moje badania były prowadzone w języku Python w wersji 3.8.3. Do badań wykorzystałem edytor Jupyter Notebook z pakietu Anaconda. Poza podstawowym pakietem bibliotek znajdujących się domyślnie w Anacondzie, wszystkie dodatkowe biblioteki potrzebne do badań zostały uprzednio zainstalowane. Potrzebne mi były biblioteki:
-	shap
-	xgboost
-	imblearn
Aby przewidzieć, czy klient byłby zainteresowany dodatkowym ubezpieczeniem pojazdu, dostałem bazę danych firmy TU z poprzedniego roku, zawierającą dane osób ubezpieczonych, którym zaproponowano dodatkowy pakiet ubezpieczenia zdrowotnego wraz z ubezpieczeniem OC.

<p align="center">
  <img src="https://github.com/TheLordWeirdSloughFeg/proj_modele/blob/main/obrazki/tabelka.jpg" />
</p>
<br />
<div align="center">
<b><h3>Rysunek 1. Przykładowe dane klientów firmy TU</h3></b>
</div>

Baza zawiera podstawowe informacje o klientach, takie jak płeć, wiek, kod regionu, posiadanie prawa jazdy, jak i dotyczące pojazdów (wiek pojazdu, uszkodzenia), polisie (czy klient wcześniej wykupił OC firmy TU, jego roczna składka, kanał pozyskiwania oraz od ilu dni klient jest powiązany z firmą). Ostatnią kolumną jest zainteresowanie pakietem zdrowotnym wraz z ubezpieczeniem OC.</br>
Wstępnie znając dane sprawdziłem ich typ oraz czy są braki w analizowanej bazie, co jest ważne z perspektywy utworzenia ramki danych do badań modeli uczenia maszynowego.
 
Rysunek 2. Informacja o danych klientów firmy TU
Jak można zauważyć na Rysunku 2 nie istnieją braki danych w żadnej z kolumn, co można przypisać dokładności firmy TU oraz ich przywiązaniem do kompletności danych. Kilka kolumn zawiera wartości tekstowe, dlatego w dalszych etapach analizy zamieniłem je na wartości numeryczne.</br>
Na początek sprawdziłem sumaryczne odpowiedzi klientów.
 
Rysunek 3. Wykres sumarycznej odpowiedzi
Większość klientów jest niezainteresowanych obecnym pakietem firmy TU. Należy zweryfikować jaki mógłby być potencjalny klient, aby przyszła kampania reklamowa kierowana do osób niezdecydowanych odniosła sukces.</br>
Następnie sprawdziłem też jak kształtuje się odpowiedź klientów w zależności od podanych zmiennych, co pozwoli na wstępną analizę.
 
Rysunek 4. Wykresy zmiennych w zależności od odpowiedzi klientów
Według wykresów na Rysunku 4 im młodszy jest ubezpieczony, tym chętniej decyduje się na dodatkowy pakiet. Osoby, które miały niższą składkę roczną również częściej chciały dodatkowe ubezpieczenie OC. Klienci bez wykupionego OC także decydują się chętniej na pakiet firmy TU, więc ewentualna przyszła kampania reklamowa ma szansę zwiększyć tę liczbę. Oczywistym jest, że jedynie osoby posiadające prawo jazdy są zainteresowane takim pakietem. Zmienne Vintage oraz id nie niosą żadnej wartości i nie są skorelowane z odpowiedzią klientów.</br>
Po wstępnej analizie starałem się ograniczyć dość duży zbiór danych, sprawdzając korelację między zmiennymi. Chciałem również potwierdzić brak skorelowania kolumn Vintage oraz id z odpowiedzią klientów.
 
Rysunek 5. Mapa korelacji Pearsona
Zgodnie z wartością współczynnika Pearsona skorelowane zmienne mają kolor od białego do granatowego (przedział 0:1), natomiast nieskorelowane dane mają kolor od białego do ciemnoczerwonego (przedział -1:0). Usunąłem nieskorelowane dane, które nie będą miały większego znaczenia dla dalszych badań. Te kolumny to: Region_Code, Annual_Premium, id oraz Vintage.</br>
Aby sprawdzić do kogo może być skierowana potencjalna kampania firmy TU sprawdziłem jeszcze podział odpowiedzi w zależności od zmiennych: płci ubezpieczonego, wieku pojazdu oraz czy mieli już wykupione OC przed zdecydowaniem się na nowy pakiet.
 
Rysunek 6. Wykres odpowiedzi klientów z podziałem na płeć w zależności od wcześniej wykupionego OC
Jeśli chodzi o stałych klientów, to decydowali się oni na nowy pakiet tak samo licznie, bez względu na płeć. W przypadku klientów bez wcześniej wykupionego OC, częściej na pakiet decydowali się mężczyźni.</br>
 
Rysunek 7. Wykres odpowiedzi klientów z podziałem na płeć w zależności od wieku pojazdu
Jak widać na wykresie z Rysunku 7, osoby posiadające pojazdy starsze niż 2 lata nie były praktycznie zainteresowane pakietem. Z kolei więcej kobiet posiadających nowe auta było bardziej zainteresowanych ofertą. Największą grupę zainteresowanych pakietem stanowią mężczyźni mający samochody wyprodukowane 1-2 lata temu.
Na koniec przygotowałem dane do badania algorytmami uczenia maszynowego.</br>
 
Rysunek 8. Fragment wstępnych danych po usunięciu nieskorelowanych kolumn
Tak przygotowane dane nie nadają się jako zbiór treningowy z uwagi na wartości tekstowe kolumn Gender, Vehicle_Age i Vehicle_Damage. Zamieniłem wartości typu string wymienionych kolumn na odpowiednio:
- Kolumna Gender	Male:0, Female:1,
- Kolumna Vehicle_Age	> 2 Years:2, 1-2 Year:1, < 1 Year:0,
- Kolumna Vehicle_Damage	No:0, Yes:1.
Po zastosowaniu powyższych kryteriów ramka danych wyglądała zgodnie z wycinkiem danych zaprezentowanych na Rysunku 9.
 
Rysunek 9. Fragment danych gotowych do sprawdzania algorytmów uczenia maszynowego
## 3.2 Badanie algorytmów klasyfikacyjnych
Po wstępnej analizie klientów zdecydowałem się na zbadanie pięciu popularnych algorytmów klasyfikacyjnych, charakteryzujących się dość dobrą skutecznością klasyfikacji. Są to:
I.	XGBoost (Extreme Gradient Boosting)</br>
II.	k-najbliższych sąsiadów (KNN)</br>
III.	drzewo decyzyjne</br>
IV.	regresja logistyczna</br>
V.	stochastyczny spadek wzdłuż gradientu (SGD)</br>
Zanim przystąpiłem do analizy powyższych algorytmów podzieliłem zbiór danych na treningowy i testowy.
 
Rysunek 10. Podział zbioru danych na treningowy i testowy
### I.	XGBoost
Algorytm XGBoost napisany przez Tianqi Chena w 2014 roku podobny do lasów losowych, polega na sekwencyjnym uczeniu drzewna na podstawie błędów predykcyjnych (tzw. rezyduów). Jego zaletą jest duża szybkość, gdyż kolejne drzewa uczą się na rezyduach poprzedzających predykcji.
Przed zastosowaniem biblioteki xgboost i wytrenowaniem modelu, przedstawiłem obliczone wartości Shapley’a za pomocą pakietu shap, które mogą sugerować jakie czynniki mają wpływ na decyzję klientów odnośnie rozszerzenia ubezpieczenia zdrowotnego o dodatkowe ubezpieczenie OC.
 
Rysunek 11. Graf przedstawiający wpływ zmiennych na decyzję klientów. Im dłuższa strzałka, tym większy wpływ ma dana zmienna
 
Rysunek 12. Wykres względnego znaczenia poszczególnych zmiennych w algorytmie XGBoost
 
Rysunek 13. Wykres względnego znaczenia poszczególnych na podstawie średniej wartości współczynnik Shapley’a
Na podstawie Rysunków 11-13 można stwierdzić, że na decyzję klienta mają wpływ: czy pojazd jest już uszkodzony, klient był już ubezpieczony w firmie TU, co może świadczyć o zadowoleniu klienta z ceny pakietu oraz wiek klienta.
Następnie trenuję model klasyfikatora XGBoost na zbiorze treningowym podzielonym zgodnie z Rysunkiem 10.
 
Rysunek 14. Parametry modelu klasyfikacyjnego XGBoost
Po wytrenowaniu modelu zbadałem jego możliwości predykcyjne przy pomocy krzywej ROC. Krzywa ROC jest wykorzystywana często jako narzędzie porównawcze do oceny modeli. W krzywej ROC obliczane jest pole pod krzywą (AUC) i traktowane jest jako miara precyzji i czułości wybranego modelu. Wartość AUC przyjmuje wartości od 0 do 1. Im wyższa wartość tym lepszy model.
 
Rysunek 15. Krzywa ROC dla XGBoost
Wartość AUC dla algorytmu XGBoost wyniosła ok. 0,85.

### II.	k-najbliższych sąsiadów (KNN)
Algorytm k-najbliższych sąsiadów polega na tym, że wyznacza k sąsiednich wartości do których badany element zbioru ma najbliżej dla wybranej metryki (np. Euklidesowej), a następnie wyznacza wynik w oparciu o uśrednioną wartość odległości od k sąsiednich wartości. Ostatecznie, dany element zostaje zaklasyfikowany do grupy, do której ma najbliżej. Im większy zbiór danych, tym zazwyczaj potrzebny jest większy współczynnik k.
Przed zastosowaniem algorytmu sprawdziłem, jakie k byłoby odpowiednie dla najlepszego dopasowania. W tym celu porównałem dokładność (accuracy) kolejnych modeli zmieniając ich wartość k od 1 do 35.
 
Rysunek 16. Szukanie optymalnej wartości k przez porównanie dokładności modeli
 
Rysunek 17.Wykres kolejnych wyników modeli k najbliższych sąsiadów w zależności od parametru k
Biorąc pod uwagę wykres z Rysunku 17 stwierdziłem, że najbardziej optymalna wartość k wynosi 10. Dla mniejszych wartości k, accuracy rośnie najbardziej, natomiast dla wartości większych od 10 różnica w accuracy nie jest na tyle znacząca, aby wziąć pod uwagę którąś z wartości z przedziału 11-35.
Następnie wytrenowałem model, po ustaleniu parametru k =10.
 
Rysunek 18. Wytrenowanie modelu k najbliższych sąsiadów dla parametru k = 10
 
Rysunek 19. Krzywa ROC dla k-najbliższych sąsiadów
Wartość AUC dla algorytmu k-najbliższych sąsiadów wyniosła ok. 0,80, czyli mniej niż w przypadku XGBoost.

### III.	Drzewo decyzyjne
Drzewo decyzyjne to przede wszystkim model procesu myślowego albo sztucznego mający na celu podejmowanie decyzji w zależności od podziału na jednorodne klasy. Na początku dany jest zbiór zawierający wszystkie analizowane obiekty. W trakcie analizy jest dzielony na określoną liczbę podzbiorów. Tworząc nowe gałęzie i posuwając się w dół drzewa każdy z podzbiorów podlega dalszemu podziałowi, tak aby na końcu analizy (w liściu) każdy obiekt stanowił oddzielną klasę. Takie drzewo odzwierciedla również, w jaki sposób na podstawie atrybutów algorytm drzewa decyzyjnego podejmuje decyzje klasyfikujące. Zaletą tego algorytmu jest jego czytelność dla człowieka, gdyż symuluje on ludzkie podejmowanie decyzji.
Wytrenowałem model w oparciu o algorytm drzewa decyzyjnego.
 
Rysunek 20. Wytrenowanie modelu drzewa decyzyjnego
Po wytrenowaniu modelu ponownie wykreśliłem krzywą ROC w celu porównania z resztą zastosowanych algorytmów.
 
Rysunek 21. Krzywa ROC dla drzewa decyzyjnego
Wartość AUC dla algorytmu drzewa decyzyjnego wyniosła ok. 0,82.

### IV.	Regresja logistyczna
Regresja logistyczna jest podobna do modelu regresji liniowej, ale nadaje się dla modeli, w których zmienna zależna jest dychotomiczna. Model regresji logistycznej oparty jest na funkcji przekształcającej prawdopodobieństwo na logarytm szansy zwany inaczej logitem, co pozwala na obliczanie prawdopodobieństwa danego zdarzenia (tzw. prawdopodobieństwo sukcesu). W przypadku regresji logistycznej współczynniki mogą być używane do oszacowania ilorazów szans dla każdej zmiennej niezależnej w modelu.
Wytrenowałem model stosując regresję logistyczną.
 
Rysunek 22. Wytrenowanie modelu regresji logistycznej
Po wytrenowaniu modelu wykreśliłem krzywą ROC, aby porównać go z resztą zastosowanych algorytmów.
 
Rysunek 23. Krzywa ROC dla regresji logistycznej
Wartość AUC dla algorytmu regresji logistycznej wyniosła ok. 0,83.

### V.	Stochastyczny spadek wzdłuż gradientu
Algorytm stochastycznego spadku wzdłuż gradientu jest algorytmem iteracyjnym, który rozpoczyna się od losowego punktu funkcji, następnie z kolejną iteracją przesuwa się stopniowo w dół zgodnie z gradientem, dopasowując funkcję do obserwacji. Algorytm wybiera element przechodząc zwykle po całym zbiorze danych w losowej kolejności. Zaletami tego modelu są szybkość z uwagi na oszczędność pamięci obliczeniowej oraz skalowalność.
Wytrenowałem model stosując algorytm stochastycznego spadku wzdłuż gradientu.
 
Rysunek 24. Wytrenowanie modelu stochastycznego spadku wzdłuż gradientu
Po wytrenowaniu modelu ponownie wykreśliłem krzywą ROC dla porównania z resztą zastosowanych algorytmów.
 
Rysunek 25. Krzywa ROC dla stochastycznego spadku gradientu
Wartość AUC dla algorytmu stochastycznego spadku gradientu wyniosła ok. 0,83.
 
Poniżej zestawienie wyników dla wszystkich pięciu algorytmów.
	
<div align="center">
  
| Algorytm | Pole pod krzywą ROC (AUC) |
| ----------- | ------------ |
| XGBoost (Extreme Gradient Boosting) | 0,8526257312343555 |
| k-najbliższych sąsiadów (KNN) |  0,8020140147211012 |
| drzewo decyzyjne | 0,8238167788731643 |
| regresja logistyczna | 0,8294442690064487 |
| stochastyczny spadek wzdłuż gradientu (SGD) | 0,8242933535429905 |
 
</div>
<div align="center">
<b><h3>Tabela 1. Zestawienie wyników pięciu przebadanych algorytmów.</h3></b>
</div>

Jak można zauważyć najlepiej sprawdził się algorytm XGBoost, którego współczynnik ROC wyniósł 0,85. Pozostałe algorytmy cechowały się podobną wartością współczynnika AUC w zakresie 0,82-0,83. Jedynie algorytm k-najbliższych sąsiadów mimo próby dopasowania współczynnika k miał najgorszy wynik, jeśli chodzi o wartość AUC.

## 3.3 Badanie algorytmów klasyfikacyjnych po użyciu nadpróbkowania SMOTE
Mimo przewagi XGBoost nad innymi algorytmami przy klasyfikacji osób zainteresowanych dodatkowym ubezpieczeniem OC, postanowiłem ponownie przeanalizować dane. Po ponownym przyjrzeniu się ilości odpowiedzi osób zainteresowanych i niezainteresowanych dodatkowym pakietem, zauważyłem że, zdecydowana większość nie była nim zainteresowana (patrz: Rysunek 3). Przy tak sporej dysproporcji, stwierdziłem, że warto byłoby wyrównać stosunek osób zainteresowanych do liczby osób niezainteresowanych, przez dodanie próbek syntetycznych, gdyż więcej danych może zwiększyć skuteczność przewidywania badanych algorytmów. Do tego celu zastosowałem algorytm nadpróbkowania SMOTE.
SMOTE to jeden ze sposobów zwiększenia liczby rzadkich przypadków. Algorytm SMOTE generuje nowe wystąpienia z istniejących przypadków mniejszości, które podano jako dane wejściowe. W praktyce powiela on losowo wybrane obserwacje danych z klasy, mającej przewagą liczebną.</br>
W związku z tym zastosowałem algorytm SMOTE zwiększając liczbę osób zainteresowanych wykupieniem dodatkowego ubezpieczenia pojazdu.
 
Rysunek 26. Zastosowanie nadpróbkowania metodą SMOTE
Po wykonaniu nadpróbkowania sprawdziłem jeszcze raz jak kształtuje się odpowiedź w nowym zbiorze danych.
 
Rysunek 27. Wykres sumarycznej odpowiedzi po użyciu algorytmu SMOTE
Jak widać na Rysunku 27 stosunek odpowiedzi klientów w obu grupach był jednakowy. Następnym krokiem było powtórzenie wytrenowania modeli na nowym zbiorze danych. Żaden z poprzednio użytych parametrów nie został zmieniony, aby można było zauważyć polepszenie wyników algorytmów.
 
### I.	XGBoost
 
Rysunek 28. Krzywa ROC dla XGBoost po użyciu algorytmu SMOTE
Wartość AUC dla algorytmu XGBoost po zastosowaniu SMOTE wyniosła ok. 0,86.

### II.	k-najbliższych sąsiadów (KNN)
 
Rysunek 29. Krzywa ROC dla k-najbliższych sąsiadów po użyciu algorytmu SMOTE
Wartość AUC dla algorytmu KNN po zastosowaniu SMOTE wyniosła ok. 0,85.
### III.	Drzewo decyzyjne
 
Rysunek 30. Krzywa ROC dla drzewa decyzyjnego po użyciu algorytmu SMOTE
Wartość AUC dla algorytmu drzewa decyzyjnego po zastosowaniu SMOTE wyniosła ok. 0,86.

### IV.	Regresja logistyczna
 
Rysunek 31. Krzywa ROC dla regresji logistycznej po użyciu algorytmu SMOTE
Wartość AUC dla algorytmu regresji logistycznej po zastosowaniu SMOTE wyniosła ok. 0,83.
### V.	Stochastyczny spadek wzdłuż gradientu
 
Rysunek 32. Krzywa ROC dla stochastycznego spadku gradientu po użyciu algorytmu SMOTE
Wartość AUC dla algorytmu SDG po zastosowaniu SMOTE wyniosła ok. 0,83.

Na koniec zrobiłem zestawienie wyników dla wszystkich pięciu algorytmów dla standardowych danych oraz po zastosowaniu nadpróbkowania.

<div align="center">
  
| Algorytm | Współczynnik AUC dla niezmienionych danych | Współczynnik AUC po zastosowaniu nadpróbkowania metodą SMOTE |
| ----------- | ------------ | ------------ |
| XGBoost	| 0,8526257312343555	| 0,8621444150609314 |
| k-najbliższych sąsiadów	| 0,8020140147211012	| 0,8449268599794624 |
| drzewo decyzyjne	| 0,8238167788731643	| 0,8654773411885064 |
| regresja logistyczna	| 0,8294442690064487	| 0,8336418975903985 |
| SGD	| 0,8242933535429905	| 0,8269947343971508 |

</div>
<div align="center">
<b><h3>Tabela 2. Zestawienie wyników pięciu przebadanych algorytmów na niezmienionych danych oraz po zastosowaniu SMOTE.</h3></b>
</div>

W przypadku modeli po zastosowaniu algorytm SMOTE najlepsze wyniki osiągnął algorytm drzewa decyzyjnego zwiększając współczynnik AUC o ponad 0,4. Co ciekawe w przypadku XGBoost współczynnik AUC zwiększył się jedynie o około 0,1. Najmniejszą różnicę widać przy algorytmie SGD, praktycznie wynik AUC pozostał ten sam.
# 4. Podsumowanie i wnioski
W celu zanalizowania i stworzenia modelu predykcyjnego danych klientów firmy TU dokonałem porównania pięciu algorytmów klasyfikujących: XGBoost, k-najbliższych sąsiadów, drzewa decyzyjnego, regresji logistycznej oraz stochastycznego spadku wzdłuż gradientu przy nadpróbkowaniu algorytmem SMOTE oraz bez niego.
Baza dostarczona przez zleceniodawca nie miała braków w danych, jednak miała ona zbyt dużo zmiennych. Po wstępnej analizie stworzyłem ramkę danych, która posłużyła mi do trenowania wspomnianych pięciu modeli uczenia maszynowego.</br>
Biorąc pod uwagę współczynnik AUC, najlepsze wyniki osiągnął algorytm drzewa decyzyjnego z nadpróbkowaniem algorytmem SMOTE przy AUC około 0,86 oraz o krzywej ROC zaprezentowanej na Rysunku 30. Nadpróbkowanie algorytmem SMOTE jest wskazane dla danych o niewielkiej wymiarowości, dlatego tutaj sprawdziło się dobrze. Z uwagi na przejrzystość i prostotę tego modelu, algorytm drzewa decyzyjnego można przedstawić firmie, dlatego też moje badania będą bardziej klarowne dla mojego zleceniodawcy. Model w oparciu o drzewo decyzyjne z nadpróbkowaniem metodą SMOTE dobrze sprawdzi się w predykcji klientów zainteresowanych dodatkowym ubezpieczeniem OC w kolejnych latach.</br>
Analizując bezpośrednio dane, kampania reklamowa firmy TU powinna być skierowana do młodych klientów płci męskiej posiadających prawo jazdy i stosunkowo nowy pojazd (roczny i dwuletni). Firma TU powinna także zastanowić się nad obniżeniem ceny pakietu, np. oferując dwa ubezpieczenia: zdrowotne i OC z rabatem, by zachęcić niezdecydowanych. Moje badania dla firmy TU mogą być zatem przyczynkiem do rozwoju firmy i poszerzeniu ich oferty np. dla młodych kierowców.
