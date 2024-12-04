# Огляд проєкту

Цей проєкт реалізовує алгоритм обробки графів, зокрема визначення **PageRank**, для аналізу наукових статей, їхніх цитувань та структурних зв’язків між ними. Алгоритми базуються на принципах дискретної математики, таких як графи, їхні орієнтовані ребра та обхід графу за допомогою пошуку в ширину (алгоритм BFS).

## Бібліотеки

`argparse` - зчитування аргументу з терміналу

`import unicodedata` - використовували у функції

normalize_text, щоб правильно відображалися лігатури

`import re` - патерн для зчитування

`import os` - перевірка наявності файлів у папці

`from collections import deque` - створення черги для алгоритму BFS

`import networkx as nx` - робота з графами

`from matplotlib.colors import LinearSegmentedColormap` - візуалізація графів по кольорах у градації

`import matplotlib.pyplot as plt` - візуалізація графу

`from pdfminer.high_level import extract_text` - зчитування текстового вмісту із PDF-документів


## Функції

Проєкт має такі основні функції:

### 1. **`name_changer`**
Ця функція змінює імена файлів, видаляючи небажані, заборонені в назвах файлів символи, наприклад `":"`,
і роблячи імена зручними для подальшої обробки.

**Аргументи:**
- `name (str)` — початкова назва файлу.

**Повертає:**
- `str` — нова назва.

---

### 2. **`normalize_text`**
Нормалізує текст, замінюючи унікальні символи на стандартні.

**Приклад:**
- Символ `ﬁ` (лігатура) перетворюється на `fi`.

---

### 3. **`read_pdf`**
Функція читає PDF-файл, виділяє посилання на інші статті ("References") та будує граф на основі цитувань.

**Аргументи:**
- `file_name (str)` — шлях до PDF-файлу.

**Повертає:**
- `dict[str: list]` — граф у вигляді словника, де ключі — назви статей, а значення — список цитованих статей.

---

### 4. **`pages_directions`**
Перетворює граф у формат списку орієнтованих ребер і змінює імена статей на літери.

**Аргументи:**
- `pages (dict)` — граф.

**Повертає:**
- `tuple[list[tuple], dict]` — список напрямків (ребер) і словник зіставлення імен статей з літерами.

---

### 5. **`bfs`**
Алгоритм обходу графу в ширину (BFS), що використовується для аналізу досяжності вершин.

**Аргументи:**
- `graph (nx.DiGraph)` — орієнтований граф.
- `start (str)` — стартовий вузол.

**Повертає:**
- `list` — впорядкована послідовність відвіданих вершин.

---

### 6. **`page_rank_calc`**
Обраховує PageRank для даного графа
**Аргументи:**
- `graph (nx.DiGraph)` - орієнтований граф
- `page_rank (dict)` - page rank для кожної вершнини
- `page_current (dict)` - тимчасовий page rank для кожної вершнини під час однієї ітерації
- `damping_factor (float)` - коефіцієнт демпінгу (0.85)
**Повертає:**
- `dict` — розрахований page rank для кожної вершнини.

### 7. **`main`**
Основна функція проєкту. Вона виконує:
- Побудову графу з PDF-файлів.
- виклик функції page_rank_calc
- Візуалізацію графу зі значеннями PageRank.

**Аргументи:**
- `path (str)` — шлях до основного PDF-файлу.

---

## Основи дискретної математики

Проєкт базується на таких принципах дискретної математики:
1. **Графи:** використання орієнтованих графів для моделювання зв’язків між статтями.
2. **Алгоритми обходу:** BFS для аналізу досяжності вершин.
3. **PageRank:** математична модель для оцінювання важливості вершин у графі.

---

# Алгоритм PageRank

## Опис алгоритму

PageRank — це ітеративний алгоритм, створений для оцінки **важливості вершин у графі**. Спочатку його було розроблено в Google для ранжування веб-сторінок, але він також використовується в інших сферах, де потрібно оцінити значимість об'єктів на основі їхніх взаємозв'язків.

### Основна ідея
- Вершина графу є важливішою, якщо на неї посилається більше вершин.
- Важливість вершин, що посилаються, також враховується: посилання від важливих вершин мають більшу вагу.

---

## Формула PageRank

\[
PR(A) = (1-d) + d * (PR(T1)/C(T1) + ... + PR(Tn)/C(Tn))
\]

де:
- \( PR(A) \) — значення PageRank вузла \(A\);
- \( d \) — коефіцієнт демпінгу (зазвичай \( d = 0.85 \)), який враховує ймовірність переходу на випадковий вузол;
- \( T_1, T_2, T_n \) — вузли, які посилаються на \(A\);
- \( C(T_i) \) — кількість вихідних зв'язків із вузла \( T_i \).

---

## Етапи роботи алгоритму

1. **Розрахунок внесків**:
   - Якщо стаття \( T_1 \) цитує статтю \( A \), її внесок у рейтинг статті \( A \) становить \( {PR(T_1)}{C(T_1)} \), де \( PR(T_1) \) — поточний рейтинг \( T_1 \), а \( C(T_1) \) — кількість статей, на які посилається \( T_1 \).

2. **Ітеративне оновлення**:
   - Значення \( PR(A) \) оновлюється, підсумовуючи внески від усіх статей, які цитують \( A \), і враховуючи коефіцієнт демпінгу \( d \).

---

## Використання у проєкті

У цьому проєкті алгоритм PageRank використовується для аналізу **графу цитувань** наукових статей. Його метою є визначення, які статті є найбільш важливими та впливовими у контексті цитованої літератури.

1. **Побудова графу цитувань**:
   - Кожна стаття представляється як вершин графу.
   - Якщо стаття \( A \) цитує статтю \( B \), створюється спрямоване ребро \( A, B \).

2. **Обчислення значень PageRank**:
   - Алгоритм визначає відносну важливість кожної статті в графі на основі кількості та ваги цитувань.

3. **Візуалізація**:
   - Вершини графу розташовуються у вигляді кола.
   - Значення PageRank відображаються кольором і розміром вузлів: більш важливі статті позначаються яскравішими кольорами.

4. **Результат**:
   - Значення PageRank сортуються, що дозволяє визначити найвпливовіші статті в контексті цитованої літератури.

---

## Практичне значення

- **Аналіз**:
  PageRank дозволяє автоматично визначати ключові статті серед великої кількості джерел.

- **Оптимізація пошуку**:
  Може бути застосований для покращення результатів пошуку, пропонуючи користувачеві найважливіші ресурси.

- **Вивчення зв’язків**:
  Використання графів дозволяє краще зрозуміти взаємозв’язки між об’єктами дослідження.

---

## Використання

1. Встановити бібліотеки `networkx`, `pdfminer`, та `matplotlib`.
2. Додати PDF-файли до каталогу.
3. Викликати функцію `main()` із шляхом до основного PDF-файлу.
4. Отримати графічну візуалізацію графу та PageRank.

---

## Розподіл роботи
1. Мних Остап - пошук бази даних та функції для читання статтей
2. Гуцуляк Олег - обрахунок BFS та функції обрахунку PageRank
3. Гаєвська Юстина - пошук бази даних та створення функції `pages_directions`
4. Антонюк Дарина - функція `main`, презентація і звіт
5. Ясінська Анна - візуалізація графів
---

## Враження та відгуки

Виконання проєкту допомогло поглибити знання у галузі роботи з графами та дискретною математикою, отримати досвід в організації ефективної командної роботи. Рекомендації:
- Розширити курс прикладами реалізації PageRank для реальних застосувань та детальнішими поясненнями роботи з PageRank.

---

## Фідбек
1. Реалізація алгоритму PageRank дозволила зрозуміти його структуру, а також сфери застосування в реальних задачах.
2. Використання Python-бібліотек значно підвищило наше розуміння та закріпило раніше набуті навички.
3. Проєкт допоміг засвоїти принципи роботи з алгоритмом BFS, графами та їх реалізацією на практиці.