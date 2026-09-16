from __future__ import annotations

import random
import math
from typing import Iterator, Iterable, Any
import torch
from torch.utils.data import Dataset, IterableDataset


class SyntheticDataGenerator:
    """Procedural generator for trilingual (Ukrainian, Russian, English) code, math, and conversational datasets.
    
    Generates billions of possible unique samples on-the-fly without saving gigabytes of files to disk.
    Supports languages: 'uk' (Ukrainian), 'ru' (Russian), 'en' (English).
    """

    def __init__(self, seed: int | None = None, languages: list[str] | None = None) -> None:
        self.rng = random.Random(seed)
        self.languages = languages or ["uk", "ru", "en"]

    def set_seed(self, seed: int) -> None:
        self.rng.seed(seed)

    def _choose_lang(self) -> str:
        return self.rng.choice(self.languages)

    # -------------------------------------------------------------------------
    # 1. MATH GENERATORS
    # -------------------------------------------------------------------------

    def _gen_arithmetic(self) -> str:
        lang = self._choose_lang()
        op = self.rng.choice(["+", "-", "*", "//", "mixed"])
        a = self.rng.randint(2, 500)
        b = self.rng.randint(2, 100)

        if op == "+":
            ans = a + b
            expr = f"{a} + {b}"
        elif op == "-":
            ans = a - b
            expr = f"{a} - {b}"
        elif op == "*":
            a = self.rng.randint(2, 50)
            b = self.rng.randint(2, 30)
            ans = a * b
            expr = f"{a} * {b}"
        elif op == "//":
            b = self.rng.randint(2, 25)
            ans = self.rng.randint(2, 50)
            a = ans * b
            expr = f"{a} / {b}"
        else:
            c = self.rng.randint(2, 20)
            ans = (a + b) * c
            expr = f"({a} + {b}) * {c}"

        if lang == "uk":
            prompts = [
                f"Скільки буде {expr}?",
                f"Обчисли значення виразу: {expr}",
                f"Порахуй: {expr}",
                f"Чому дорівнює {expr}?",
            ]
            answers = [
                f"{expr} = {ans}",
                f"Результат обчислення {expr} дорівнює {ans}.",
                f"Відповідь: {ans}.",
                f"Рахуємо по кроках:\n{expr} = {ans}.",
            ]
            q, a_text = self.rng.choice(prompts), self.rng.choice(answers)
            return f"Користувач: {q}\nАсистент: {a_text}"
        elif lang == "ru":
            prompts = [
                f"Сколько будет {expr}?",
                f"Вычисли значение выражения: {expr}",
                f"Посчитай: {expr}",
                f"Чему равно {expr}?",
            ]
            answers = [
                f"{expr} = {ans}",
                f"Результат вычисления {expr} равен {ans}.",
                f"Ответ: {ans}.",
                f"Считаем по шагам:\n{expr} = {ans}.",
            ]
            q, a_text = self.rng.choice(prompts), self.rng.choice(answers)
            return f"Пользователь: {q}\nАссистент: {a_text}"
        else:
            prompts = [
                f"What is {expr}?",
                f"Calculate the value of: {expr}",
                f"Compute: {expr}",
                f"Evaluate {expr}.",
            ]
            answers = [
                f"{expr} = {ans}",
                f"The result of {expr} is {ans}.",
                f"Answer: {ans}.",
                f"Step by step:\n{expr} = {ans}.",
            ]
            q, a_text = self.rng.choice(prompts), self.rng.choice(answers)
            return f"User: {q}\nAssistant: {a_text}"

    def _gen_linear_equation(self) -> str:
        lang = self._choose_lang()
        a = self.rng.randint(2, 12)
        x_val = self.rng.randint(1, 20)
        b = self.rng.randint(1, 50)
        c = a * x_val + b

        if lang == "uk":
            q = f"Розв'яжи рівняння: {a}x + {b} = {c}"
            a_text = (
                f"Розв'язуємо лінійне рівняння {a}x + {b} = {c}:\n"
                f"1) Переносимо {b} у праву частину: {a}x = {c} - {b}\n"
                f"   {a}x = {c - b}\n"
                f"2) Ділимо обидві частини на {a}: x = {c - b} / {a}\n"
                f"   x = {x_val}\n"
                f"Відповідь: x = {x_val}."
            )
            return f"Користувач: {q}\nАсистент: {a_text}"
        elif lang == "ru":
            q = f"Реши уравнение: {a}x + {b} = {c}"
            a_text = (
                f"Решаем линейное уравнение {a}x + {b} = {c}:\n"
                f"1) Переносим {b} в правую часть: {a}x = {c} - {b}\n"
                f"   {a}x = {c - b}\n"
                f"2) Делим обе части на {a}: x = {c - b} / {a}\n"
                f"   x = {x_val}\n"
                f"Ответ: x = {x_val}."
            )
            return f"Пользователь: {q}\nАссистент: {a_text}"
        else:
            q = f"Solve the equation: {a}x + {b} = {c}"
            a_text = (
                f"Solving the linear equation {a}x + {b} = {c}:\n"
                f"1) Subtract {b} from both sides: {a}x = {c} - {b}\n"
                f"   {a}x = {c - b}\n"
                f"2) Divide both sides by {a}: x = {c - b} / {a}\n"
                f"   x = {x_val}\n"
                f"Answer: x = {x_val}."
            )
            return f"User: {q}\nAssistant: {a_text}"

    def _gen_word_problem(self) -> str:
        lang = self._choose_lang()
        items_uk = ["яблук", "книг", "ручок", "цукерок", "олівців", "машин", "зошитів"]
        names_uk = ["Тарас", "Богдан", "Оксана", "Софія", "Андрій", "Марія", "Данило", "Олена"]
        items_ru = ["яблок", "книг", "ручек", "конфет", "карандашей", "машин", "тетрадей"]
        names_ru = ["Алексей", "Иван", "Мария", "Анна", "Дмитрий", "Елена", "Михаил", "Ольга"]
        items_en = ["apples", "books", "pens", "candies", "pencils", "cookies", "notebooks"]
        names_en = ["Alice", "Bob", "Charlie", "Diana", "Emily", "Frank", "George", "Helen"]

        if lang == "uk":
            item = self.rng.choice(items_uk)
            name1 = self.rng.choice(names_uk)
            n1 = self.rng.randint(10, 50)
            n2 = self.rng.randint(2, n1 - 1)
            rem = n1 - n2
            q = f"У {name1} було {n1} {item}. Потім {name1} віддав(ла) другові {n2} {item}. Скільки {item} залишилося?"
            a_text = (
                f"Покрокове розв'язання задачі:\n"
                f"1. Початкова кількість: {n1}\n"
                f"2. Віддано: {n2}\n"
                f"3. Обчислюємо залишок: {n1} - {n2} = {rem}\n"
                f"Відповідь: залишилося {rem} {item}."
            )
            return f"Користувач: {q}\nАсистент: {a_text}"
        elif lang == "ru":
            item = self.rng.choice(items_ru)
            name1 = self.rng.choice(names_ru)
            n1 = self.rng.randint(10, 50)
            n2 = self.rng.randint(2, n1 - 1)
            rem = n1 - n2
            q = f"У {name1} было {n1} {item}. Затем {name1} отдал(а) другу {n2} {item}. Сколько {item} осталось?"
            a_text = (
                f"Решение задачи:\n"
                f"1. Начальное количество: {n1}\n"
                f"2. Отдано: {n2}\n"
                f"3. Вычисляем остаток: {n1} - {n2} = {rem}\n"
                f"Ответ: осталось {rem} {item}."
            )
            return f"Пользователь: {q}\nАссистент: {a_text}"
        else:
            item = self.rng.choice(items_en)
            name1 = self.rng.choice(names_en)
            n1 = self.rng.randint(10, 50)
            n2 = self.rng.randint(2, n1 - 1)
            rem = n1 - n2
            q = f"{name1} had {n1} {item}. Then {name1} gave {n2} {item} to a friend. How many {item} are left?"
            a_text = (
                f"Step-by-step solution:\n"
                f"1. Initial count: {n1}\n"
                f"2. Given away: {n2}\n"
                f"3. Remaining count: {n1} - {n2} = {rem}\n"
                f"Answer: {rem} {item} left."
            )
            return f"User: {q}\nAssistant: {a_text}"

    # -------------------------------------------------------------------------
    # 2. CODE & ALGORITHMS GENERATORS
    # -------------------------------------------------------------------------

    def _gen_python_function(self) -> str:
        lang = self._choose_lang()
        fn_type = self.rng.choice([
            "factorial", "is_even", "sum_list", "reverse_string", "find_max",
            "is_palindrome", "fibonacci", "count_vowels", "filter_positive"
        ])

        snippets = {
            "factorial": (
                "функцію для обчислення факторіала числа",
                "функцию для вычисления факториала числа",
                "a function to compute the factorial of a number",
                "def factorial(n: int) -> int:\n    if n <= 1:\n        return 1\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result",
                "Функція ітерується від 2 до n та перемножує числа.",
                "Функция итерируется от 2 до n и перемножает числа.",
                "The function iterates from 2 to n and multiplies the values."
            ),
            "is_even": (
                "функцію для перевірки, чи є число парним",
                "функцию для проверки, является ли число чётным",
                "a function to check if an integer is even",
                "def is_even(n: int) -> bool:\n    return n % 2 == 0",
                "Оператор % повертає остачу від ділення на 2.",
                "Оператор % возвращает остаток от деления на 2.",
                "The % operator returns the remainder when divided by 2."
            ),
            "sum_list": (
                "функцію, яка повертає суму елементів списку",
                "функцию, которая возвращает сумму элементов списка",
                "a function that returns the sum of elements in a list",
                "def sum_list(numbers: list[int | float]) -> float:\n    total = 0\n    for num in numbers:\n        total += num\n    return total",
                "Функція сумує всі елементи масиву в циклі.",
                "Функция суммирует все элементы массива в цикле.",
                "The function sums all items in the array using a loop."
            ),
            "reverse_string": (
                "функцію для розвороту рядка",
                "функцию для разворота строки",
                "a function to reverse a string",
                "def reverse_string(s: str) -> str:\n    return s[::-1]",
                "Використовується зріз [::-1] для швидкого розвороту рядка.",
                "Используется срез [::-1] для быстрого разворота строки.",
                "Uses python slicing [::-1] for fast string reversal."
            ),
            "find_max": (
                "функцію пошуку максимального числа у списку",
                "функцию поиска максимального числа в списке",
                "a function to find the maximum number in a list",
                "def find_max(numbers: list[int]) -> int:\n    if not numbers:\n        raise ValueError('List is empty')\n    current_max = numbers[0]\n    for num in numbers[1:]:\n        if num > current_max:\n            current_max = num\n    return current_max",
                "Алгоритм проходить по списку та оновлює максимум.",
                "Алгоритм проходит по списку и обновляет максимум.",
                "The algorithm traverses the list and updates the maximum."
            ),
            "is_palindrome": (
                "функцію перевірки паліндрома (читається однаково вперед і назад)",
                "функцию проверки палиндрома (читается одинаково вперед и назад)",
                "a function to check if a word is a palindrome",
                "def is_palindrome(text: str) -> bool:\n    cleaned = ''.join(c.lower() for c in text if c.isalnum())\n    return cleaned == cleaned[::-1]",
                "Очищує рядок від пробілів та регістру, потім порівнює з перевернутою копією.",
                "Очищает строку от пробелов и регистра, затем сравнивает с перевернутой версией.",
                "Cleans spaces and case, then compares with its reversed copy."
            ),
            "fibonacci": (
                "функцію генерації перших n чисел Фібоначчі",
                "функцию генерации n чисел Фибоначчи",
                "a function to generate the first n Fibonacci numbers",
                "def fibonacci(n: int) -> list[int]:\n    if n <= 0:\n        return []\n    if n == 1:\n        return [0]\n    seq = [0, 1]\n    while len(seq) < n:\n        seq.append(seq[-1] + seq[-2])\n    return seq",
                "Кожне наступне число обчислюється як сума двох попередніх.",
                "Каждое следующее число вычисляется как сумма двух предыдущих.",
                "Each subsequent number is computed as the sum of the prior two."
            ),
            "count_vowels": (
                "функцію для підрахунку голосних літер у рядку",
                "функцию для подсчета гласных букв в строке",
                "a function to count vowels in a string",
                "def count_vowels(s: str) -> int:\n    vowels = set('aeiouаеєиіїоуюяAEIOUАЕЄИІЇОУЮЯаеёиоуыэюяAEIOUАЕЁИОУЫЭЮЯ')\n    return sum(1 for ch in s if ch in vowels)",
                "Використовує множину голосних для швидкої перевірки кожного символу за O(1).",
                "Использует множество гласных для эффективной O(1) проверки каждого символа.",
                "Uses a set of vowels for O(1) membership check of each character."
            ),
            "filter_positive": (
                "функцію, яка повертає тільки додатні числа зі списку",
                "функцию, которая оставляет только положительные числа из списка",
                "a function that filters and returns only positive numbers from a list",
                "def filter_positive(nums: list[int]) -> list[int]:\n    return [x for x in nums if x > 0]",
                "Використовує спискове включення (list comprehension) для фільтрації елементів.",
                "Использует list comprehension для фильтрации элементов больше нуля.",
                "Uses list comprehension to filter values strictly greater than zero."
            ),
        }

        desc_uk, desc_ru, desc_en, code, exp_uk, exp_ru, exp_en = snippets[fn_type]

        if lang == "uk":
            q = f"Напиши на Python {desc_uk}."
            a_text = f"Ось реалізація на Python:\n\n```python\n{code}\n```\n\n{exp_uk}"
            return f"Користувач: {q}\nАсистент: {a_text}"
        elif lang == "ru":
            q = f"Напиши на Python {desc_ru}."
            a_text = f"Вот реализация на Python:\n\n```python\n{code}\n```\n\n{exp_ru}"
            return f"Пользователь: {q}\nАссистент: {a_text}"
        else:
            q = f"Write in Python {desc_en}."
            a_text = f"Here is the Python implementation:\n\n```python\n{code}\n```\n\n{exp_en}"
            return f"User: {q}\nAssistant: {a_text}"

    def _gen_sql_query(self) -> str:
        lang = self._choose_lang()
        tables = ["users", "orders", "products", "employees", "customers"]
        table = self.rng.choice(tables)

        queries = [
            (
                f"Як вибрати всі записи з таблиці `{table}`?",
                f"Как выбрать все записи из таблицы `{table}`?",
                f"How do I select all records from `{table}` table?",
                f"SELECT * FROM {table};",
                f"Запит повертає всі рядки та стовпчики з таблиці {table}.",
                f"Запрос возвращает все строки и столбцы из таблицы {table}.",
                f"This query returns all rows and columns from table {table}."
            ),
            (
                f"Напиши SQL-запит для вибору записів з `{table}`, відсортованих за id за спаданням.",
                f"Напиши SQL-запрос для выбора записей из `{table}`, отсортированных по id по убыванию.",
                f"Write a SQL query to select records from `{table}` sorted by id descending.",
                f"SELECT * FROM {table} ORDER BY id DESC;",
                "Ключове слово ORDER BY з модифікатором DESC сортує результати за спаданням.",
                "Ключевое слово ORDER BY с модификатором DESC сортирует по убыванию.",
                "The ORDER BY clause with DESC modifier sorts results in descending order."
            ),
            (
                f"Як порахувати кількість рядків у таблиці `{table}`?",
                f"Как посчитать количество строк в таблице `{table}`?",
                f"How to count the total number of rows in table `{table}`?",
                f"SELECT COUNT(*) AS total_count FROM {table};",
                "Агрегатна функція COUNT(*) підраховує кількість записів.",
                "Агрегатная функция COUNT(*) подсчитывает число записей.",
                "The aggregate function COUNT(*) calculates the total number of records."
            ),
        ]

        q_uk, q_ru, q_en, sql, exp_uk, exp_ru, exp_en = self.rng.choice(queries)
        if lang == "uk":
            return f"Користувач: {q_uk}\nАсистент: Ось SQL-запит:\n\n```sql\n{sql}\n```\n\n{exp_uk}"
        elif lang == "ru":
            return f"Пользователь: {q_ru}\nАссистент: Вот SQL-запрос:\n\n```sql\n{sql}\n```\n\n{exp_ru}"
        else:
            return f"User: {q_en}\nAssistant: Here is the SQL query:\n\n```sql\n{sql}\n```\n\n{exp_en}"

    # -------------------------------------------------------------------------
    # 3. CONVERSATIONAL & GENERAL KNOWLEDGE GENERATORS
    # -------------------------------------------------------------------------

    def _gen_qa(self) -> str:
        lang = self._choose_lang()
        qa_triplets = [
            (
                "Хто ти і що вмієш робити?",
                "Кто ты и что умеешь делать?",
                "Who are you and what can you do?",
                "Я NexusAI — інтелектуальний асистент. Я можу відповідати на запитання, писати та налагоджувати програмний код, розв'язувати математичні задачі та аналізувати дані трьома мовами (українською, російською та англійською).",
                "Я NexusAI — интеллектуальный ассистент. Я могу отвечать на вопросы, писать и отлаживать программный код, решать математические задачи и анализировать данные.",
                "I am NexusAI, an intelligent AI assistant. I can answer questions, write and debug software code, solve mathematical problems, and analyze data."
            ),
            (
                "Що таке машинне навчання простими словами?",
                "Что такое машинное обучение простыми словами?",
                "What is machine learning in simple words?",
                "Машинне навчання — це галузь штучного інтелекту, де комп'ютер навчається знаходити закономірності у великих обсягах даних і робити прогнози без жорсткого програмування кожного правила.",
                "Машинное обучение — это раздел искусственного интеллекта, где компьютер учится находить закономерности в больших объёмах данных и делать предсказания без жесткого программирования каждого правила.",
                "Machine learning is a branch of artificial intelligence where computers learn patterns from large amounts of data to make predictions without being explicitly programmed for every rule."
            ),
            (
                "У чому різниця між TCP і UDP?",
                "В чём разница между TCP и UDP?",
                "What is the difference between TCP and UDP?",
                "TCP гарантує надійну доставку пакетів із підтвердженням і контролем черговості (підходить для вебу та файлів), а UDP відправляє пакети без перевірки доставки, забезпечуючи мінімальну затримку (підходить для онлайн-ігор і стримінгу).",
                "TCP гарантирует надёжную доставку пакетов с подтверждением и контролем порядка (подходит для веба и файлов), а UDP отправляет пакеты без проверки доставки, обеспечивая минимальную задержку (подходит для онлайн-игр и видеостримов).",
                "TCP ensures reliable packet delivery with confirmation and ordering (ideal for web and files), whereas UDP transmits packets without delivery confirmation for minimal latency (ideal for gaming and video streaming)."
            ),
            (
                "Яка швидкість світла у вакуумі?",
                "Какова скорость света в вакууме?",
                "What is the speed of light in vacuum?",
                "Швидкість світла у вакуумі становить приблизно 299 792 458 метрів на секунду (близько 300 000 км/с).",
                "Скорость света в вакууме составляет приблизительно 299 792 458 метров в секунду (около 300 000 км/с).",
                "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 300,000 km/s)."
            ),
            (
                "Що таке стек (Stack) у структурах даних?",
                "Что такое стек (Stack) в структурах данных?",
                "What is a Stack in data structures?",
                "Стек — це структура даних, що працює за принципом LIFO (Last In, First Out — останнім прийшов, першим пішов). Основні операції: push (додавання елемента на вершину) та pop (видалення елемента з вершини).",
                "Стек — это структура данных, работающая по принципу LIFO (Last In, First Out — последним пришёл, первым ушёл). Основные операции: push (добавление элемента на вершину) и pop (удаление элемента с вершины).",
                "A stack is a data structure operating on the LIFO principle (Last In, First Out). The primary operations are push (add to top) and pop (remove from top)."
            ),
            (
                "Яка столиця України?",
                "Какова столица Украины?",
                "What is the capital of Ukraine?",
                "Столиця України — місто Київ, величне історичне місто на берегах річки Дніпро.",
                "Столица Украины — город Киев, древний исторический город на берегах реки Днепр.",
                "The capital of Ukraine is Kyiv, a historic city situated on the Dnipro River."
            ),
            (
                "Що таке Git і навіщо він потрібен?",
                "Что такое Git и зачем он нужен?",
                "What is Git and why is it needed?",
                "Git — це розподілена система контролю версій, яка дозволяє відстежувати зміни в коді, повертатися до попередніх версій і комфортно співпрацювати в команді.",
                "Git — это распределённая система контроля версий, которая позволяет отслеживать изменения в коде, возвращаться к предыдущим версиям и комфортно работать над проектом в команде.",
                "Git is a distributed version control system that tracks code changes, allows rolling back to prior versions, and facilitates collaboration across teams."
            ),
            (
                "Поясни принцип роботи алгоритму бінарного пошуку.",
                "Объясни принцип работы алгоритма бинарного поиска.",
                "Explain the principle of binary search.",
                "Бінарний пошук працює на відсортованому масиві: він порівнює шуканий елемент із середнім. Якщо шукане менше за середнє, пошук продовжується в лівій половині, інакше — в правій. Складність алгоритму становить O(log n).",
                "Бинарный поиск работает на отсортированном массиве: он сравнивает искомый элемент со средним. Если искомое меньше среднего, поиск продолжается в левой половине, иначе — в правой. Сложность алгоритма составляет O(log n).",
                "Binary search works on sorted arrays: it compares the target with the middle element. If smaller, it searches the left half; if larger, the right half. Time complexity is O(log n)."
            ),
        ]

        q_uk, q_ru, q_en, a_uk, a_ru, a_en = self.rng.choice(qa_triplets)
        if lang == "uk":
            return f"Користувач: {q_uk}\nАсистент: {a_uk}"
        elif lang == "ru":
            return f"Пользователь: {q_ru}\nАссистент: {a_ru}"
        else:
            return f"User: {q_en}\nAssistant: {a_en}"

    # -------------------------------------------------------------------------
    # MAIN SAMPLE GENERATOR
    # -------------------------------------------------------------------------

    def generate_sample(self) -> str:
        """Procedurally generate one high-quality training sample in UKR, RUS, or ENG."""
        category = self.rng.choices(
            ["arithmetic", "equation", "word_problem", "code", "sql", "qa"],
            weights=[25, 15, 15, 25, 10, 10],
            k=1
        )[0]

        if category == "arithmetic":
            return self._gen_arithmetic()
        elif category == "equation":
            return self._gen_linear_equation()
        elif category == "word_problem":
            return self._gen_word_problem()
        elif category == "code":
            return self._gen_python_function()
        elif category == "sql":
            return self._gen_sql_query()
        else:
            return self._gen_qa()

    def generate_batch(self, count: int) -> list[str]:
        """Generate a batch of unique synthetic samples."""
        return [self.generate_sample() for _ in range(count)]

    def stream(self) -> Iterator[str]:
        """Infinite generator of synthetic text samples."""
        while True:
            yield self.generate_sample()


# -----------------------------------------------------------------------------
# PYTORCH DATASET WRAPPERS FOR DIRECT STREAMING
# -----------------------------------------------------------------------------

class ProceduralDataset(Dataset):
    """Memory-efficient PyTorch Dataset that generates tokenized synthetic samples deterministically on-the-fly."""

    def __init__(
        self,
        tokenizer: Any,
        total_samples: int = 100000,
        max_length: int = 256,
        seed: int = 42,
        languages: list[str] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.total_samples = total_samples
        self.max_length = max_length
        self.seed = seed
        self.languages = languages or ["uk", "ru", "en"]
        self.generator = SyntheticDataGenerator(seed=seed, languages=self.languages)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Seed generator based on sample index for reproducible deterministic retrieval
        rng_seed = (self.seed + index * 9973) % 2147483647
        self.generator.set_seed(rng_seed)
        text = self.generator.generate_sample()

        bos_id = getattr(self.tokenizer, "bos_token_id", 1)
        eos_id = getattr(self.tokenizer, "eos_token_id", 2)
        content_limit = self.max_length - 2
        content_tokens = self.tokenizer.encode(text)[:content_limit]
        token_ids = [bos_id] + content_tokens + [eos_id]

        if len(token_ids) < 2:
            token_ids = [bos_id, eos_id]

        inp = torch.tensor(token_ids[:-1], dtype=torch.long)
        tgt = torch.tensor(token_ids[1:], dtype=torch.long)
        return inp, tgt


class ProceduralIterableDataset(IterableDataset):
    """Infinite PyTorch IterableDataset for continuous streaming training over synthetic data."""

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 256,
        seed: int = 42,
        languages: list[str] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed
        self.languages = languages or ["uk", "ru", "en"]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        gen = SyntheticDataGenerator(seed=self.seed + worker_id * 1009, languages=self.languages)

        bos_id = getattr(self.tokenizer, "bos_token_id", 1)
        eos_id = getattr(self.tokenizer, "eos_token_id", 2)
        content_limit = self.max_length - 2

        while True:
            text = gen.generate_sample()
            content_tokens = self.tokenizer.encode(text)[:content_limit]
            token_ids = [bos_id] + content_tokens + [eos_id]
            if len(token_ids) < 2:
                token_ids = [bos_id, eos_id]
            inp = torch.tensor(token_ids[:-1], dtype=torch.long)
            tgt = torch.tensor(token_ids[1:], dtype=torch.long)
            yield inp, tgt
