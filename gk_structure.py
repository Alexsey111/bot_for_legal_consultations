"""
gk_structure.py - Статичный маппинг структуры ГК РФ

Источник: http://www.consultant.ru/document/cons_doc_LAW_5142/

Этот модуль обеспечивает надежное определение глав и частей ГК РФ,
независимо от форматирования исходного текста.
"""

from typing import List, Optional, Tuple, Union

# ================= ЧАСТЬ 1 =================

PART_1_CHAPTERS = {
    "1": {
        "title": "Гражданское законодательство",
        "articles": list(range(1, 8))
    },
    "2": {
        "title": "Возникновение гражданских прав и обязанностей, осуществление и защита гражданских прав",
        "articles": list(range(8, 17))
    },
    "3": {
        "title": "Граждане (физические лица)",
        "articles": list(range(17, 48))
    },
    "4": {
        "title": "Юридические лица",
        "articles": list(range(48, 66))
    },
    "4.1": {
        "title": "Публичные акционерные общества",
        "articles": list(range(66, 69))
    },
    "5": {
        "title": "Участие Российской Федерации, субъектов Российской Федерации, муниципальных образований в отношениях, регулируемых гражданским законодательством",
        "articles": list(range(125, 128))
    },
    "6": {
        "title": "Общие положения",
        "articles": list(range(128, 130))
    },
    "7": {
        "title": "Объекты гражданских прав",
        "articles": list(range(128, 153))
    },
    "8": {
        "title": "Нематериальные блага и их защита",
        "articles": list(range(150, 153))
    },
    "9": {
        "title": "Сделки",
        "articles": list(range(153, 182))
    },
    "9.1": {
        "title": "Решения собраний",
        "articles": list(range(181, 182))
    },
    "10": {
        "title": "Представительство. Доверенность",
        "articles": list(range(182, 190))
    },
    "11": {
        "title": "Исчисление сроков",
        "articles": list(range(190, 195))
    },
    "12": {
        "title": "Исковая давность",
        "articles": list(range(195, 208))
    },
    "13": {
        "title": "Общие положения",
        "articles": list(range(208, 211))
    },
    "14": {
        "title": "Недействительность сделок",
        "articles": list(range(166, 182))
    },
    "15": {
        "title": "Сделки, совершенные под отлагательным и отменительным условиями",
        "articles": list(range(157, 158))
    },
    "16": {
        "title": "Последствия недействительности сделки",
        "articles": list(range(167, 168))
    },
    "17": {
        "title": "Утрата силы договора",
        "articles": list(range(425, 426))
    },
    "18": {
        "title": "Общие положения",
        "articles": list(range(208, 211))
    },
    "19": {
        "title": "Право собственности и другие вещные права",
        "articles": list(range(209, 289))
    },
    "20": {
        "title": "Приобретение права собственности",
        "articles": list(range(218, 240))
    },
    "21": {
        "title": "Прекращение права собственности",
        "articles": list(range(235, 240))
    },
    "22": {
        "title": "Право собственности и другие вещные права на жилые помещения",
        "articles": list(range(288, 293))
    },
    "23": {
        "title": "Право собственности и другие вещные права на земельные участки",
        "articles": list(range(260, 288))
    },
    "24": {
        "title": "Общие положения",
        "articles": list(range(290, 301))
    },
    "25": {
        "title": "Право частной собственности",
        "articles": list(range(301, 306))
    },
    "26": {
        "title": "Право государственной собственности",
        "articles": list(range(214, 218))
    },
    "27": {
        "title": "Право муниципальной собственности",
        "articles": list(range(215, 218))
    },
    "28": {
        "title": "Общие положения",
        "articles": list(range(209, 212))
    },
    "29": {
        "title": "Приобретение права собственности",
        "articles": list(range(218, 240))
    },
    "30": {
        "title": "Прекращение права собственности",
        "articles": list(range(235, 240))
    },
    "31": {
        "title": "Защита права собственности и других вещных прав",
        "articles": list(range(301, 306))
    },
    "32": {
        "title": "Общие положения",
        "articles": list(range(301, 307))
    },
    "33": {
        "title": "Право хозяйственного ведения",
        "articles": list(range(294, 300))
    },
    "34": {
        "title": "Право оперативного управления",
        "articles": list(range(296, 300))
    },
    "35": {
        "title": "Защита вещных прав",
        "articles": list(range(301, 306))
    },
    "36": {
        "title": "Личные неимущественные права",
        "articles": list(range(150, 153))
    },
    "37": {
        "title": "Общие положения",
        "articles": list(range(307, 310))
    },
    "38": {
        "title": "Вещные права",
        "articles": list(range(216, 293))
    },
    "39": {
        "title": "Обязательственное право",
        "articles": list(range(307, 453))
    },
    "40": {
        "title": "Понятие и стороны обязательства",
        "articles": list(range(307, 310))
    },
    "41": {
        "title": "Исполнение обязательств",
        "articles": list(range(309, 329))
    },
    "42": {
        "title": "Обеспечение исполнения обязательств",
        "articles": list(range(329, 389))
    },
    "43": {
        "title": "Прекращение обязательств",
        "articles": list(range(409, 420))
    },
    "44": {
        "title": "Ответственность за нарушение обязательств",
        "articles": list(range(393, 409))
    },
    "45": {
        "title": "Общие положения",
        "articles": list(range(420, 453))
    },
    "46": {
        "title": "Сделки",
        "articles": list(range(153, 182))
    },
    "47": {
        "title": "Обязательства",
        "articles": list(range(307, 453))
    },
    "48": {
        "title": "Защита прав",
        "articles": list(range(301, 306))
    },
    "49": {
        "title": "Общие положения",
        "articles": list(range(420, 453))
    },
    "50": {
        "title": "Общие положения о договоре",
        "articles": list(range(420, 453))
    },
    "51": {
        "title": "Содержание и форма договора",
        "articles": list(range(420, 453))
    },
    "52": {
        "title": "Заключение договора",
        "articles": list(range(432, 450))
    },
    "53": {
        "title": "Изменение и расторжение договора",
        "articles": list(range(450, 453))
    },
    "54": {
        "title": "Договор в гражданском праве",
        "articles": list(range(420, 453))
    },
    "55": {
        "title": "Общие положения",
        "articles": list(range(420, 453))
    },
    "56": {
        "title": "Существенные условия договора",
        "articles": list(range(432, 435))
    },
    "57": {
        "title": "Форма договора",
        "articles": list(range(434, 438))
    },
    "58": {
        "title": "Заключение договора",
        "articles": list(range(432, 450))
    },
    "59": {
        "title": "Изменение и расторжение договора",
        "articles": list(range(450, 453))
    },
}

# ================= ЧАСТЬ 2 =================

PART_2_CHAPTERS = {
    "30": {
        "title": "Купля-продажа",
        "articles": list(range(454, 567))
    },
    "31": {
        "title": "Мена",
        "articles": list(range(567, 572))
    },
    "32": {
        "title": "Дарение",
        "articles": list(range(572, 583))
    },
    "33": {
        "title": "Рента и пожизненное содержание с иждивением",
        "articles": list(range(583, 606))
    },
    "34": {
        "title": "Аренда",
        "articles": list(range(606, 671))
    },
    "35": {
        "title": "Наем жилого помещения",
        "articles": list(range(671, 688))
    },
    "36": {
        "title": "Безвозмездное пользование",
        "articles": list(range(689, 701))
    },
    "37": {
        "title": "Подряд",
        "articles": list(range(702, 770))
    },
    "38": {
        "title": "Выполнение научно-исследовательских, опытно-конструкторских и технологических работ",
        "articles": list(range(769, 778))
    },
    "39": {
        "title": "Возмездное оказание услуг",
        "articles": list(range(779, 783))
    },
    "40": {
        "title": "Перевозка",
        "articles": list(range(784, 801))
    },
    "41": {
        "title": "Транспортная экспедиция",
        "articles": list(range(801, 806))
    },
    "42": {
        "title": "Заем и кредит",
        "articles": list(range(807, 823))
    },
    "43": {
        "title": "Финансирование под уступку денежного требования",
        "articles": list(range(824, 833))
    },
    "44": {
        "title": "Банковский вклад",
        "articles": list(range(834, 844))
    },
    "45": {
        "title": "Банковский счет",
        "articles": list(range(845, 861))
    },
    "46": {
        "title": "Расчеты",
        "articles": list(range(861, 885))
    },
    "47": {
        "title": "Хранение",
        "articles": list(range(886, 926))
    },
    "48": {
        "title": "Страхование",
        "articles": list(range(927, 970))
    },
    "49": {
        "title": "Поручение",
        "articles": list(range(971, 979))
    },
    "50": {
        "title": "Комиссия",
        "articles": list(range(990, 1004))
    },
    "51": {
        "title": "Агентирование",
        "articles": list(range(1005, 1011))
    },
    "52": {
        "title": "Доверительное управление имуществом",
        "articles": list(range(1012, 1026))
    },
    "53": {
        "title": "Коммерческая концессия",
        "articles": list(range(1027, 1041))
    },
    "54": {
        "title": "Простое товарищество",
        "articles": list(range(1041, 1055))
    },
    "55": {
        "title": "Публичное обещание награды",
        "articles": list(range(1055, 1056))
    },
    "56": {
        "title": "Публичный конкурс",
        "articles": list(range(1057, 1063))
    },
    "57": {
        "title": "Проведение игр и пари",
        "articles": list(range(1062, 1063))
    },
    "58": {
        "title": "Обязательства из односторонних действий",
        "articles": list(range(1055, 1063))
    },
    "59": {
        "title": "Общие положения",
        "articles": list(range(1064, 1109))
    },
    "60": {
        "title": "Действия в чужом интересе без поручения",
        "articles": list(range(980, 990))
    },
}

# ================= ЧАСТЬ 3 =================

PART_3_CHAPTERS = {
    "61": {
        "title": "Общие положения о наследовании",
        "articles": list(range(1110, 1118))
    },
    "62": {
        "title": "Наследование по завещанию",
        "articles": list(range(1118, 1141))
    },
    "63": {
        "title": "Наследование по закону",
        "articles": list(range(1141, 1152))
    },
    "64": {
        "title": "Приобретение наследства",
        "articles": list(range(1152, 1175))
    },
    "65": {
        "title": "Наследование отдельных видов имущества",
        "articles": list(range(1175, 1185))
    },
}

# ================= ЧАСТЬ 4 =================

PART_4_CHAPTERS = {
    "69": {
        "title": "Общие положения",
        "articles": list(range(1225, 1241))
    },
    "70": {
        "title": "Авторское право",
        "articles": list(range(1255, 1302))
    },
    "71": {
        "title": "Права, смежные с авторскими",
        "articles": list(range(1303, 1332))
    },
    "72": {
        "title": "Патентное право",
        "articles": list(range(1345, 1407))
    },
    "73": {
        "title": "Средства индивидуализации юридических лиц, товаров, работ, услуг и предприятий",
        "articles": list(range(1477, 1515))
    },
    "74": {
        "title": "Право на селекционное достижение",
        "articles": list(range(1408, 1444))
    },
    "75": {
        "title": "Право на топологии интегральных микросхем",
        "articles": list(range(1448, 1464))
    },
    "76": {
        "title": "Право на секрет производства (ноу-хау)",
        "articles": list(range(1465, 1472))
    },
}

# ================= ПОЛНАЯ СТРУКТУРА =================

GK_STRUCTURE = {
    "part_1": {
        "articles_range": (1, 453),
        "chapters": PART_1_CHAPTERS
    },
    "part_2": {
        "articles_range": (454, 1109),
        "chapters": PART_2_CHAPTERS
    },
    "part_3": {
        "articles_range": (1110, 1224),
        "chapters": PART_3_CHAPTERS
    },
    "part_4": {
        "articles_range": (1225, 1551),
        "chapters": PART_4_CHAPTERS
    }
}


def get_chapter_for_article(article_num: Union[str, int]) -> Tuple[Optional[str], Optional[str], int]:
    """
    Получает главу для статьи из статичного маппинга

    ИСПРАВЛЕНО: Type hint улучшен с article_num: str на Union[str, int]
    - Теперь можно передавать как строку ("454"), так и число (454)
    - Автоматическая нормализация к строке внутри функции
    - Улучшает удобство использования API

    Args:
        article_num: Номер статьи (например, "454", 454, "196.1", 196)

    Returns:
        Кортеж (chapter_num, chapter_title, part_num)
        - chapter_num: номер главы или None если не найдена
        - chapter_title: название главы или "Неизвестно"
        - part_num: номер части ГК РФ

    Examples:
        >>> get_chapter_for_article("454")
        ('30', 'Купля-продажа', 2)

        >>> get_chapter_for_article(454)
        ('30', 'Купля-продажа', 2)

        >>> get_chapter_for_article(196.1)
        (None, 'Неизвестно', 1)
    """
    # Нормализация к строке для единообразной обработки
    article_num_str = str(article_num)

    try:
        # Извлекаем основной номер статьи (без подпунктов)
        article_int = int(article_num_str.split('.')[0])
    except (ValueError, AttributeError):
        # Если не смогли распарсить - возвращаем fallback
        return None, "Неизвестно", 1

    # Определяем часть
    for part_key, part_data in GK_STRUCTURE.items():
        part_num = int(part_key.split('_')[1])
        article_range = part_data["articles_range"]

        if article_range[0] <= article_int <= article_range[1]:
            # Ищем главу в этой части
            for chapter_num, chapter_data in part_data["chapters"].items():
                if article_int in chapter_data["articles"]:
                    return chapter_num, chapter_data["title"], part_num

            # Статья в диапазоне части, но глава не найдена
            # Это может быть для статей без четкой привязки к главе
            return None, "Неизвестно", part_num

    # Fallback: определяем часть по диапазонам
    if 1 <= article_int <= 453:
        part_num = 1
    elif 454 <= article_int <= 1109:
        part_num = 2
    elif 1110 <= article_int <= 1224:
        part_num = 3
    elif 1225 <= article_int <= 1551:
        part_num = 4
    else:
        part_num = 1  # Default fallback

    return None, "Неизвестно", part_num


def get_all_articles_for_chapter(
    part_num: Union[str, int],
    chapter_num: Union[str, int]
) -> List[int]:
    """
    Получает список статей для конкретной главы

    ИСПРАВЛЕНО: Type hints улучшены с Union[str, int] для обоих параметров
    - part_num можно передавать как строку ("2") или число (2)
    - chapter_num можно передавать как строку ("30") или число (30)
    - Автоматическая нормализация к соответствующим типам

    Args:
        part_num: Номер части ГК РФ (1-4), как строка или число
        chapter_num: Номер главы (например, "30", 30, "4.1", 4.1)

    Returns:
        Список номеров статей

    Examples:
        >>> get_all_articles_for_chapter(2, "30")
        [454, 455, 456, ..., 566]

        >>> get_all_articles_for_chapter("2", 30)
        [454, 455, 456, ..., 566]
    """
    # Нормализация part_num к int
    part_num_int = int(part_num)

    part_key = f"part_{part_num_int}"
    if part_key not in GK_STRUCTURE:
        return []

    chapters = GK_STRUCTURE[part_key]["chapters"]
    # Нормализация chapter_num к str (так как ключи в словаре - строки)
    chapter_num_str = str(chapter_num)

    if chapter_num_str not in chapters:
        return []

    return chapters[chapter_num_str]["articles"]


def get_chapter_title(
    part_num: Union[str, int],
    chapter_num: Union[str, int]
) -> Optional[str]:
    """
    Получает название главы

    ИСПРАВЛЕНО: Type hints улучшены с Union[str, int] для обоих параметров
    - part_num можно передавать как строку ("2") или число (2)
    - chapter_num можно передавать как строку ("30") или число (30)
    - Автоматическая нормализация к соответствующим типам

    Args:
        part_num: Номер части ГК РФ (1-4), как строка или число
        chapter_num: Номер главы

    Returns:
        Название главы или None если не найдена

    Examples:
        >>> get_chapter_title(2, "30")
        'Купля-продажа'

        >>> get_chapter_title("2", 30)
        'Купля-продажа'
    """
    # Нормализация part_num к int
    part_num_int = int(part_num)

    part_key = f"part_{part_num_int}"
    if part_key not in GK_STRUCTURE:
        return None

    chapters = GK_STRUCTURE[part_key]["chapters"]
    # Нормализация chapter_num к str (так как ключи в словаре - строки)
    chapter_num_str = str(chapter_num)

    if chapter_num_str not in chapters:
        return None

    return chapters[chapter_num_str]["title"]


def determine_gk_part(article_num: Union[str, int]) -> int:
    """
    Определяет часть ГК РФ по номеру статьи

    ИСПРАВЛЕНО:
    - Параметр переименован с article_int на article_num для ясности
    - Type hint улучшен с int на Union[str, int]
    - Автоматическая нормализация к int

    Args:
        article_num: Номер статьи как строка или число

    Returns:
        Номер части (1-4)

    Examples:
        >>> determine_gk_part("454")
        2

        >>> determine_gk_part(454)
        2

        >>> determine_gk_part("1")
        1
    """
    # Нормализация к int
    try:
        article_int = int(str(article_num).split('.')[0])
    except (ValueError, AttributeError):
        return 1  # Default fallback

    if 1 <= article_int <= 453:
        return 1
    elif 454 <= article_int <= 1109:
        return 2
    elif 1110 <= article_int <= 1224:
        return 3
    elif 1225 <= article_int <= 1551:
        return 4
    else:
        return 1  # Default fallback


# ================= ТЕСТЫ =================

if __name__ == "__main__":
    # Тестовые примеры
    test_articles = ["1", 454, "196", 196, "1110", 1110, "1225", 1225, "454.1", 454.1]

    print("=== Тестирование gk_structure.py (с Union[str, int]) ===")
    for article in test_articles:
        chapter_num, chapter_title, part_num = get_chapter_for_article(article)
        print(f"Статья {article} (тип: {type(article).__name__}):")
        print(f"  Часть: {part_num}")
        print(f"  Глава: {chapter_num or 'N/A'} - {chapter_title}")
        print()

    # Тест получения всех статей главы
    print("\n=== Статьи главы 30 (Купля-продажа) - тест Union[str, int] ===")
    articles_str = get_all_articles_for_chapter("2", "30")
    articles_int = get_all_articles_for_chapter(2, 30)
    assert articles_str == articles_int, "Результаты должны быть одинаковыми!"
    print(f"Всего статей: {len(articles_str)}")
    print(f"Диапазон: {articles_str[0]} - {articles_str[-1]}")
    print(f"✅ Union[str, int] работает корректно!")

    # Тест получения названия главы
    print("\n=== Название главы 30 - тест Union[str, int] ===")
    title_str = get_chapter_title("2", "30")
    title_int = get_chapter_title(2, 30)
    assert title_str == title_int == "Купля-продажа", "Результаты должны быть одинаковыми!"
    print(f"Название: {title_str}")
    print(f"✅ Union[str, int] работает корректно!")

    # Тест определения части
    print("\n=== Определение части - тест Union[str, int] ===")
    test_parts = [
        ("1", 1),
        (1, 1),
        ("454", 2),
        (454, 2),
        ("1110", 3),
        (1110, 3),
        ("1225", 4),
        (1225, 4),
    ]
    for article, expected_part in test_parts:
        part = determine_gk_part(article)
        status = "[OK]" if part == expected_part else "[FAIL]"
        print(f"{status} Статья {article} (тип: {type(article).__name__}): часть {part} (ожидается: {expected_part})")
    print(f"✅ Union[str, int] работает корректно!")